import dgl
from scipy import ndimage
import numpy as np
import torch.nn as nn
from parts import *
import networkx as nx
import os

from utils import draw_mask_tile_single_view, draw_mask_tile_singleview_heatmap, windowing

class Initializer:

    def initialize(self, module):
        raise NotImplementedError("need subclassing to implement.")


class HeNorm(Initializer):

    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'fan_in')

    def initialize(self, module):
        def init_weights(m):
            if type(m) in [nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
                torch.nn.init.kaiming_normal_(m.weight, mode=self.mode)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.reset_parameters()

        module.apply(init_weights)

def pooling_dense_features(dense_outs, lungs, pooling_method='avg'):
    B, C = dense_outs.shape[0], dense_outs.shape[1]
    if pooling_method == 'global_avg':
        dense_outs_pool = F.adaptive_avg_pool3d(dense_outs, 1).view(B, C)
    elif pooling_method == 'global_max':
        dense_outs_pool = F.adaptive_max_pool3d(dense_outs, 1).view(B, C)
    else:
        # n_lung_volumes = tuple([int(pl.sum().item()) for pl in lungs])
        lungs_expand = lungs.expand_as(dense_outs)
        dense_outs_pool = (dense_outs * lungs_expand).view(B, C, -1).sum(dim=-1) \
                          / lungs_expand.view(B, C, -1).sum(dim=-1)

    return dense_outs_pool


class DC3D(nn.Module):

    def __init__(self, n_layers, in_ch_list, base_ch_list,
                 end_ch_list, out_ch, padding_list,
                 checkpoint_layers, dropout,
                 upsample_ksize=3, upsample_sf=2, kernel_sizes=None, stacking=0,
                 norm_method="bn", act_method='relu', pooling_method='avg', out_cls_ch=6):
        super(DC3D, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.padding_list = padding_list
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        if kernel_sizes is None:
            kernel_sizes = [3] * (self.n_layers * 2 + 1)
        self.kernel_sizes = kernel_sizes
        self.end_ch_list = end_ch_list
        self.upsample_ksize = upsample_ksize
        self.upsample_sf = upsample_sf
        self.checkpoint_layers = checkpoint_layers
        self.norm_method = norm_method
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.stacking = stacking
        self.out_cls_ch = out_cls_ch
        self.pooling_method = pooling_method
        conv_bias = True if self.norm_method is None else False
        self.ds_modules = nn.ModuleList(
            [
                ConvPoolBlock5d([in_ch_list[n], base_ch_list[n]],
                                [base_ch_list[n], end_ch_list[n]],
                                checkpoint_layers[n], self.kernel_sizes[n], conv_bias, padding_list[n],
                                2, 2, 0, norm_method=norm_method,
                                act_method=act_method,
                                dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers],
                              conv_bias, padding_list[n_layers],
                              dropout, norm_method=norm_method,
                              act_method=act_method)
        if (self.n_layers + 1) < len(in_ch_list):
            self.us_modules = nn.ModuleList(
                [
                    UpsampleConvBlock5d([in_ch_list[n_layers + 1 + n], base_ch_list[n_layers + 1 + n]],
                                        [base_ch_list[n_layers + 1 + n], end_ch_list[n_layers + 1 + n]],
                                        checkpoint_layers[n_layers + 1 + n], self.upsample_sf,
                                        self.kernel_sizes[n_layers + 1 + n], conv_bias, padding_list[n_layers + 1 + n],
                                        norm_method=norm_method, act_method=act_method, dropout=dropout)
                    for n in range(n_layers)
                ]
            )
        else:
            self.us_modules = None
        self.top_layer = nn.Conv3d(self.end_ch_list[self.n_layers + self.stacking],
                                   self.out_ch, kernel_size=1, padding=0)
        self.dummy = torch.ones(1, requires_grad=True)
        self.trace_path = None

    def init(self, initializer):
        initializer.initialize(self)

    def pooling_dense_features(self, dense_outs, lungs, pooling_method='avg'):
        return pooling_dense_features(dense_outs, lungs, pooling_method)

    def forward(self, x, lungs=None):
        ds_feat_list = [(x,)]
        for idx, ds in enumerate(self.ds_modules):
            if self.checkpoint_layers[idx] > 0:
                if idx == 0:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1], self.dummy))
                else:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1]))
            else:
                ds_feat_list.append(ds(ds_feat_list[-1][-1]))
        ds_feat_list.pop(0)
        if self.checkpoint_layers[self.n_layers] > 0:
            xbg = checkpoint(self.bg, ds_feat_list[-1][-1])
        else:
            xbg = self.bg(ds_feat_list[-1][-1])
        us_feat_list = [xbg]
        if self.us_modules is not None:
            for idx, (us, ds_feat) in enumerate(zip(self.us_modules, reversed(ds_feat_list))):
                if self.stacking == idx:
                    break
                if self.checkpoint_layers[self.n_layers + idx] > 0:
                    us_feat_list.append(checkpoint(us, us_feat_list[-1], ds_feat[0]))
                else:
                    us_feat_list.append(us(us_feat_list[-1], ds_feat[0]))
        outs = us_feat_list[-1]
        dense_outs = self.top_layer(outs)
        dense_outs = nn.Upsample(size=x.shape[-3:], mode='trilinear', align_corners=True)(dense_outs)
        return dense_outs, dense_outs


class PCM(nn.Module):

    def __init__(self, pool_size, in_ch, g_ch, f_dim, geo_f_dim, g_dim, non_local_iter, k_size,
                 merge_type='l2', self_loop=True, connectivity=2, residual=False, p_enc_dim=32):
        super(PCM, self).__init__()
        self.in_ch = in_ch
        self.g_ch = g_ch
        self.f_dim = f_dim
        self.g_dim = g_dim
        self.pool_size = pool_size
        self.merge_type = merge_type
        self.self_loop = self_loop
        self.non_local_iter = non_local_iter
        self.k_size = k_size
        self.connectivity = connectivity
        self.residual = residual
        self.p_enc_dim = p_enc_dim
        self.geo_f_dim = geo_f_dim
        if self.g_dim > 0:
            self.G = nn.Linear(g_ch, g_dim)
            self.r = nn.Linear(g_dim, g_ch)
        else:
            self.G = Identity()
            self.r = Identity()
            self.g_dim = g_ch

        if f_dim > 0:
            self.theta = nn.Linear(in_ch, f_dim)
            self.phi = nn.Linear(in_ch, f_dim)
        else:
            self.theta = Identity()
            self.phi = Identity()
            self.f_dim = in_ch

        if self.p_enc_dim > 0:
            if geo_f_dim > 0:
                self.geo_theta = nn.Linear(p_enc_dim, geo_f_dim)
                self.geo_phi = nn.Linear(p_enc_dim, geo_f_dim)
            else:
                self.geo_theta = Identity()
                self.geo_phi = Identity()
                self.geo_f_dim = p_enc_dim
        self.graph = None

    def build_geo_feature(self, x):
        spatial_size = x.shape[-3:]

        t = torch.ones(spatial_size).type(x.type())
        p = t.nonzero().type(x.type()).view(*spatial_size, len(spatial_size))
        # center_slices = tuple([slice(s // 2 - s // 4, s // 2 + s // 4) for s in spatial_size])
        # t[center_slices] = 0
        # p = p.permute(0, 1).view(len(spatial_size), *spatial_size).contiguous()
        pe = torch.zeros(self.p_enc_dim, *spatial_size).type(x.type())
        if self.p_enc_dim % (2 * len(spatial_size)) != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(self.p_enc_dim))
        d_model = int(self.p_enc_dim / len(spatial_size))
        c = 1e-4
        div = torch.pow(c, torch.arange(0., d_model, 2) / d_model).type(x.type())

        for d in range(len(spatial_size)):
            start = d * d_model
            end = (d + 1) * d_model
            pe[start:end:2, ::] = torch.sin(p[..., d].expand(len(div), *spatial_size)
                                            * div.view(len(div), *([1] * len(spatial_size))))

            pe[(start + 1):end:2, ::] = torch.cos(p[..., d].expand(len(div), *spatial_size)
                                                  * div.view(len(div), *([1] * len(spatial_size))))

        pe = pe.unsqueeze(0)
        pe = pe.expand(x.shape[0], pe.shape[1], *x.shape[-3:])
        return pe

    def init_graph(self, spatial_size, k_size):
        # find nn-k neighboring dependencies for each voxel.
        grid = np.zeros(spatial_size)
        grid[tuple([slice(k_size // 2, s - (k_size // 2)) for s in spatial_size])] = 1
        eff_node_locs = np.asarray(np.where(grid > 0)).T
        side_node_locs = np.asarray(np.where(grid == 0)).T
        node_locs = np.asarray(np.where(np.ones(spatial_size) > 0)).T

        g = nx.empty_graph(len(node_locs), create_using=nx.DiGraph)
        base_struct = ndimage.generate_binary_structure(3, connectivity=self.connectivity)
        base_struct = ndimage.zoom(base_struct, self.k_size / 3.0, order=0)
        loc_offset = np.asarray(np.where(base_struct > 0)).T - [self.k_size // 2, self.k_size // 2, self.k_size // 2]
        eff_node_neighbor_locs = np.expand_dims(eff_node_locs, 1) + np.expand_dims(loc_offset, 0)
        n_effect_node_locs, n_effect_node_neighbors, _ = eff_node_neighbor_locs.shape
        ravel_eff_node_neighbor_locs = np.ravel_multi_index(np.transpose(eff_node_neighbor_locs, (2, 0, 1))
                                                            .reshape(3, -1), spatial_size) \
            .reshape(n_effect_node_locs, n_effect_node_neighbors)

        side_node_neighbor_locs = np.expand_dims(side_node_locs, 1) + np.expand_dims(loc_offset, 0)
        ravel_side_node_neighbor_locs = [np.ravel_multi_index(np.asarray(
            [x for x in n_nodes if np.all(x >= 0) and
             all([xx < ss for xx, ss in zip(x, spatial_size)])]).T, spatial_size)
                                         for n_nodes in side_node_neighbor_locs]

        ravel_eff_node_locs = np.ravel_multi_index(eff_node_locs.T, spatial_size)
        ravel_side_node_locs = np.ravel_multi_index(side_node_locs.T, spatial_size)
        edge_maps = [(src.tolist(), np.repeat(dst, len(src)).tolist())
                     for src, dst in zip(ravel_eff_node_neighbor_locs, ravel_eff_node_locs)]
        side_edge_maps = [(src.tolist(), np.repeat(dst, len(src)).tolist())
                          for src, dst in zip(ravel_side_node_neighbor_locs, ravel_side_node_locs)]
        all_edge_maps = edge_maps + side_edge_maps
        src_ids, dst_ids = zip(*all_edge_maps)
        g.add_edges_from(list(zip([xx for x in src_ids for xx in x], [xx for x in dst_ids for xx in x])))
        graph = dgl.DGLGraph(g)
        if not self.self_loop:
            graph = dgl.transform.remove_self_loop(graph)
        return graph

    def merge_func(self, x_theta, x_phi, x_geo_theta, x_geo_phi):
        if self.merge_type == 'l2':
            f = torch.exp(5.0 * (-(x_theta - x_phi) ** 2))
            f_sm = f / f.sum(dim=-1, keepdim=True)
        elif self.merge_type == 'sm':
            f = torch.matmul(x_theta, x_phi)
            f_sm = F.softmax(f, dim=-1)
        elif self.merge_type == 'l2sm':
            f = torch.matmul(x_theta, x_phi)
            f = F.normalize(f, dim=-1)
            f_sm = F.softmax(f, dim=-1)
        elif self.merge_type == 'scaled_dot_product':
            f = torch.matmul(x_theta, x_phi)
            f_sm = F.softmax(f / np.sqrt(f.shape[-1]), dim=-1)
        elif self.merge_type == 'scaled_dot_product_relu':
            f = F.relu(torch.matmul(x_theta, x_phi))
            f_sm = F.softmax(f / np.sqrt(f.shape[-1]), dim=-1)
        elif self.merge_type == 'scaled_dot_product_geo':
            f = torch.matmul(x_theta, x_phi)
            f_geo = torch.matmul(x_geo_theta, x_geo_phi)
            f = f + f_geo
            f_sm = F.softmax(f / np.sqrt(f.shape[-1]), dim=-1)
        elif self.merge_type == 'scaled_dot_product_geo_relu':
            f = F.relu(torch.matmul(x_theta, x_phi))
            f_geo = torch.matmul(x_geo_theta, x_geo_phi)
            f = f + f_geo
            f_sm = F.softmax(f / np.sqrt(f.shape[-1]), dim=-1)
        elif self.merge_type == 'att_is_all':
            f = torch.matmul(x_theta + x_geo_theta, x_phi + x_geo_phi)
            f_sm = F.softmax(f / np.sqrt(f.shape[-1]), dim=-1)
        elif self.merge_type == 'smscaled':
            f = torch.matmul(x_theta, x_phi)
            f_sm = F.softmax(f / 0.01, dim=-1)
        elif self.merge_type == 'l2smrelu':
            f = torch.matmul(x_theta, x_phi)
            f = F.normalize(F.relu(f), dim=-1)
            f_sm = F.softmax(f, dim=-1)
        elif self.merge_type == 'cosine':
            f = F.cosine_similarity(x_theta.transpose(-1, -2), x_phi, dim=-2).unsqueeze(-2)
            f_sm = f / f.sum(dim=-1, keepdim=True)
        elif self.merge_type == 'smrelu':
            f = torch.matmul(x_theta, x_phi)
            f = F.relu(f)
            f_sm = F.softmax(f, dim=-1)
        elif self.merge_type == 'heu1':
            f = torch.matmul(x_theta, x_phi) / (
                        1.0 + torch.abs(x_theta.transpose(-1, -2) - x_phi).sum(dim=-2, keepdim=True))
            with torch.no_grad():
                mask_f = torch.ones_like(f)
                mask_f[f < 0.03] = 0.0
                f = f * mask_f
            f_sm = f / (1e-7 + f.sum(dim=-1, keepdim=True))
        elif self.merge_type == 'heu2':
            f = torch.matmul(x_theta, x_phi) / (
                        1.0 + torch.abs(x_theta.transpose(-1, -2) - x_phi).sum(dim=-2, keepdim=True))
            f = F.relu(f)
            f_sm = f / (1e-7 + f.sum(dim=-1, keepdim=True))
        else:
            raise NotImplementedError
        return f_sm

    def forward(self, cam, f, args=None):
        if self.graph is None:
            self.graph = self.init_graph(self.pool_size, self.k_size)
            self.graph = self.graph.to(f.device)
        g = self.graph

        f_flat = f.view(*f.shape[:-3], -1)
        f_flat = f_flat.permute(2, 0, 1).contiguous()
        g.ndata['f'] = f_flat
        if self.p_enc_dim > 0:
            geo_f = self.build_geo_feature(f).detach()
            geo_f_flat = geo_f.view(*geo_f.shape[:-3], -1)
            geo_f_flat = geo_f_flat.permute(2, 0, 1).contiguous()
            g.ndata['geo_f'] = geo_f_flat
        for i in range(self.non_local_iter):
            cam_flat = cam.view(*cam.shape[:-3], -1)
            cam_flat = cam_flat.permute(2, 0, 1).contiguous()
            g.ndata['cam'] = cam_flat
            g.update_all(self.message_func, self.reduce_func)
            refined_cam = g.ndata.pop('refined_cam')
            refined_cam = refined_cam.permute(1, 2, 0).contiguous()
            refined_cam = refined_cam.view(cam.shape)
            if self.residual:
                cam = refined_cam + cam
            else:
                cam = refined_cam
            g.ndata.pop('cam')
        g.ndata.pop('f')
        # # set 0 sums to background
        # zerosums = (cam.sum(dim=1, keepdim=True) < 1e-7).nonzero(as_tuple=True)
        # cam[:, :1,::][zerosums] = 1.0
        return cam

    def compute_cross_x(self, f_agg, geo_f_agg, f_feat, geo_f_feat, cam_agg):
        # appearance term
        # f_agg: node_batchs, edge_batches, batches, f_ch
        # f: node_batchs, batches, f_ch
        # cam: node_batchs, batches, 1
        node_batches = f_agg.shape[0]
        edge_batches = f_agg.shape[1]
        batches = f_agg.shape[2]
        f_ch = f_agg.shape[3]
        if self.p_enc_dim > 0:
            geo_f_ch = geo_f_agg.shape[3]
            x_geo_phi = self.geo_phi(geo_f_agg.view(-1, geo_f_ch)).reshape(node_batches, edge_batches, batches,
                                                                           self.geo_f_dim)
            x_geo_theta = self.geo_theta(geo_f_feat.view(-1, geo_f_ch)).reshape(node_batches, batches,
                                                                                self.geo_f_dim).unsqueeze(1)
            x_geo_theta = x_geo_theta.permute(2, 0, 1, 3).contiguous()
            x_geo_phi = x_geo_phi.permute(2, 0, 3, 1).contiguous()
        else:
            x_geo_theta = None
            x_geo_phi = None

        x_phi = self.phi(f_agg.view(-1, f_ch)).reshape(node_batches, edge_batches, batches, self.f_dim)
        x_theta = self.theta(f_feat.view(-1, f_ch)).reshape(node_batches, batches, self.f_dim).unsqueeze(1)
        x_theta = x_theta.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, 1, proj_dim]
        x_phi = x_phi.permute(2, 0, 3, 1).contiguous()  # [batches, node_batches, proj_dim, edge_batches]
        f_sm = self.merge_func(x_theta, x_phi, x_geo_theta, x_geo_phi)
        # f = torch.matmul(x_theta, x_phi) # [batches, node_batches, 1, edge_batches]
        # f = F.normalize(f, dim=-1, p=2)
        # f_sm = F.softmax(f, dim=-1)  # normalization
        x_g = self.G(cam_agg.view(-1, self.g_ch)).reshape(node_batches, edge_batches, batches, self.g_dim)
        x_g = x_g.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, edge_batches, g_dim]
        y = torch.matmul(f_sm, x_g).type(x_g.type())  # x_g is [batches, node_batches, 1, self.g_dim]
        cross_x = self.r(y.view(-1, self.g_dim)).reshape(batches, node_batches, self.g_ch)
        cross_x = cross_x.permute(1, 0, 2).contiguous()  # [node_batches, batches, in_ch]
        return cross_x

    def message_func(self, edges):
        f = edges.src['f']
        if self.p_enc_dim > 0:
            geo_f = edges.src['geo_f']
        cam = edges.src['cam']
        if self.p_enc_dim > 0:
            return {'f_agg': f, 'geo_f_agg': geo_f, 'cam_agg': cam}
        else:
            return {'f_agg': f, 'cam_agg': cam}

    def reduce_func(self, nodes):
        f_agg = nodes.mailbox['f_agg']
        cam_agg = nodes.mailbox['cam_agg']
        f = nodes.data['f']
        if self.p_enc_dim > 0:
            geo_f_agg = nodes.mailbox['geo_f_agg']
            geo_f = nodes.data['geo_f']
            refined_cam = self.compute_cross_x(f_agg, geo_f_agg, f, geo_f, cam_agg)
        else:
            refined_cam = self.compute_cross_x(f_agg, None, f, None, cam_agg)
        return {'refined_cam': refined_cam}



class DC3DATGeneric(nn.Module):

    def __init__(self, n_layers, in_ch_list, base_ch_list,
                 end_ch_list, out_ch, padding_list,
                 checkpoint_layers, dropout, at_spatial_size, at_f_dim, at_g_dim, at_p_enc_dim, at_geo_f_dim,
                 at_g_iter, at_k_size, at_merge_type, at_self_loop, at_layers,
                 upsample_ksize=3, upsample_sf=2, kernel_sizes=None, stacking=3,
                 norm_method="bn", act_method='relu', pooling_method='avg', out_cls_ch=6):
        super(DC3DATGeneric, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.padding_list = padding_list
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.at_spatial_size = at_spatial_size
        self.out_cls_ch = out_cls_ch
        if kernel_sizes is None:
            kernel_sizes = [3] * (self.n_layers * 2 + 1)
        self.kernel_sizes = kernel_sizes
        self.end_ch_list = end_ch_list
        self.upsample_ksize = upsample_ksize
        self.upsample_sf = upsample_sf
        self.checkpoint_layers = checkpoint_layers
        self.norm_method = norm_method
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.stacking = stacking
        self.pooling_method = pooling_method
        self.at_f_dim = at_f_dim
        self.at_g_dim = at_g_dim
        self.at_g_iter = at_g_iter
        self.at_k_size = at_k_size
        self.at_p_enc_dim = at_p_enc_dim
        self.at_geo_f_dim = at_geo_f_dim
        self.at_merge_type = at_merge_type
        self.at_self_loop = at_self_loop
        self.at_layers = at_layers
        conv_bias = True if self.norm_method is None else False
        self.ds_modules = nn.ModuleList(
            [
                ConvPoolBlock5d([in_ch_list[n], base_ch_list[n]],
                                [base_ch_list[n], end_ch_list[n]],
                                checkpoint_layers[n], self.kernel_sizes[n], conv_bias, padding_list[n],
                                2, 2, 0, norm_method=norm_method,
                                act_method=act_method,
                                dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers],
                              conv_bias, padding_list[n_layers],
                              dropout, norm_method=norm_method,
                              act_method=act_method)
        self.us_modules = nn.ModuleList(
            [
                UpsampleConvBlock5d([in_ch_list[n_layers + 1 + n], base_ch_list[n_layers + 1 + n]],
                                    [base_ch_list[n_layers + 1 + n], end_ch_list[n_layers + 1 + n]],
                                    checkpoint_layers[n_layers + 1 + n], self.upsample_sf,
                                    self.kernel_sizes[n_layers + 1 + n], conv_bias, padding_list[n_layers + 1 + n],
                                    norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.top_layer = nn.Conv3d(self.end_ch_list[self.n_layers + self.stacking], self.out_ch, kernel_size=1,
                                   padding=0)
        # self.cls_head = nn.Conv3d(self.end_ch_list[self.n_layers + self.stacking],
        #                self.out_cls_ch, kernel_size=1, padding=0)
        at_layers = [s for s in self.at_layers if s != -1]
        n_at_in_ch = self.at_f_dim * (len(self.at_layers) - 1) + 1 if -1 in self.at_layers \
            else self.at_f_dim * len(self.at_layers)

        self.reshape = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.end_ch_list[l_id], self.at_f_dim, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm3d(self.at_f_dim),
                nn.ReLU(inplace=True),
            ) for l_id in at_layers if l_id != -1
        ])
        self.attention_module = PCM(self.at_spatial_size, n_at_in_ch,
                                    self.out_ch, self.at_f_dim, self.at_geo_f_dim, self.at_g_dim, self.at_g_iter,
                                    self.at_k_size,
                                    self.at_merge_type, self.at_self_loop, p_enc_dim=self.at_p_enc_dim)
        self.dummy = torch.ones(1, requires_grad=True)
        # self.dummy_pcm = DummyPCM(self.out_ch, n_at_in_ch, self.at_f_dim)
        self.trace_path = None
        self.n_pcm_layer = 0

    def init(self, initializer):
        initializer.initialize(self)

    def pooling_dense_features(self, dense_outs, lungs, pooling_method='avg'):
        return pooling_dense_features(dense_outs, lungs, pooling_method)

    def apply_attention(self, x, lungs, dense_out, attention_features):

        # refine steps
        raw_spatial_size = dense_out.shape[2:]
        refined_dense_out = self.attention_module(F.interpolate(dense_out, size=self.at_spatial_size,
                                                                mode='trilinear', align_corners=True),
                                                  attention_features)
        refined_dense_out = F.interpolate(refined_dense_out, size=raw_spatial_size,
                                          mode='trilinear', align_corners=True)

        if self.trace_path is not None:
            trace_path_base, trace_metas = self.trace_path
            walk_trace_path = os.path.join(trace_path_base, 'apply_attention')
            if not os.path.exists(walk_trace_path):
                os.makedirs(walk_trace_path)
            for idx, (do, do_re) in enumerate(zip(dense_out, refined_dense_out)):
                uid = trace_metas['uid'][idx]
                original_size = trace_metas['original_size'][idx]
                do_np = np.squeeze(F.interpolate(do.detach().unsqueeze(0), size=original_size,
                                                 mode='trilinear', align_corners=True).cpu().numpy())
                do_re_np = np.squeeze(F.interpolate(do_re.detach().unsqueeze(0), size=original_size,
                                                    mode='trilinear', align_corners=True).cpu().numpy())
                x_np = np.squeeze(F.interpolate(x[idx].unsqueeze(0), size=original_size,
                                                mode='trilinear', align_corners=True).cpu().numpy())
                lung_np = np.squeeze(F.interpolate(lungs[idx].unsqueeze(0), size=original_size,
                                                   mode='nearest').cpu().numpy())
                scan_debug_path = os.path.join(walk_trace_path, f"{uid}")
                if not os.path.exists(scan_debug_path):
                    os.makedirs(scan_debug_path)
                debug_path = os.path.join(scan_debug_path, f"{self.at_g_iter}_{self.at_k_size}"
                f"_{self.at_merge_type}_{self.at_self_loop}_{self.at_f_dim}_{self.at_g_dim}")
                draw_mask_tile_singleview_heatmap(windowing(x_np, from_span=(0, 1)).astype(np.uint8),
                                                  [[(windowing(do_np, from_span=None) * lung_np).astype(np.uint8)],
                                                   [(windowing(do_re_np, from_span=None) * lung_np).astype(np.uint8)]],
                                                  do_re_np > 0, 5,
                                                  debug_path,
                                                  titles=["dram", "dram_refine"])

        return refined_dense_out

    def forward(self, x, lungs=None):
        ds_feat_list = [(x,)]
        attention_features = [] if -1 not in self.at_layers else [x]
        nc = 0
        for idx, ds in enumerate(self.ds_modules):
            if self.checkpoint_layers[idx] > 0:
                if idx == 0:
                    ds_feats = checkpoint(ds, ds_feat_list[-1][-1], self.dummy)
                else:
                    ds_feats = checkpoint(ds, ds_feat_list[-1][-1])
            else:
                ds_feats = ds(ds_feat_list[-1][-1])
            ds_feat_list.append(ds_feats)
            if idx in self.at_layers:
                attention_features.append(self.reshape[nc](ds_feats[0].detach()))
                nc += 1
        ds_feat_list.pop(0)
        if self.checkpoint_layers[self.n_layers] > 0:
            xbg = checkpoint(self.bg, ds_feat_list[-1][-1])
        else:
            xbg = self.bg(ds_feat_list[-1][-1])
        if self.n_layers in self.at_layers:
            attention_features.append(self.reshape[nc](xbg.detach()))
            nc += 1
        us_feat_list = [xbg]
        for idx, (us, ds_feat) in enumerate(zip(self.us_modules, reversed(ds_feat_list))):
            if self.stacking == idx:
                break
            if self.checkpoint_layers[self.n_layers + 1 + idx] > 0:
                us_feat = checkpoint(us, us_feat_list[-1], ds_feat[0])
            else:
                us_feat = us(us_feat_list[-1], ds_feat[0])
            us_feat_list.append(us_feat)
            if self.n_layers + idx + 1 in self.at_layers:
                attention_features.append(self.reshape[nc](us_feat.detach()))
                nc += 1
        outs = us_feat_list[-1]
        dense_outs = self.top_layer(outs)
        dense_outs = F.interpolate(dense_outs, size=x.shape[-3:], mode='trilinear', align_corners=True)
        # cls_dense_outs = self.cls_head(outs)
        # cls_dense_outs = F.interpolate(cls_dense_outs, size=x.shape[-3:], mode='trilinear', align_corners=True)
        attention_features = [F.interpolate(feat, size=self.at_spatial_size, mode='trilinear',
                                            align_corners=True) for feat in attention_features]

        attention_features = torch.cat(attention_features, dim=1)
        refined_dense_outs = self.apply_attention(x, lungs, dense_outs, attention_features)
        # refined_dense_outs = self.dummy_pcm(dense_outs, attention_features)
        return dense_outs, refined_dense_outs

