import torch
import torch.nn.functional as F
from data_transforms import Rescale3DOneShot, Flip3DOneShot, Rotate903DOneShot
import numpy as np
from itertools import permutations
import random
import os
from utils import windowing, draw_mask_tile_single_view

class BootBinCrossEntropy():

    def __init__(self, smoothing=0.1):
        super(BootBinCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.eps = 1e-7

    def __call__(self, p, t, voi, class_weights=None):
        # y : N1DHW
        # y_hat: sigmoid outputs
        assert (t.size() == p.size())
        assert (t.sum() <= voi.sum())
        # compute loss first part
        # outside t
        tb = voi < 1e-7
        po = p[tb]
        to = t[tb]
        pto = po * to + (1.0 - po) * (1.0 - to)
        ptoc = pto.clamp(self.eps, 1. - self.eps)
        o_nll_loss = - 1.0 * torch.log(ptoc)
        bceo_loss = o_nll_loss.mean()

        # part II, inside
        tf = voi > 0.0
        if tf.sum() > 0:
            pi = p[tf]
            ti = t[tf]
            alpha = (1.0 - ti.sum() / tf.sum()).clamp(0.25, 0.75)
            pti = pi * ti + (1.0 - pi) * (1.0 - ti)  # pt = p if t > 0 else 1-p
            w = alpha * ti + (1.0 - alpha) * (1.0 - ti)
            ptic = pti.clamp(self.eps, 1. - self.eps)
            i_nll_loss = - 1.0 * torch.log(ptic) * w
            bce_loss = i_nll_loss.sum() / w.sum()

            ti_hat = (pi > 0.5).float()
            pit_hat = pi * ti_hat + (1.0 - pi) * (1.0 - ti_hat)
            pit_hatc = pit_hat.clamp(self.eps, 1. - self.eps)
            nll_loss_hat = - 1.0 * torch.log(pit_hatc)
            boostrape_loss = nll_loss_hat.mean()
            return bceo_loss + (1.0 - self.smoothing) * bce_loss + self.smoothing * boostrape_loss
        else:
            return bceo_loss

class BinaryCrossEntropySmooth():

    def __init__(self, smooth):
        super(BinaryCrossEntropySmooth, self).__init__()
        self.smooth = smooth
        self.eps = 1e-6

    def __call__(self, probs, targets):
        # y : N1DHW
        # y_hat: N1DHW
        assert (probs.size() == targets.size())
        p = probs.view(-1)
        t = targets.view(-1)
        alpha = (1.0 - t.sum() / t.shape[0]).clamp(0.3, 0.7)
        p = p.clamp(self.eps, 1. - self.eps)
        pt = torch.log(p) * t + torch.log(1.0 - p) * (1.0 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1.0 - alpha) * (1.0 - t)  # w = alpha if t > 0 else 1-alpha
        nll_loss = -self.smooth * pt * w
        bce_loss = nll_loss.sum() / w.sum()
        return bce_loss


class IntRegLoss():
    ctss_ratio_map = {
        0: (0.0, 0.001),
        1: (0.001, 0.01),
        2: (0.01, 0.05),
        3: (0.05, 0.35),
        4: (0.35, 0.5),
        5: (0.5, 1.00001)
    }

    ctss_frequency_map = {
        0: 0.3,
        1: 0.25,
        2: 0.23,
        3: 0.2,
        4: 0.18,
        5: 0.15
    }

    def __init__(self, band_width=5e-2):
        super(IntRegLoss, self).__init__()
        self.band_width = band_width
        self.trace = False
        self.qwk = self.gen_qwk(6)

    def gen_qwk(self, N):
        qwk = np.zeros((N, N), dtype=np.float32)
        indices_comb = permutations(range(N), 2)
        for i, j in indices_comb:
            qwk[i, j] = (abs(i - j) + 1) ** 2 / (N ** 2)
        qwk += 1.0

        return qwk

    def ratio_to_label(self, ratios, ratio_map=None):
        if ratio_map is None:
            ratio_map = self.ctss_ratio_map
        inv_ratio_map = {v: k for k, v in ratio_map.items()}
        labels = [[inv_ratio_map[k] for k in inv_ratio_map.keys()
                   if k[0] <= ratio.item() and ratio.item() < k[1]][0] for ratio in ratios]
        return labels

    def get_one_label(self, ctss):
        ctss_lb, ctss_ub = self.ctss_ratio_map[int(float(ctss))]
        return ctss_lb, ctss_ub

    def get_labels(self, ctsses, lesion_ps):
        labels = []
        for ctss, lesion_p in zip(ctsses, lesion_ps):
            lb, ub = max(0.0, lesion_p.item() - self.band_width), min(1.0, lesion_p.item() + self.band_width)
            ctss_lb, ctss_ub = self.ctss_ratio_map[int(float(ctss))]
            label_band = (max(ctss_lb, lb)), (min(ctss_ub, ub))
            if label_band[1] < label_band[0]:
                if ub <= ctss_lb:
                    label_band = (lb, ub)
                elif lb >= ctss_ub:
                    label_band = (ctss_lb, ctss_ub)
                else:
                    raise RuntimeError("cannot reach here!")
            labels.append(label_band)

        labels = torch.FloatTensor(labels).cuda()
        return labels

    def get_label(self, ctss, lesion_p):
        lb, ub = max(0.0, lesion_p.item() - self.band_width), min(1.0, lesion_p.item() + self.band_width)
        ctss_lb, ctss_ub = self.ctss_ratio_map[int(float(ctss))]
        label_band = (max(ctss_lb, lb)), (min(ctss_ub, ub))
        if label_band[1] < label_band[0]:
            if ub <= ctss_lb:
                label_band = (lb, ub)
            elif lb >= ctss_ub:
                label_band = (ctss_lb, ctss_ub)
            else:
                raise RuntimeError("cannot reach here!")

        labels = torch.FloatTensor(label_band).cuda()
        return labels

    def compute_enc_loss(self, p):
        entropy_loss = ((-p * torch.log(p + 1e-7)) + (p - 1.0) * torch.log(1.0 - p + 1e-7)).mean()
        return entropy_loss

    def compute_reg_loss_with_probs(self, probs, lobes, lesion_candidates, ctsses, **kwargs):
        B = probs.shape[0]
        ratio_upper_bound = (lesion_candidates * lobes).view(B, 1, -1).sum(dim=-1) / lobes.view(B, 1, -1).sum(
            dim=-1)
        lobe_wise_probs = probs[lobes > 0]
        n_lobe_volumes = tuple([int(pl.sum().item()) for pl in lobes])
        lobe_wise_cams_batch = torch.split(lobe_wise_probs.view(-1), n_lobe_volumes, 0)
        pred_ratio = torch.stack([lb.mean() for lb in lobe_wise_cams_batch])
        regression_targets = self.get_labels(ctsses, ratio_upper_bound)
        _d = torch.cat([pred_ratio.unsqueeze(1), regression_targets], dim=1)
        K = (0.5 * (_d[:, 2] - _d[:, 1])) ** 2
        loss_unhinge = (_d[:, 0] - (_d[:, 2] + _d[:, 1]) / 2.0) ** 2 - K
        loss_unweight = torch.stack([torch.zeros_like(loss_unhinge), loss_unhinge]).max(0)[0]
        obj = kwargs.get('obj')
        cle_frequency_map = obj.ctss_frequency_map
        weight_factors = torch.Tensor([cle_frequency_map[int(float(cle))] for cle in ctsses]).float().cuda()
        weight_factors = torch.clamp(weight_factors, 0.2, 0.8)
        loss_reg = loss_unweight / weight_factors
        # loss_reg = F.smooth_l1_loss(pred_ratio, regression_targets)
        return loss_reg.sum()

    def compute_reg_loss_with_ratio(self, pred_ratio, ratio_upper_bound, ctss, **kwargs):
        regression_targets = self.get_label(ctss, ratio_upper_bound)
        K = (0.5 * (regression_targets[1] - regression_targets[0])) ** 2
        loss_unhinge = (pred_ratio - (regression_targets[1] + regression_targets[0]) / 2.0) ** 2 - K
        loss_unweight = torch.stack([torch.zeros_like(loss_unhinge), loss_unhinge]).max(0)[0]
        obj = kwargs.get('obj')
        cle_frequency_map = obj.ctss_frequency_map
        weight_factors = torch.Tensor([cle_frequency_map[int(float(ctss))]]).float().cuda()
        weight_factors = torch.clamp(weight_factors, 0.2, 0.8)[0]
        loss_reg = loss_unweight / weight_factors
        return loss_reg

    def compute_reg_loss(self, model, images, lobes, lesions, ctsses, **kwargs):
        dense_outs, _, _ = model(images, lobes)
        probs = F.sigmoid(dense_outs)
        loss = self.compute_reg_loss_with_probs(probs, lobes, lesions, ctsses, **kwargs)
        return loss, probs, dense_outs

    def before_call(self, model, **kwargs):
        model_trace_path = os.path.join(kwargs.get('obj').debug_path, "model_detailed_trace")
        epoch_debug_path = os.path.join(model_trace_path, f"epoch_{kwargs.get('obj').epoch_n}")
        if not os.path.exists(epoch_debug_path) and self.trace:
            os.makedirs(epoch_debug_path)
        model.trace_path = (epoch_debug_path, kwargs.get('metas'))

    def __call__(self, model, images, lobes, lesions, ctsses, **kwargs):
        self.before_call(model, **kwargs)
        _, dense_outs, cls_dense_outs = model(images, lobes)
        probs = F.sigmoid(dense_outs)
        reg_loss = self.compute_reg_loss_with_probs(probs, lobes, lesions, ctsses, **kwargs)
        enc_loss = self.compute_enc_loss(probs)
        return reg_loss, enc_loss


class IntRegAffLoss(IntRegLoss):
    def __init__(self, rescale_jitter, band_width=5e-2):
        super(IntRegAffLoss, self).__init__(band_width)
        self.rescale_jitter = rescale_jitter
        self.trace = False

    def get_affine_transform(self):
        rescale_jitter = self.rescale_jitter

        class _T(object):

            def aug_sampling(self, aug_list):
                return [x for x in aug_list if np.random.randint(0, 10) < (10 * 0.6)]

            def __init__(self):
                self.transform_pool = [
                    Rescale3DOneShot(rescale_jitter, None, mode='size'),
                    Flip3DOneShot(),
                    Rotate903DOneShot(),
                    # Rotate3DXOneShot(),
                ]
                all_p = list(permutations(self.transform_pool, 3))
                p = list(random.sample(all_p, 1)[0])
                self.p = self.aug_sampling(p)

            def __call__(self, sample):
                for _c in self.p:
                    sample = _c(sample)
                return sample

        return _T()

    def __call__(self, model, images, lobes, lesions, ctsses, **kwargs):
        self.before_call(model, **kwargs)
        # aff loss computation
        T = self.get_affine_transform()
        aff_images = T({
            "#image": images
        })["#image"]
        aff_lobes = T({
            "#reference": lobes
        })["#reference"].contiguous()
        aff_lesions = T({
            "#reference": lesions
        })["#reference"].contiguous()
        if self.trace:
            obj = kwargs["obj"]
            metas = kwargs["metas"]
            aff_debug_path = os.path.join(obj.debug_path, f"{self.__class__.__name__}")

            for idx, (image, lobe, lesion, aff_image, aff_lobe,
                      aff_lesion) \
                    in enumerate(zip(images, lobes, lesions,
                                     aff_images, aff_lobes, aff_lesions)):
                aff_debug_path_pair = os.path.join(aff_debug_path, metas['uid'][idx])
                if not os.path.exists(aff_debug_path_pair):
                    os.makedirs(aff_debug_path_pair)
                i_np = image.squeeze(0).cpu().numpy()
                l_np = lobe.squeeze(0).cpu().numpy()
                le_np = lesion.squeeze(0).cpu().numpy()
                draw_mask_tile_single_view(windowing(i_np, from_span=(0, 1)),
                                           [[(l_np > 0).astype(np.uint8)],
                                            [(le_np > 0).astype(np.uint8)]],
                                           l_np > 0, 5,
                                           aff_debug_path_pair + f'/original',
                                           colors=[(0, 0, 255)], thickness=[-1],
                                           alpha=0.3)
                i_np = aff_image.squeeze(0).cpu().numpy()
                l_np = aff_lobe.squeeze(0).cpu().numpy()
                le_np = aff_lesion.squeeze(0).cpu().numpy()
                draw_mask_tile_single_view(windowing(i_np, from_span=(0, 1)),
                                           [[(l_np > 0).astype(np.uint8)],
                                            [(le_np > 0).astype(np.uint8)]],
                                           l_np > 0, 5,
                                           aff_debug_path_pair + f'/aff',
                                           colors=[(0, 0, 255)], thickness=[-1],
                                           alpha=0.3)

                with open(aff_debug_path_pair + '/aff_params.txt', "wt", newline='') as fp:
                    lines = []
                    for p in T.p:
                        lines.append(f"{p.__dict__}\r\n")
                    fp.writelines(lines)

        reg_loss, probs, _ = self.compute_reg_loss(model, images, lobes, lesions, ctsses, **kwargs)
        enc_loss = self.compute_enc_loss(probs)
        probs_T = T({
            "#image": probs
        })["#image"]

        aff_reg_loss, aff_probs, _ = self.compute_reg_loss(model, aff_images, aff_lobes, aff_lesions, ctsses,
                                                           **kwargs)
        aff_lobes_exp = aff_lobes.expand_as(probs_T)
        aff_loss = F.smooth_l1_loss(probs_T[aff_lobes_exp > 0], aff_probs[aff_lobes_exp > 0])
        ce_loss = (reg_loss + aff_reg_loss) / 2.0
        return ce_loss, aff_loss, enc_loss


class IntRegRefineLoss(IntRegLoss):

    def __init__(self, band_width=1e-2,
                 smoothing=0.1,
                 refine_method='th',
                 config_param={}):
        super(IntRegRefineLoss, self).__init__(band_width)
        self.smoothing = smoothing
        self.trace = False
        self.refine_method = refine_method
        self.config_param = config_param

        self.bootstrap_loss = BootBinCrossEntropy(smoothing)

    def threshold_postprocessing(self, pred_np, lobe_np, lesion_np, ctss):
        cand_np = np.logical_and(pred_np > 0, lesion_np > 0).astype(np.uint8)
        if float(ctss) < 1e-7:
            cand_np = np.zeros_like(cand_np)
        return cand_np

    def compute_seg_loss(self, dense_outs, refined_dense_outs, images, lobes,
                         lesions, scores, metas, obj, tag='fixed'):
        probs = F.sigmoid(dense_outs).detach()
        refine_probs = F.sigmoid(refined_dense_outs).detach()
        pseudo_refs = torch.zeros_like(lobes)
        for idx, (dense_out, image, lobe, lesion, prob, refine_prob, score) \
                in enumerate(zip(dense_outs, images, lobes, lesions, probs, refine_probs, scores)):
            lobe_np = lobe.cpu().squeeze(0).numpy()
            lesion_np = lesion.cpu().squeeze(0).numpy()
            prob_np = prob.cpu().squeeze(0).numpy()
            refine_prob_np = refine_prob.cpu().squeeze(0).numpy()
            prob_np[lobe_np == 0] = 0.0
            refine_prob_np[lobe_np == 0] = 0.0

            pred_np = prob_np > 0.5
            if self.refine_method == 'th':
                pseudo_ref = self.threshold_postprocessing(pred_np, lobe_np, lesion_np, score)
            else:
                raise NotImplementedError(f"Do not support refine method :{self.refine_method}!")

            pseudo_t = torch.from_numpy(pseudo_ref).long()
            pseudo_refs[idx, 0, ...] = pseudo_t

        t = pseudo_refs.long()
        p = F.sigmoid(refined_dense_outs)
        voi = lobes > 0
        loss = self.bootstrap_loss(p, t, voi)
        return loss

    def __call__(self, model, images, lobes, lesions, ctsses, **kwargs):
        self.before_call(model, **kwargs)
        dense_outs, refined_dense_outs = model(images, lobes)
        probs = F.sigmoid(dense_outs)
        reg_loss = self.compute_reg_loss_with_probs(probs, lobes, lesions,
                                                    ctsses, **kwargs)

        # compute refine segmentation loss
        metas = kwargs["metas"]
        obj = kwargs["obj"]
        # self.bootstrap_loss.beceloss.set_epoch(obj.epoch_n)
        seg_loss = self.compute_seg_loss(dense_outs, refined_dense_outs, images, lobes,
                                         lesions, ctsses, metas, obj)
        return reg_loss, seg_loss


class IntRegAffRefineLoss(IntRegLoss):

    def __init__(self, rescale_jitter, band_width=5e-2,
                 smoothing=0.05,
                 refine_method='th', config_param={}):
        super(IntRegAffRefineLoss, self).__init__(band_width)
        self.rescale_jitter = rescale_jitter
        self.trace = False

        self.seg_loss = IntRegRefineLoss(band_width,
                                         smoothing,
                                         config_param=config_param,
                                         refine_method=refine_method)
        self.seg_loss.trace = self.trace

    def get_affine_transform(self):
        rescale_jitter = self.rescale_jitter

        class _T(object):

            def aug_sampling(self, aug_list):
                return [x for x in aug_list if np.random.randint(0, 10) < (10 * 0.5)]

            def __init__(self):
                self.transform_pool = [
                    Rescale3DOneShot(rescale_jitter, None, mode='size'),
                    Flip3DOneShot(),
                    Rotate903DOneShot(),
                    # Rotate3DXOneShot(),
                ]
                all_p = list(permutations(self.transform_pool, 3))
                p = list(random.sample(all_p, 1)[0])
                self.p = self.aug_sampling(p)

            def __call__(self, sample):
                for _c in self.p:
                    sample = _c(sample)
                return sample

        return _T()

    def __call__(self, model, images, lobes, lesions, ctsses, **kwargs):
        self.before_call(model, **kwargs)
        # aff loss computation
        T = self.get_affine_transform()
        aff_images = T({
            "#image": images
        })["#image"]
        aff_lobes = T({
            "#reference": lobes
        })["#reference"].contiguous()
        aff_lesions = T({
            "#reference": lesions
        })["#reference"].contiguous()
        metas = kwargs["metas"]
        obj = kwargs["obj"]

        dense_outs, refined_dense_outs, cls_dense_outs = model(images, lobes)
        probs = F.sigmoid(dense_outs)
        reg_loss = self.compute_reg_loss_with_probs(probs, lobes, lesions, ctsses, **kwargs)
        probs_T = T({
            "#image": probs
        })["#image"]
        cls_dense_outs_T = T({
            "#image": cls_dense_outs
        })["#image"]
        aff_dense_outs, aff_refined_dense_outs, aff_cls_dense_outs = model(aff_images, aff_lobes)
        aff_probs = F.sigmoid(aff_dense_outs)
        aff_reg_loss = self.compute_reg_loss_with_probs(aff_probs, aff_lobes,
                                                        aff_lesions,
                                                        ctsses, **kwargs)
        aff_lobes_exp = aff_lobes.expand_as(probs_T)
        aff_loss = F.smooth_l1_loss(probs_T[aff_lobes_exp > 0], aff_probs[aff_lobes_exp > 0])
        aff_lobes_exp_cls = aff_lobes.expand_as(cls_dense_outs_T)
        aff_loss_cls = F.smooth_l1_loss(cls_dense_outs_T[aff_lobes_exp_cls > 0],
                                        aff_cls_dense_outs[aff_lobes_exp_cls > 0])
        all_reg_loss = (reg_loss + aff_reg_loss) / 2.0
        all_aff_loss = (aff_loss + aff_loss_cls) / 2.0
        # compute seg loss
        seg_loss = self.seg_loss.compute_seg_loss(dense_outs, refined_dense_outs, images, lobes,
                                                  lesions, ctsses, metas, obj, "fixed")
        seg_aff_loss = self.seg_loss.compute_seg_loss(aff_dense_outs, aff_refined_dense_outs,
                                                      aff_images, aff_lobes, aff_lesions,
                                                      ctsses, metas, obj, "aff")

        all_seg_loss = (seg_loss + seg_aff_loss) / 2.0
        return all_reg_loss, all_aff_loss, all_seg_loss
