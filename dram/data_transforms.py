import numpy as np
import copy
from scipy import ndimage
from scipy.ndimage.interpolation import affine_transform
from skimage import exposure
from itertools import combinations, permutations
import random
import math
from utils import resample, windowing
import torch.nn.functional as F
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors if # sign in its keys."""

    def __call__(self, sample, is_pin=False):
        if is_pin:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()).pin_memory() if "#" in k else v)
                      for k, v in sample.items()}
        else:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()) if "#" in k else v)
                      for k, v in sample.items()}
        return sample

class RemoveMeta(object):
    """Convert ndarrays in sample to Tensors if # sign in its keys."""

    def __call__(self, sample, keep_keys=("uid", "size", "spacing", "slices", "crop_slices",
                                          'original_spacing', 'original_size', "origin",
                                          "direction", "cle", "pse")):
        d = copy.deepcopy(sample['meta'])
        [d.pop(kk, None) for kk in sample['meta'].keys() if kk not in keep_keys]
        sample['meta'] = d
        return sample

class Windowing(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, min=-1200, max=600, out_min=0, out_max=1):
        self.min = min
        self.max = max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, sample):
        from_span = (self.min, self.max) if self.min is not None else None
        sample = {(k if "#" in k and "image" in k else k): (windowing(v.astype(np.float32),
                                                                      from_span=from_span,
                                                                      to_span=(self.out_min,
                                                                               self.out_max))
                                                            if "#" in k and "image" in k else v)
                  for k, v in sample.items()}
        return sample

class Resample(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, mode, factor, size=None):
        self.mode = mode
        self.factor = factor
        if size:
            self.size = list(size)

    def __call__(self, sample):
        new_sample = {"meta": {}}
        spacing = sample['meta']['spacing']
        if self.mode == 'random_spacing':
            factor = np.random.uniform(self.factor[0], self.factor[1])
            require_spacing = [factor] * len(spacing)
            new_size = None
        elif self.mode == 'fixed_factor':
            factor = self.factor
            require_spacing = None
            new_size = None
        elif self.mode == 'fixed_spacing':
            if isinstance(self.factor, (float, int)):
                factor = self.factor
                require_spacing = [factor] * len(spacing)
            elif isinstance(self.factor, (tuple, list)):
                require_spacing = self.factor
                factor = 2  # dummy number meaningless.
            new_size = None
        elif self.mode == "inplane_spacing_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], self.factor[1],
                               self.factor[2]]
            new_size = None
            factor = 2
        elif self.mode == "inplane_resolution_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [current_size[0], self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_jittering":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            z_spacing_base = spacing[0]
            offset = np.random.uniform(-self.factor, self.factor)
            z_spacing = z_spacing_base + offset
            require_spacing = [z_spacing, spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / z_spacing)), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_min_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if spacing[0] < self.factor[0]:
                print("set spacing to {} from {}.".format(self.factor[0], spacing[0]))
                require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                            self.size[2]]
            else:
                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
            factor = 2
        elif self.mode == "fixed_spacing_min_in_plane_resolution":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if not isinstance(self.factor, (tuple, list)):
                factor = [self.factor] * 3
            else:
                factor = self.factor
            new_y_size = int(round(current_size[1] * spacing[1] / factor[1]))
            if new_y_size > self.size[1]:

                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
                print(
                    "new_size:{} > target_size {}. fixed_in_plane_resolution mode. {}.".format(new_y_size, self.size[1],
                                                                                               new_size))
            else:
                require_spacing = [spacing[0], factor[1],
                                   factor[2]]
                new_size = None
                print(
                    "new_size:{} <= target_size {}. fixed_spacing. {}.".format(new_y_size, self.size[1],
                                                                               require_spacing))
            factor = 2
        elif self.mode == "iso_minimal":
            factor = spacing[0]
            require_spacing = [np.min(spacing)] * len(spacing)
            new_size = None
        elif self.mode == "fixed_output_size":
            current_size = sample['meta']['size']
            ratio = current_size[-1] / self.size[-1]
            require_spacing = [spacing[-1] * ratio] * len(spacing)
            new_size = self.size[:]
            new_size[0] = int(round(current_size[0] * spacing[0] / require_spacing[0]))
            new_size[1] = int(round(current_size[1] * spacing[1] / require_spacing[1]))
            factor = 2
        elif self.mode == "fixed_size":
            current_size = sample['meta']['size']
            ratios = np.asarray(current_size) / np.asarray(self.size)
            require_spacing = (spacing * ratios).tolist()
            new_size = self.size[:]
            factor = 2
        elif self.mode == "spacing_size_match":
            require_spacing = self.factor[:]
            new_size = self.size[:]
            factor = 2
        else:
            raise NotImplementedError
        for k, v in sample.items():
            if "#" in k:
                if "reference" in k or 'weight_map' in k:
                    mode = 'nearest'
                else:
                    mode = 'linear'
                if v.ndim == 4:
                    r_results = [resample(vv, spacing, factor=factor,
                                          required_spacing=require_spacing,
                                          new_size=new_size, interpolator=mode) for vv
                                 in v]
                    new_spacing = r_results[0][-1]
                    nv = np.stack([r[0] for r in r_results], axis=0)
                elif v.ndim == 3:
                    nv, new_spacing = resample(v, spacing, factor=factor,
                                               required_spacing=require_spacing,
                                               new_size=new_size, interpolator=mode)
                else:
                    raise NotImplementedError
                new_sample[k] = nv
                new_size = nv.shape
            else:
                new_sample[k] = v
        old_size = sample['meta']['size']
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta']['spacing'] = tuple(new_spacing)
        new_sample['meta']['size'] = new_size
        new_sample['meta']['size_before_resample'] = old_size
        new_sample['meta']['resample_factor'] = factor
        return new_sample

class IntensityInverse:

    def __init__(self, channel_dim=0):
        self.channel_dim = channel_dim
        self.epsilon = 1e-7

    def _intensity_inverse(self, data):
        d_min = data.min()
        d_max = data.max()
        d_range = d_max - d_min
        data_rescale = (data - d_min) / float(d_range + self.epsilon)
        data_rescale = 1.0 - data_rescale
        # rescale back to the original range
        data = (data_rescale - data_rescale.min()) * d_range + d_min
        return data

    def intensity_inverse(self, data):
        if not self.channel_dim:
            data = self._intensity_inverse(data)
        else:
            data_c_list = [self._intensity_inverse(data.take(c, axis=self.channel_dim))
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        new_meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.intensity_inverse(v.astype(np.float32))
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_meta.update({
            "channel_dim": self.channel_dim
        })
        new_sample['meta'][self.__class__.__name__] = copy.deepcopy(new_meta)
        return new_sample


class HistogramEqual:

    def __init__(self, channel_dim=0):
        self.epsilon = 1e-7
        self.channel_dim = channel_dim

    def _he_transform(self, data, channel_id, meta):
        data = exposure.equalize_hist(data)
        return data

    def he_transform(self, data, meta):
        if not self.channel_dim:
            data = self._he_transform(data, -1, meta)
        else:
            data_c_list = [self._he_transform(data.take(c, axis=self.channel_dim), c, meta)
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        new_meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.he_transform(v.astype(np.float32), new_meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        return new_sample


class GammaTransform:

    def __init__(self, gamma_range=(0.5, 2), channel_dim=0):
        self.gamma_range = gamma_range
        self.epsilon = 1e-7
        self.channel_dim = channel_dim

    def _gamma_transform(self, data, channel_id, meta):
        d_min = data.min()
        d_max = data.max()
        d_range = d_max - d_min
        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        meta['gamma'] = factor
        data = (((data - d_min) / float(d_range + self.epsilon)) ** factor) * d_range + d_min
        meta[channel_id] = factor
        return data

    def gamma_transform(self, data, meta):
        if not self.channel_dim:
            data = self._gamma_transform(data, -1, meta)
        else:
            data_c_list = [self._gamma_transform(data.take(c, axis=self.channel_dim), c, meta)
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        new_meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.gamma_transform(v.astype(np.float32), new_meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_meta.update({
            "gamma_range": tuple(self.gamma_range),
        })
        new_sample['meta'][f"{self.__class__.__name__}_gamma"] = new_meta['gamma']
        return new_sample


class ContrastStretchingTransform:

    def __init__(self, gamma_range=(0.5, 2), middle_point=(0.3, 0.7), channel_dim=0):
        self.gamma_range = gamma_range
        self.middle_point = middle_point
        self.epsilon = 1e-7
        self.channel_dim = channel_dim

    def _transform(self, data, channel_id, meta):
        d_min = data.min()
        d_max = data.max()
        d_range = d_max - d_min
        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        mp = np.random.uniform(self.middle_point[0], self.middle_point[1])
        # 1./(1 + (m./(f + eps)).^E)
        f = ((data - d_min) / float(d_range + self.epsilon))
        d = 1.0 / (1.0 + ((mp / (f + self.epsilon)) ** factor))
        data = d * d_range + d_min
        meta[channel_id] = (factor, mp)
        meta['factor'] = factor
        meta['mp'] = mp
        return data

    def transform(self, data, meta):
        if not self.channel_dim:
            data = self._transform(data, -1, meta)
        else:
            data_c_list = [self._transform(data.take(c, axis=self.channel_dim), c, meta)
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.transform(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # meta.update({
        #     "gamma_range": tuple(self.gamma_range),
        #     "middle_point": tuple(self.middle_point)
        # })
        # new_sample['meta'][f"{self.__class__.__name__}_factor"] = meta['factor']
        # new_sample['meta'][f"{self.__class__.__name__}_mp"] = meta['mp']
        return new_sample


class GaussianAddictive:

    def __init__(self, sigma, channel_dim=0):
        self.sigma = sigma
        self.channel_dim = channel_dim
        self.epsilon = 1e-7

    def _gaussian_addictive(self, data, channel_id, meta):
        variance_v = np.random.uniform(self.sigma[0], self.sigma[1])
        meta['sigma'] = variance_v
        d_min = data.min()
        d_max = data.max()
        d_range = d_max - d_min
        # rescale to 0-1
        data_rescale = ((data - d_min) / float(d_range + self.epsilon))
        # data_rescale_mean = np.mean(data_rescale)
        noise = np.random.normal(0, variance_v, size=data.shape)
        data_rescale += noise
        data_rescale[data_rescale < 0] = 0.0
        data_rescale[data_rescale > 1] = 1.0
        # rescale back to the original range
        data = data_rescale * d_range + d_min
        meta[channel_id] = variance_v
        return data

    def gaussian_addictive(self, data, meta):
        if not self.channel_dim:
            data = self._gaussian_addictive(data, -1, meta)
        else:
            data_c_list = [self._gaussian_addictive(data.take(c, axis=self.channel_dim), c, meta)
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.gaussian_addictive(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # new_sample['meta'][f"{self.__class__.__name__}_sigma"] = meta['sigma']
        return new_sample


class MinimalIntensityProjection:

    def __init__(self, slab_thickness=(3, 10), angle=(0, 3)):
        self.slab_thickness = slab_thickness
        self.angle = angle
        self.epsilon = 1e-7

    def maip(self, data, meta):
        assert data.ndim == 3
        slab_thickness = np.random.randint(self.slab_thickness[0], self.slab_thickness[1])
        angle = np.random.randint(self.angle[0], self.angle[1])
        meta['slab_thickness'] = slab_thickness
        meta['angle'] = angle
        projection = np.zeros_like(data)
        projection = np.moveaxis(projection, angle, 0)
        for si in range(data.shape[angle]):
            start = max(0, si - slab_thickness)
            p = np.min(np.stack([data.take(n, axis=angle)
                                 for n in range(start, si + 1)], axis=angle), angle)
            projection[si, ::] = p
        projection = np.moveaxis(projection, 0, angle)
        return projection

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.maip(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_slab_thickness"] = meta['slab_thickness']
        new_sample['meta'][f"{self.__class__.__name__}_angle"] = meta['angle']
        return new_sample


class MinimalIntensityAxialProjection:

    def __init__(self, slab_thickness=(3, 10)):
        self.slab_thickness = slab_thickness
        self.epsilon = 1e-7

    def maip(self, data, meta):
        assert data.ndim == 3
        spacing = meta['spacing']
        slab_thickness = np.random.randint(self.slab_thickness[0], self.slab_thickness[1])
        axial_thickness = int(slab_thickness / spacing[0])
        meta['axial_thickness'] = axial_thickness
        projection = np.zeros_like(data)
        for si in range(data.shape[0]):
            start = max(0, si - slab_thickness)
            p = np.min(np.stack([data.take(n, axis=0)
                                 for n in range(start, si + 1)], axis=0), 0)
            projection[si, ::] = p
        return projection

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.maip(v.astype(np.float32), sample['meta'])
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # new_sample['meta'][f"{self.__class__.__name__}_axial_thickness"] = meta['axial_thickness']
        return new_sample


class MaximumIntensityProjection:

    def __init__(self, slab_thickness=(3, 10), angle=(0, 3)):
        self.slab_thickness = slab_thickness
        self.angle = angle
        self.epsilon = 1e-7

    def maip(self, data, meta):
        assert data.ndim == 3
        slab_thickness = np.random.randint(self.slab_thickness[0], self.slab_thickness[1])
        angle = np.random.randint(self.angle[0], self.angle[1])
        meta['slab_thickness'] = slab_thickness
        meta['angle'] = angle
        projection = np.zeros_like(data)
        projection = np.moveaxis(projection, angle, 0)
        for si in range(data.shape[angle]):
            start = max(0, si - slab_thickness)
            p = np.max(np.stack([data.take(n, axis=angle)
                                 for n in range(start, si + 1)], axis=angle), angle)
            projection[si, ::] = p
        projection = np.moveaxis(projection, 0, angle)
        return projection

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.maip(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # new_sample['meta'][f"{self.__class__.__name__}_slab_thickness"] = meta['slab_thickness']
        # new_sample['meta'][f"{self.__class__.__name__}_angle"] = meta['angle']
        return new_sample


class GaussianBlur:

    def __init__(self, sigma, mode='fixed'):
        self.sigma = sigma
        self.mode = mode

    # def _gaussian_blur(self, data, channel_id, meta):
    #     variance_v = np.random.uniform(self.sigma[0], self.sigma[1])
    #     data = ndimage.gaussian_filter(data, variance_v)
    #     meta[channel_id] = variance_v
    #     return data

    def gaussian_blur(self, data, meta):
        if self.mode == 'fixed':
            variance_v = self.sigma[0]
        else:
            variance_v = np.random.uniform(self.sigma[0], self.sigma[1])
        data = ndimage.gaussian_filter(data, variance_v)
        meta["sigma"] = variance_v
        return data

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.gaussian_blur(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # new_sample['meta'][f"{self.__class__.__name__}_sigma"] = meta['sigma']
        return new_sample


class ContrastJitter:

    def __init__(self, jitter_range=(0.75, 1.25), if_keep_range=True, channel_dim=0):
        self.jitter_range = jitter_range
        self.if_keep_range = if_keep_range
        self.channel_dim = channel_dim

    def _contrast_jitter(self, data, channel_id, meta):
        d_mean = data.mean()
        d_min = data.min()
        d_max = data.max()
        factor = np.random.uniform(self.jitter_range[0], self.jitter_range[1])
        data = (data - d_mean) * factor + d_mean
        if self.if_keep_range:
            data[data < d_min] = d_min
            data[data > d_max] = d_max

        meta[channel_id] = factor
        return data

    def contrast_jitter(self, data, meta):
        if self.channel_dim is None:
            data = self._contrast_jitter(data, -1, meta)
        else:
            data_c_list = [self._contrast_jitter(data.take(c, axis=self.channel_dim), c, meta)
                           for c in range(data.shape[self.channel_dim])]
            data = np.stack(data_c_list, axis=self.channel_dim)
        return data

    def __call__(self, sample):
        meta = {}
        new_sample = {(k if "#" in k and "image" in k else k): (self.contrast_jitter(v.astype(np.float32), meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # meta.update({
        #     "jitter_range": tuple(self.jitter_range),
        #     "if_keep_range": 1 if self.if_keep_range else 0,
        # })
        # new_sample['meta'][f"{self.__class__.__name__}_jitter_range"] = meta['jitter_range']
        # new_sample['meta'][f"{self.__class__.__name__}_if_keep_range"] = meta['if_keep_range']
        return new_sample


class RandomCrop:

    def __init__(self, shift_from_center, crop_sizes_ratio, spatial_dim=3,
                 padding_mode='minimum', keep_size=True):
        self.shift_from_center = shift_from_center
        self.crop_sizes_ratio = crop_sizes_ratio
        self.spatial_dim = spatial_dim
        self.padding_mode = padding_mode
        self.keep_size = keep_size
        assert (len(crop_sizes_ratio) == spatial_dim == len(shift_from_center))

    def _crop(self, data, meta):
        assert (data.ndim >= self.spatial_dim)
        padding = meta['padding']
        crop_sizes = meta["crop_sizes"]
        extend_padding = tuple([(0, 0)] * (data.ndim - self.spatial_dim) + padding)
        data = np.pad(data, extend_padding, mode=self.padding_mode)
        shifted_center = meta['shifted_center']
        crop_slices = [slice(c - s // 2 + p[0], c + (s - s // 2) + p[0])
                       for c, p, s in zip(shifted_center, padding, crop_sizes)]
        extend_crop_slices = [slice(None, None)] * (data.ndim - self.spatial_dim) + crop_slices
        cropped = data[tuple(extend_crop_slices)]
        meta['size'] = tuple(cropped.shape[-self.spatial_dim:])
        return cropped

    def __call__(self, sample):
        # compute center offset
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        crop_sizes_ratio = tuple([np.random.uniform(ratio, 1.0) for ratio in self.crop_sizes_ratio])
        crop_sizes = tuple([int(cs * ds) for cs, ds in zip(crop_sizes_ratio, data_shape)])
        center = np.asarray(data_shape) // 2
        offset = tuple([int(np.random.uniform(-c * sh, c * sh))
                        for c, sh in zip(center, self.shift_from_center)])
        shifted_center = tuple([c + offs for c, offs in zip(center, offset)])
        padding = [(max(0, si // 2 - cc),
                    max(0, cc + si // 2 - sh))
                   for sh, si, cc in zip(data_shape, crop_sizes, shifted_center)]
        meta = {"offset": offset, "shifted_center": shifted_center,
                "padding": padding, "crop_sizes": crop_sizes}
        new_sample = {(k if "#" in k else k): (self._crop(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_crop_sizes_ratio"] = crop_sizes_ratio
        new_sample['meta'][f"{self.__class__.__name__}_padding_mode"] = self.padding_mode
        new_sample['meta'][f"{self.__class__.__name__}_offset"] = meta['offset']
        new_sample['meta'][f"{self.__class__.__name__}_padding"] = meta['padding']
        new_sample['meta'][f"{self.__class__.__name__}_shifted_center"] = meta['shifted_center']
        new_sample['meta'][f"{self.__class__.__name__}_crop_sizes"] = meta['crop_sizes']
        new_sample['meta']['size'] = meta['size']
        if self.keep_size:
            t = Resample('fixed_size', 1, data_shape)
            new_sample = t(new_sample)
        return new_sample


class RandomCubeMask:

    def __init__(self, shift_from_center, crop_sizes_ratio, spatial_dim=3):
        self.shift_from_center = shift_from_center
        self.crop_sizes_ratio = crop_sizes_ratio
        self.spatial_dim = spatial_dim
        assert (len(crop_sizes_ratio) == spatial_dim == len(shift_from_center))

    def _mask(self, data, meta):
        assert (data.ndim >= self.spatial_dim)
        data_shape = data.shape[-self.spatial_dim:]
        crop_sizes = meta["crop_sizes"]
        shifted_center = meta['shifted_center']
        crop_slices = [slice(max(0, c - s // 2), min(c + (s - s // 2), sp))
                       for c, sp, s in zip(shifted_center, data_shape, crop_sizes)]
        extend_crop_slices = [slice(None, None)] * (data.ndim - self.spatial_dim) + crop_slices
        data_m = np.zeros_like(data)
        data_m[tuple(extend_crop_slices)] = data[tuple(extend_crop_slices)]
        return data_m

    def __call__(self, sample):
        # compute center offset
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        crop_sizes_ratio = tuple([np.random.uniform(ratio, 1.0) for ratio in self.crop_sizes_ratio])
        crop_sizes = tuple([int(cs * ds) for cs, ds in zip(crop_sizes_ratio, data_shape)])
        center = np.asarray(data_shape) // 2
        offset = tuple([int(np.random.uniform(-c * sh, c * sh))
                        for c, sh in zip(center, self.shift_from_center)])
        shifted_center = tuple([c + offs for c, offs in zip(center, offset)])
        meta = {"shifted_center": shifted_center,
                "crop_sizes": crop_sizes}
        new_sample = {(k if "#" in k else k): (self._mask(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_shifted_center"] = meta['shifted_center']
        new_sample['meta'][f"{self.__class__.__name__}_crop_sizes"] = meta['crop_sizes']
        new_sample['meta'][f"{self.__class__.__name__}_crop_sizes_ratio"] = meta['crop_sizes_ratio']
        return new_sample


class RandomMaskGaussian:

    def __init__(self, times=5, region_range=((0.2, 0.8), (0.2, 0.8), (0.2, 0.8)),
                 radius_range=((0.01, 0.1), (0.01, 0.1), (0.01, 0.1)),
                 spatial_dim=3, assign_value=0):
        self.region_range = region_range
        self.radius_range = radius_range
        self.spatial_dim = spatial_dim
        self.assign_value = assign_value
        self.times = times

    def gen_gaussian_kernel(self, shape, mean):
        coors = [range(shape[d]) for d in range(len(shape))]
        k = np.zeros(shape=shape)
        cartesian_product = [[]]
        for coor in coors:
            cartesian_product = [x + [y] for x in cartesian_product for y in coor]
        cartesian_product = np.asarray(cartesian_product)
        var = cartesian_product.var()
        for c in cartesian_product:
            s = np.sum((c - mean) ** 2)
            # s = 0
            # for cc, m in zip(c, mean):
            #     s += (cc - m) ** 2
            k[tuple(c)] = np.exp(-s / (2 * var))
        return k

    def _mask_gaussian(self, data, meta):
        assert (data.ndim >= self.spatial_dim)
        data_shape = data.shape[-self.spatial_dim:]
        mask_centers = meta["gaussian_centers"]
        radius = meta['gaussian_radius']
        data_m = copy.deepcopy(data)
        min_v = data.min()
        max_v = data.max()
        grids = np.ogrid[tuple([slice(0, n) for n in data.shape])]
        canvas = np.zeros_like(data, dtype=np.float32)
        for mask_center, rad in zip(mask_centers, radius):
            rad = min(rad)
            v_scale = np.random.uniform(1.0, 3.0)
            v = int(rad * v_scale)
            b_mask = np.sum([(gr - n) ** 2 for gr, n in zip(grids, mask_center)]) <= rad ** 2
            mask_slices = [slice(max(0, c - v), min(c + v, sp))
                           for c, sp in zip(mask_center, data_shape)]
            actual_size = [ss.stop - ss.start for ss in mask_slices]
            g_center = [n // 2 for n in actual_size]

            g_kernel = self.gen_gaussian_kernel(actual_size, g_center)
            extend_crop_slices = [slice(None, None)] * (data.ndim - self.spatial_dim) + mask_slices
            canvas[tuple(extend_crop_slices)] = g_kernel
            data_m[b_mask] = canvas[b_mask]
            canvas.fill(0.0)
        return data_m

    def __call__(self, sample):
        # compute center offset
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        gaussian_centers = [
            tuple([int(ds * np.random.uniform(r[0], r[1])) for ds, r in zip(data_shape, self.region_range)])
            for _ in range(self.times)]
        gaussian_radius = [
            tuple([int(np.random.uniform(rs[0], rs[1]) * ds) for rs, ds in zip(self.radius_range, data_shape)])
            for _ in range(self.times)]
        meta = {"gaussian_centers": gaussian_centers,
                "gaussian_radius": gaussian_radius}
        new_sample = {(k if "#" in k and "image" in k else k): (self._mask_gaussian(v, meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_gaussian_centers"] = meta['gaussian_centers']
        new_sample['meta'][f"{self.__class__.__name__}_gaussian_radius"] = meta['gaussian_radius']
        return new_sample


class RandomMaskOut:

    def __init__(self, times=5, region_range=((0.2, 0.8), (0.2, 0.8), (0.2, 0.8)),
                 region_size=((0.01, 0.06), (0.01, 0.06), (0.01, 0.06)),
                 spatial_dim=3, assign_value=0):
        self.region_range = region_range
        self.region_size = region_size
        self.spatial_dim = spatial_dim
        self.assign_value = assign_value
        self.times = times
        assert (len(region_range) == spatial_dim == len(region_size))

    def _mask_out(self, data, meta):
        assert (data.ndim >= self.spatial_dim)
        data_shape = data.shape[-self.spatial_dim:]
        mask_centers = meta["mask_centers"]
        mask_sizes = meta['mask_sizes']
        data_m = copy.deepcopy(data)
        min_v = data.min()
        max_v = data.max()
        for mask_center, mask_size in zip(mask_centers, mask_sizes):
            crop_slices = [slice(max(0, c - s // 2), min(c + (s - s // 2), sp))
                           for c, sp, s in zip(mask_center, data_shape, mask_size)]
            extend_crop_slices = [slice(None, None)] * (data.ndim - self.spatial_dim) + crop_slices
            data_m[tuple(extend_crop_slices)] = np.random.uniform(min_v, max_v)
        return data_m

    def __call__(self, sample):
        # compute center offset
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        mask_centers = [tuple([int(ds * np.random.uniform(r[0], r[1])) for ds, r in zip(data_shape, self.region_range)])
                        for _ in range(self.times)]
        mask_sizes = [tuple([int(np.random.uniform(rs[0], rs[1]) * ds) for rs, ds in zip(self.region_size, data_shape)])
                      for _ in range(self.times)]
        meta = {"mask_centers": mask_centers,
                "mask_sizes": mask_sizes}
        new_sample = {(k if "#" in k and "image" in k else k): (self._mask_out(v, meta)
                                                                if "#" in k and "image" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # meta.update(meta)
        # new_sample['meta'][f"{self.__class__.__name__}_mask_centers"] = meta['mask_centers']
        # new_sample['meta'][f"{self.__class__.__name__}_mask_sizes"] = meta['mask_sizes']
        return new_sample


class CenterCrop:

    def __init__(self, crop_sizes_ratio, spatial_dim=3):
        self.crop_sizes_ratio = crop_sizes_ratio
        self.spatial_dim = spatial_dim
        assert (len(crop_sizes_ratio) == spatial_dim)

    def _crop(self, data, meta):
        assert (data.ndim >= self.spatial_dim)
        crop_sizes = meta["crop_sizes"]
        center = meta['center']
        crop_slices = [slice(c - s // 2, c + (s - s // 2))
                       for c, s in zip(center, crop_sizes)]
        extend_crop_slices = [slice(None, None)] * (data.ndim - self.spatial_dim) + crop_slices
        cropped = data[extend_crop_slices]
        meta['size'] = tuple(cropped.shape[-self.spatial_dim:])
        return cropped

    def __call__(self, sample):
        # compute center offset
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        crop_sizes = tuple([int(cs * ds) for cs, ds in zip(self.crop_sizes_ratio, data_shape)])
        center = tuple((np.asarray(data_shape) // 2).tolist())
        meta = {"center": center,
                "crop_sizes": crop_sizes}
        new_sample = {(k if "#" in k else k): (self._crop(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        meta.update({"crop_sizes_ratio": self.crop_sizes_ratio})
        new_sample['meta'][f"{self.__class__.__name__}_center"] = meta['center']
        new_sample['meta'][f"{self.__class__.__name__}_crop_sizes"] = meta['crop_sizes']
        new_sample['meta']['size'] = meta['size']
        return new_sample


class DiskMaskOut:

    def __init__(self, select_axis=-3, spatial_dim=3):
        self.spatial_dim = spatial_dim
        self.select_axis = select_axis

    def _disk_mask_out(self, data):
        assert (data.ndim >= (self.spatial_dim - 1))
        data_shape = data.shape[-self.spatial_dim + 1:]
        center = tuple((np.asarray(data_shape) // 2).tolist())
        mask_radius = np.min(data_shape) // 2
        mask_slices = tuple([slice(0, d) for d in data_shape])
        spans = np.ogrid[mask_slices]
        mask = sum([(sp - c) ** 2 for sp, c in zip(spans, center)]) <= (mask_radius ** 2)
        data = data * mask
        return data

    def disk_mask_out(self, data):
        data_c_list = [self._disk_mask_out(data.take(c, axis=self.select_axis))
                       for c in range(data.shape[self.select_axis])]
        data = np.stack(data_c_list, axis=self.select_axis)
        return data

    def __call__(self, sample):
        # compute center offset
        new_sample = {(k if "#" in k else k): (self.disk_mask_out(v)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_placeholder"] = 1
        return new_sample


class StandarizeChannel(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, ch_dim):
        self.ch_dim = ch_dim

    def stand(self, a):
        a = a - a.mean()
        a /= a.std()
        return a

    def stan_per_channel(self, a):
        # looking for a 4d tensor
        if len(a.shape) == 4:
            a = [self.stand(a.take(n, axis=self.ch_dim)) for n in range(a.shape[self.ch_dim])]
            return np.stack(a, axis=self.ch_dim)
        elif len(a.shape) == 3:
            a = self.stand(a)
            return a
        else:
            raise NotImplementedError

    def __call__(self, sample):
        sample = {(k if "#" in k and "image" in k else k): (
            self.stan_per_channel(v.astype(np.float32)) if "#" in k and "image" in k else v)
            for k, v in sample.items()}
        return sample


class RandomMoveAxis:

    def __init__(self, spatial_dim):
        self.spatial_dim = spatial_dim

    def _move_axis(self, data, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        sampled_comb = meta['sampled_comb']
        # copy is to suppress opencv error
        data = np.moveaxis(data, sampled_comb[0], sampled_comb[1]).copy()
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        all_combs = list(combinations([-n for n in range(1, self.spatial_dim + 1)], 2))
        sampled_comb = tuple(random.sample(all_combs, 1)[0])
        meta = {"sampled_comb": sampled_comb}
        new_sample = {(k if "#" in k else k): (self._move_axis(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_sampled_comb"] = sampled_comb
        new_sample['meta']['size'] = meta['size']
        return new_sample





class RandomFlip:

    def __init__(self, spatial_dim):
        self.spatial_dim = spatial_dim

    def _flip_axis(self, data, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        flip_axis = meta['flip_axis']
        data = np.flip(data, axis=flip_axis).copy()
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        flip_axis = random.sample([-n for n in range(1, self.spatial_dim + 1)], 1)[0]
        meta = {"flip_axis": flip_axis}
        new_sample = {(k if "#" in k else k): (self._flip_axis(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        # new_sample['meta'][f"{self.__class__.__name__}_flip_axis"] = flip_axis
        return new_sample


class RandomRotate90:

    def __init__(self, spatial_dim):
        self.spatial_dim = spatial_dim

    def _rotate90(self, data, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        rotate_axis = meta['rotate_axis']
        rotate_times = meta['rotate_times']
        # copy is to supress opencv error
        data = np.rot90(data, axes=rotate_axis, k=rotate_times).copy()
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        rotate_times = random.sample(range(4), 1)[0]
        all_combs = list(combinations([-n for n in range(1, self.spatial_dim + 1)], 2))
        rotate_axis = tuple(random.sample(list(all_combs), 1)[0])
        meta = {"rotate_axis": rotate_axis, "rotate_times": rotate_times}
        new_sample = {(k if "#" in k else k): (self._rotate90(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_rotate_axis"] = rotate_axis
        new_sample['meta'][f"{self.__class__.__name__}_rotate_times"] = rotate_times
        new_sample['meta']['size'] = meta['size']
        return new_sample


class RandomRotate:

    def __init__(self, spatial_dim, rotate_range):
        self.spatial_dim = spatial_dim
        self.rotate_range = rotate_range

    def _rotate(self, data, key, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        rotate_axis = meta['rotate_axis']
        rotate_angle = meta['rotate_angle']

        # copy is to supress opencv error
        if "image" in key:
            data = ndimage.rotate(data, rotate_angle, reshape=False, axes=rotate_axis,
                                  order=3, mode='constant', cval=data.min()).copy()
        else:
            data = ndimage.rotate(data, rotate_angle, reshape=False,
                                  axes=rotate_axis, order=0, mode='constant', cval=data.min()).copy()
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        rotate_angle = random.randint(*self.rotate_range)
        all_combs = list(combinations([-n for n in range(1, self.spatial_dim + 1)], 2))
        rotate_axis = tuple(random.sample(list(all_combs), 2)[0])
        meta = {"rotate_axis": rotate_axis, "rotate_angle": rotate_angle}
        new_sample = {(k if "#" in k else k): (self._rotate(v, k, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta']['size'] = meta['size']
        return new_sample


class RandomAffineTransform3D(object):

    def __init__(self, spatial_dim, rotations=(0.2 * math.pi,
                                               0.2 * math.pi,
                                               0.2 * math.pi),
                 scales=(0.05, 0.05, 0.05)):
        self.spatial_dim = spatial_dim
        self.rotations = rotations
        # self.translations = translations
        self.scales = scales

    def _affine(self, data, key, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        scales = meta['scales']
        rotate_angles = meta['rotate_angles']
        # translations = meta['translations']
        T0 = np.array([[scales[0], 0, 0, - data.shape[0] / 2.0],
                       [0, scales[1], 0, - data.shape[1] / 2.0],
                       [0, 0, scales[2], - data.shape[2] / 2.0],
                       [0, 0, 0, 1.]])

        alpha, beta, theta = rotate_angles
        rotz = np.array([[math.cos(alpha), -math.sin(alpha), 0, 0],
                         [math.sin(alpha), math.cos(alpha), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        roty = np.array([[math.cos(beta), 0, math.sin(beta), 0],
                         [0, 1, 0, 0],
                         [-math.sin(beta), 0, math.cos(beta), 0],
                         [0, 0, 0, 1]])
        rotx = np.array([[1, 0, 0, 0],
                         [0, math.cos(theta), -math.sin(theta), 0],
                         [0, math.sin(theta), math.cos(theta), 0],
                         [0, 0, 0, 1]])
        T1 = np.array([[1, 0, 0, data.shape[0] / 2.0],
                       [0, 1, 0, data.shape[1] / 2.0],
                       [0, 0, 1, data.shape[2] / 2.0],
                       [0, 0, 0, 1.]])

        transform_matrix = T1.dot(rotz.dot(roty.dot(rotx.dot(T0))))
        transform_matrix_inverse = np.linalg.inv(transform_matrix)
        order = 3 if "image" in key else 0
        data = affine_transform(data, transform_matrix_inverse[:3, :3],
                                offset=transform_matrix_inverse[:3, 3], output_shape=data.shape,
                                mode="constant", order=order, cval=data.min())
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        d = [v for k, v in sample.items() if "#" in k][0]
        data_shape = d.shape[-self.spatial_dim:]
        scales = [float(np.random.uniform(1.0 - rp, 1.0 + rp)) for rp in self.scales]
        rotate_angle = [float(np.random.uniform(-rp, rp)) for rp in self.rotations]
        # translation = [random.randint(-int(ds * tp), int(ds * tp))
        #                for ds, tp in zip(data_shape, self.translations)]
        meta = {"scales": scales,
                # "translations": translation,
                "rotate_angles": rotate_angle}
        new_sample = {(k if "#" in k else k): (self._affine(v, k, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta'][f"{self.__class__.__name__}_scales"] = scales
        new_sample['meta'][f"{self.__class__.__name__}_rotate_angle"] = rotate_angle
        new_sample['meta']['size'] = meta['size']
        return new_sample


class RandomRotateInplane90:

    def __init__(self, spatial_dim):
        self.spatial_dim = spatial_dim

    def _rotate90(self, data, meta):
        # assuming spatial dimensions are always at the end in dim ordering
        rotate_times = meta['rotate_times']
        # copy is to supress opencv error
        rotate_axis = meta['rotate_axis']
        data = np.rot90(data, axes=rotate_axis, k=rotate_times).copy()
        if "size" in meta.keys():
            assert (data.shape[-self.spatial_dim:] == meta["size"])
        else:
            meta["size"] = data.shape[-self.spatial_dim:]

        return data

    def __call__(self, sample):
        rotate_times = random.sample(range(4), 1)[0]
        meta = {"rotate_axis": (-1, -2), "rotate_times": rotate_times}
        new_sample = {(k if "#" in k else k): (self._rotate90(v, meta)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta']['size'] = meta['size']
        return new_sample

class Identity:

    def __init__(self):
        pass
    def __call__(self, sample):
        return sample

class Flip3DOneShot:

    def __init__(self, flip_axis=None, spatial_dim=2):
        self.spatial_dim = spatial_dim
        if flip_axis is None:
            toss_int = random.randint(1, 3)
            all_p = list(combinations([n for n in range(self.spatial_dim, 5)], toss_int))
            flip_axis = random.sample(all_p, 1)[0]
        self.flip_axis = flip_axis

    def _flip_axis(self, data):
        assert data.dim() == 5
        data = torch.flip(data, self.flip_axis)
        return data

    def __call__(self, sample):
        new_sample = {(k if "#" in k else k): (self._flip_axis(v)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        return new_sample

class Rotate903DOneShot:

    def __init__(self, rotate_axis=None, rotate_times=None, spatial_dim=2):
        self.spatial_dim = spatial_dim
        if rotate_axis is None:
            all_p = list(permutations(list(range(self.spatial_dim, 5)), 2))
            rotate_axis = random.sample(all_p, 1)[0]
        self.rotate_axis = rotate_axis
        if rotate_times is None:
            self.rotate_times = random.randint(1, 3)
        else:
            self.rotate_times = rotate_times

    def _rotate_axis(self, data):
        assert data.dim() == 5
        data = torch.rot90(data, self.rotate_times, self.rotate_axis)
        return data

    def __call__(self, sample):
        new_sample = {(k if "#" in k else k): (self._rotate_axis(v)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        return new_sample


class Rotate3DXOneShot:

    def __init__(self, theta=(0, np.pi)):
        self.theta = np.random.uniform(theta[0], theta[1], 1)

    def get_rot_mat(self):
        theta = torch.tensor(self.theta)
        return torch.tensor([[1, 0 ,0 ,0],
                             [0, torch.cos(theta), -torch.sin(theta), 0],
                             [0, torch.sin(theta), torch.cos(theta), 0]])

    def _rotate(self, data):
        assert data.dim() == 5
        rot_mat = self.get_rot_mat()[None, ...].type(data.type()).repeat(data.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, data.size()).type(data.type())
        data = F.grid_sample(data, grid)
        return data

    def __call__(self, sample):
        new_sample = {(k if "#" in k else k): (self._rotate(v)
                                               if "#" in k else v)
                      for k, v in sample.items()}
        return new_sample

class Rescale3DOneShot:

    def __init__(self, rescale_factor_pool=None, scale_factor=None, mode='size'):
        self.rescale_factor_pool = rescale_factor_pool
        self.mode = mode
        if scale_factor is None:
            scale_factor = tuple(np.random.choice(self.rescale_factor_pool, 3))
        self.scale_factor = scale_factor

    def _rescale(self, data, mode):
        if self.mode == 'factor':
            data = F.interpolate(data, scale_factor=self.scale_factor, mode=mode)
        elif self.mode == 'size':
            data = F.interpolate(data, size=self.scale_factor, mode=mode)
        return data

    def __call__(self, sample):
        new_sample = {}

        for k, v in sample.items():
            if "#" in k:
                if "image" in k:
                    mode = 'trilinear'
                elif "reference" in k:
                    mode = 'nearest'
                else:
                    raise NotImplementedError
                v = self._rescale(v, mode)
            new_sample[k] = v
        return new_sample