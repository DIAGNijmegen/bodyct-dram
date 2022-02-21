from importlib import import_module
import os, csv
from scipy import ndimage
import math
import torch
import numpy as np
from skimage.filters import thresholding
import SimpleITK as sitk
from pathlib import Path
import cv2

import importlib.util

import os
import pandas as pd

def convert_dict_string(d, i=1):
    sp = "".join(["    "] * i)
    sp0 = "".join(["    "] * (i - 1))
    s = f"\r\n{sp0}{{"

    for k, v in d.items():
        if isinstance(v, dict):
            s += f"\r\n{sp}{k}:{convert_dict_string(v, i+1)}"
        else:
            s += f"\r\n{sp}{k}:{v}"
    s += f"\r\n{sp0}}}"
    return s

class Settings:
    def __init__(self, settings_module_path, settings_name="settings"):
        # store the settings module in case someone later cares
        self.settings_module_path = settings_module_path
        spec = importlib.util.spec_from_file_location(settings_name, settings_module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        compulsory_settings = (
            "EXP_NAME",
            "MODEL_NAME",
        )

        self._explicit_settings = set()
        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)
                if setting in compulsory_settings and setting is None:
                    raise AttributeError("The %s setting must be Not None. " % setting)
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

    def is_overridden(self, setting):
        return setting in self._explicit_settings

    def __str__(self):
        # return "{}".format(self.__dict__)
        return convert_dict_string(self.__dict__)


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MovingAverage():
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val

def expand_dims(tensors, expected_dim):
    if tensors.dim() < expected_dim:
        for n in range(expected_dim - tensors.dim()):
            tensors = tensors.unsqueeze(0)

    return tensors


def squeeze_dims(tensors, expected_dim, squeeze_start_index=0):
    if tensors.dim() > expected_dim:
        for n in range(tensors.dim() - expected_dim):
            tensors = tensors.squeeze(squeeze_start_index)

    return tensors

def write_array_to_mha_itk(target_path, arrs, names, type=np.int16,
                           origin=[0.0, 0.0, 0.0],
                           direction=np.eye(3, dtype=np.float64).flatten().tolist(),
                           spacing=[1.0, 1.0, 1.0], orientation='RAI'):
    """ arr is z-y-x, spacing is z-y-x."""
    # size = arrs[0].shape
    for arr, name in zip(arrs, names):
        # assert (arr.shape == size)
        simage = sitk.GetImageFromArray(arr.astype(type))
        simage.SetSpacing(np.asarray(spacing, np.float64).tolist())
        simage.SetDirection(direction)
        simage.SetOrigin(origin)
        fw = sitk.ImageFileWriter()
        fw.SetFileName(target_path + '/{}.mha'.format(name))
        fw.SetDebug(False)
        fw.SetUseCompression(True)
        fw.SetGlobalDefaultDebug(False)
        fw.Execute(simage)


def get_value_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_value_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_value_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found

def windowing(image, from_span=(-1150, 350), to_span=(0, 255)):
    if from_span is None:
        min_input = np.min(image)
        max_input = np.max(image)
    else:
        min_input = from_span[0]
        max_input = from_span[1]
    image = np.clip(image, a_min=min_input, a_max=max_input)
    image = ((image - min_input) / float(max_input - min_input)) * (to_span[1] - to_span[0]) + to_span[0]
    return image

def collate_tensors_simple(tensor):
    t = [t for t in tensor]
    return torch.stack(t, dim=0)

def merge_dict(list_dict):
    new_d = {}
    for k in list_dict[0].keys():
        new_d[k] = tuple(d[k] for d in list_dict)

    return new_d

def collate_func_dict_fix(batch):
    merge_d = {}
    for k in batch[0].keys():
        if not isinstance(batch[0][k], (dict, )):
            merge_d[k] = [b[k] for b in batch]
        else:
            merge_d[k] = merge_dict([b[k] for b in batch])

    for k in merge_d.keys():
        if not isinstance(merge_d[k], dict) and isinstance(merge_d[k][0], torch.Tensor):
            collate_tensors = collate_tensors_simple(merge_d[k])
            merge_d[k] = collate_tensors

    return merge_d

def binary_cam(cam_probs, scaler=1.0, from_span=(0, 1)):
    if isinstance(cam_probs, torch.Tensor):
        cam_np = cam_probs.cpu().numpy()
    else:
        cam_np = cam_probs
    if cam_np.size == 0:
        raise ValueError("empty array encountered! cam_probs.size == 0.")

    cam_np_w = windowing(cam_np, from_span=from_span).astype(np.uint8)
    if len(np.unique(cam_np_w)) < 2:  # if there is only one color
        th = np.unique(cam_np_w)[0]
        return np.ones_like(cam_np_w).astype(np.bool), th / 255.0

    th = min(thresholding.threshold_otsu(cam_np_w) * scaler, 255.0)
    cam_th = (th / 255.0)
    # print("cam th: {}".format(cam_th))
    return cam_np_w >= th, cam_th

def find_crops(mask, spacing, border):
    object_slices = ndimage.find_objects(mask > 0)[0]
    if border > 0:
        pad_object_slices = tuple([
            slice(max(0, os.start - int(math.ceil(border / sp))), min(ss, os.stop + int(math.ceil(border / sp))))
            for os, ss, sp in zip(object_slices, mask.shape, spacing)]
        )
    else:
        pad_object_slices = object_slices

    return pad_object_slices

def read_csv_in_dict_double(csv_file_path, column_keys, fieldnames=None):
    row_dict = {}
    if not os.path.exists(csv_file_path):
        return row_dict, None
    with open(csv_file_path, "rt") as fp:
        cr = csv.DictReader(fp, delimiter=',', fieldnames=fieldnames)
        for row in cr:
            row_dict[tuple([row[column_key] for column_key in column_keys])] = row

        field_names = cr.fieldnames
    return row_dict, field_names

def read_csv_in_dict(csv_file_path, column_key, fieldnames=None):
    row_dict = {}
    if not os.path.exists(csv_file_path):
        return row_dict, None
    with open(csv_file_path, "rt") as fp:
        cr = csv.DictReader(fp, delimiter=',', fieldnames=fieldnames)
        for row in cr:
            row_dict[row[column_key]] = row

        field_names = cr.fieldnames
    return row_dict, field_names

def get_callable_by_name(module_name):
    cls = getattr(import_module(module_name.rpartition('.')[0]),
                  module_name.rpartition('.')[-1])
    return cls


_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0, new_size=None):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, '
                '32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(
            _SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    if new_size is None:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(
            np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in
                    new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


def resample_image(input_filename=None, output_filename=None,
                   factor=2, interpolator='linear'):
    """
    Resample the input filename to new voxel size.
    Image is downsampled using linear interpolator (linear) and segmentations
    as nearest neighbour (nearest).
    Input:
       - input_filename: filename of the image to be downsampled
       - factor: factor by which the image needs to rescaled. >1 is downsample
         and <1 is upsample
    Output:
       -  output_filename: filename of the output image
    """
    sitk_image = sitk.ReadImage(input_filename)
    orig_spacing = np.array(sitk_image.GetSpacing())
    req_spacing = factor * orig_spacing
    req_spacing = tuple([float(s) for s in req_spacing])
    resampled_image = resample_sitk_image(sitk_image,
                                          spacing=req_spacing,
                                          interpolator=interpolator,
                                          fill_value=0)

    sitk.WriteImage(resampled_image, output_filename)

    assert (os.path.exists(output_filename))


def resample(narray, orig_spacing, factor=2, required_spacing=None, new_size=None, interpolator='linear'):
    if new_size is not None and narray.shape == new_size:
        print("size is equal not resampling!")
        return narray, orig_spacing
    s_image = sitk.GetImageFromArray(narray)
    s_image.SetSpacing(np.asarray(orig_spacing[::-1], dtype=np.float64).tolist())

    req_spacing = factor * np.asarray(orig_spacing)
    req_spacing = tuple([float(s) for s in req_spacing])
    if required_spacing is not None:
        req_spacing = required_spacing
    if new_size:
        new_size = new_size[::-1]
    resampled_image = resample_sitk_image(s_image,
                                          spacing=req_spacing[::-1],
                                          interpolator=interpolator,
                                          fill_value=0, new_size=new_size)

    resampled = sitk.GetArrayFromImage(resampled_image)

    return resampled, req_spacing


def IOU(predict, target, smooth):
    intersection = np.sum(np.logical_and(predict, target))
    union = np.sum(np.logical_or(predict, target))
    iou = (intersection + smooth) / \
          (union + smooth)
    return iou

def Dice(predict, target, smooth):
    intersection = np.sum(np.logical_and(predict, target))
    return (2. * intersection.sum() + smooth) / (predict.sum() + target.sum() + smooth)

def TP_measure(predict, target):
    if np.sum(target) == 0:
        tpr = np.Infinity
    else:
        tpr = np.sum(np.logical_and(predict > 0, target > 0)) / np.sum(target > 0)

    return tpr

def FDR_measure(predict, target):
    if np.sum(predict > 0) == 0:
        fdr = np.Infinity
    else:
        fdr = np.sum(np.logical_and(predict > 0, ~((predict > 0) & (target > 0)))) / np.sum(predict > 0)

    return fdr

def draw_mask_tile_single_view(image, masks_list, coord_mask, num_slices, output_path, colors,
                               thickness, ext='jpg', alpha=0.5, flip_axis=0, zoom_size=360,
                               coord_axis=1, titles=None, title_offset=10, title_color=(0, 255, 0)):
    assert (all([image.shape == mask.shape for mask_list in masks_list for mask in mask_list]))
    if flip_axis is not None:
        image = np.flip(image, axis=flip_axis)
        coord_mask = np.flip(coord_mask, axis=flip_axis)
        m_shape = np.asarray(masks_list).shape
        masks_list = np.asarray([np.flip(mask, axis=flip_axis)
                                 for mask_list in masks_list for mask in mask_list]).reshape(m_shape)
    n_mask_list = len(masks_list)
    n_mask_per_list = len(masks_list[0])
    if zoom_size is not None:
        sp = [image.shape[s] for s in set(list(range(image.ndim))) - {coord_axis}]
        zoom_max_ratio = zoom_size / np.max(sp)
        zoom_ratio = [1.0 if n == coord_axis else zoom_max_ratio for n in range(image.ndim)]

        def zoom_and_pad(i, ratio, target_size, pad_ignore_axis, order):
            i_z = ndimage.zoom(i, ratio, order=order)
            crop_slices = tuple([slice(0, min(n, target_size)) if i != pad_ignore_axis else slice(None, None)
                                 for i, n in enumerate(i_z.shape)])
            i_z = i_z[crop_slices]
            pad_size = tuple([(0, 0) if n == pad_ignore_axis else (
                (target_size - zs) // 2, target_size - zs - (target_size - zs) // 2)
                              for n, zs in zip(range(i.ndim), i_z.shape)])
            i_z_p = np.pad(i_z, pad_size, mode='constant')

            assert (all(i_z_p.shape[n] == target_size for n in range(i.ndim) if n != pad_ignore_axis))
            return i_z_p

        image = zoom_and_pad(image, zoom_ratio, zoom_size, coord_axis, order=1)
        coord_mask = zoom_and_pad(coord_mask, zoom_ratio, zoom_size, coord_axis, order=0)
        masks_list = [zoom_and_pad(mask, zoom_ratio, zoom_size, coord_axis, order=0)
                      for mask_list in masks_list for mask in mask_list]

    if np.sum(coord_mask) > 0:
        foreground_slices = ndimage.find_objects(coord_mask)[0]
        s = foreground_slices[coord_axis].start
        e = foreground_slices[coord_axis].stop
        stride = (e - s) // num_slices
        if stride == 0:
            e = coord_mask.shape[coord_axis] - 1
            s = 0
            stride = (e - s) // num_slices
        slices_ids = list(range(s, e, stride))[:num_slices]
        assert (len(slices_ids) == num_slices)
    else:
        print("no object found!")
        return
    all_slice_tiles = []
    for slice_id in slices_ids:
        # form one slice source from image and masks.
        slice_image = np.take(image, slice_id, axis=coord_axis)
        slice_image_tiles = [np.dstack((slice_image, slice_image, slice_image))]
        for mask_list_id in range(n_mask_list):
            masks = masks_list[mask_list_id * n_mask_per_list: mask_list_id * n_mask_per_list + n_mask_per_list]
            mask_array = [np.take(mask, slice_id, axis=coord_axis) for mask in masks]
            rendered_image = draw_2d(slice_image, mask_array, colors, thickness, alpha=alpha)
            if titles:
                cv2.putText(rendered_image, titles[mask_list_id], (title_offset, title_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, title_color, 1, cv2.LINE_AA)
            slice_image_tiles.append(rendered_image)
        # put all sources into a tile
        slice_image_tiles = np.vstack(slice_image_tiles)
        all_slice_tiles.append(slice_image_tiles)

    draw_ = np.hstack(all_slice_tiles)
    pad_size = ((0, 0), ((1920 - draw_.shape[1]) // 2, (1920 - draw_.shape[1]) - (1920 - draw_.shape[1]) // 2), (0, 0))
    draw_ = np.pad(draw_, pad_size, mode="constant")
    if output_path:

        output_path = Path(output_path).absolute()
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        cv2.imwrite(str(output_path) + '.{}'.format(ext), draw_)


def draw_mask_tile_singleview_heatmap(image, masks_list, coord_mask, num_slices, output_path,
                                 ext='jpg', alpha=0.5, flip_axis=0, draw_anchor=True, zoom_size=360,
                                 anchor_color=(0, 255, 0), colormap='jet', coord_axis=1,
                                 titles=None, title_offset=50, title_color=(0, 255, 0)):
    assert (all([image.shape == mask.shape for mask_list in masks_list for mask in mask_list]))
    if flip_axis is not None:
        image = np.flip(image, axis=flip_axis)
        coord_mask = np.flip(coord_mask, axis=flip_axis)
        m_shape = np.asarray(masks_list).shape
        masks_list = np.asarray([np.flip(mask, axis=flip_axis)
                                 for mask_list in masks_list for mask in mask_list]).reshape(m_shape)

    n_mask_list = len(masks_list)
    n_mask_per_list = len(masks_list[0])
    if zoom_size is not None:
        sp = [image.shape[s] for s in set(list(range(image.ndim))) - {coord_axis}]
        zoom_max_ratio = zoom_size / np.max(sp)
        zoom_ratio = [1.0 if n == coord_axis else zoom_max_ratio for n in range(image.ndim)]

        def zoom_and_pad(i, ratio, target_size, pad_ignore_axis, order):
            i_z = ndimage.zoom(i, ratio, order=order)
            crop_slices = tuple([slice(0, min(n, target_size)) if i != pad_ignore_axis else slice(None, None)
                                 for i, n in enumerate(i_z.shape)])
            i_z = i_z[crop_slices]
            pad_size = tuple([(0, 0) if n == pad_ignore_axis else (
                (target_size - zs) // 2, target_size - zs - (target_size - zs) // 2)
                              for n, zs in zip(range(i.ndim), i_z.shape)])
            i_z_p = np.pad(i_z, pad_size, mode='constant')

            assert (all(i_z_p.shape[n] == target_size for n in range(i.ndim) if n != pad_ignore_axis))
            return i_z_p

        image = zoom_and_pad(image, zoom_ratio, zoom_size, coord_axis, order=1)
        coord_mask = zoom_and_pad(coord_mask, zoom_ratio, zoom_size, coord_axis, order=0)
        masks_list = [zoom_and_pad(mask, zoom_ratio, zoom_size, coord_axis, order=0)
                      for mask_list in masks_list for mask in mask_list]

    if np.sum(coord_mask) > 0:
        foreground_slices = ndimage.find_objects(coord_mask)[0]
        s = foreground_slices[coord_axis].start
        e = foreground_slices[coord_axis].stop
        stride = (e - s) // num_slices
        if stride == 0:
            e = coord_mask.shape[coord_axis] - 1
            s = 0
            stride = (e - s) // num_slices
        slices_ids = list(range(s, e, stride))[:num_slices]
        assert (len(slices_ids) == num_slices)
    else:
        print("no object found!")
        return


    all_slice_tiles = []
    for slice_id in slices_ids:
        # form one slice source from image and masks.
        slice_image = np.take(image, slice_id, axis=coord_axis)
        slice_image_tiles = [np.dstack((slice_image, slice_image, slice_image))]
        for mask_list_id in range(n_mask_list):
            masks = masks_list[mask_list_id * n_mask_per_list: mask_list_id * n_mask_per_list + n_mask_per_list]
            mask_array = [np.take(mask, slice_id, axis=coord_axis) for mask in masks]
            rendered_image = draw_2d_heatmap(slice_image, mask_array, alpha=alpha, color_map=colormap)
            if titles:
                cv2.putText(rendered_image, titles[mask_list_id], (title_offset, title_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, title_color, 1, cv2.LINE_AA)
            slice_image_tiles.append(rendered_image)
        # put all sources into a tile
        slice_image_tiles = np.vstack(slice_image_tiles)
        all_slice_tiles.append(slice_image_tiles)
    draw_ = np.hstack(all_slice_tiles)
    pad_size = ((0, 0), ((1920 - draw_.shape[1]) // 2, (1920 - draw_.shape[1]) - (1920 - draw_.shape[1]) // 2), (0, 0))
    draw_ = np.pad(draw_, pad_size, mode="constant")
    if output_path:

        output_path = Path(output_path).absolute()
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        cv2.imwrite(str(output_path) + '.{}'.format(ext), draw_)


def draw_2d(image_2d, masks_2d, colors, thickness, alpha=0.5):
    original = np.dstack((image_2d, image_2d, image_2d))
    blending = np.dstack((image_2d, image_2d, image_2d))

    for mask, color, thick in zip(masks_2d, colors, thickness):
        _, contours, _ = cv2.findContours(mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blending, contours, -1, color, thick)

    return original * (1 - alpha) + blending * alpha


def draw_2d_heatmap(image_2d, masks_2d, alpha=0.5, color_map='jet'):
    blend_image = np.dstack((image_2d, image_2d, image_2d))
    for mask in masks_2d:
        if color_map == 'jet':
            mask_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        elif color_map == 'summer':
            mask_map = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)
        else:
            raise  NotImplementedError
        blend_image = cv2.addWeighted(mask_map, alpha, blend_image, 1 - alpha, 0.0)
    return blend_image