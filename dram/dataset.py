import os

import numpy as np
from utils import read_csv_in_dict, find_crops, read_csv_in_dict_double, windowing,binary_cam
import shutil
from torch.utils.data import Dataset
import random
import SimpleITK as sitk
from collections import defaultdict
import glob
from pathlib import Path

class COPDGeneSubtypingLobeChunk(Dataset):
    ON_PREMISE_ROOT = None

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'uid')
        return sorted(list(scan_selected.keys()))

    @classmethod
    def get_series_uids_unique_scans(cls, csv_file):
        metas, _ = read_csv_in_dict(csv_file, 'uid')
        scan_lobe_map = defaultdict(list)
        for lobe_wise_uid in metas.keys():
            scan_lobe_map[lobe_wise_uid[:-2]].append(lobe_wise_uid[-1])

        selected_uids = [f"{scan_uid}-{random.sample(scan_lobe_map[scan_uid], 1)[0]}" for scan_uid in
                         scan_lobe_map.keys()]
        return sorted(selected_uids)

    def __init__(self, archive_path, uids, keep_sorted=True,
                 transforms=None):
        super(COPDGeneSubtypingLobeChunk, self).__init__()
        self.archive_path = archive_path
        self.meta_csv = archive_path + '/memo.csv'
        self.meta, _ = read_csv_in_dict(self.meta_csv, "uid")
        self.uids = uids
        self.transforms = transforms
        if keep_sorted:
            self.uids = sorted(self.uids)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        d = self.get_data(self.uids[index])
        return d

    def read_file(self, uid, path):
        file_path = os.path.join(path, f"{uid}.mha")
        sitk_image = sitk.ReadImage(file_path)
        spacing = sitk_image.GetSpacing()[::-1]
        origin = sitk_image.GetOrigin()[::-1]
        direction = np.asarray(sitk_image.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        scan = sitk.GetArrayFromImage(sitk_image)
        return scan, origin, spacing, direction

    def get_data(self, uid):
        scan, origin, spacing, direction = self.read_file(uid, self.archive_path + '/images/')
        lobe, *ignored = self.read_file(uid, self.archive_path + '/lobes/')
        lesion, *ignored = self.read_file(uid, self.archive_path + '/lesions/')

        base_dict = dict(self.meta[uid])
        base_dict.update({"size": scan.shape,
                          "spacing": spacing,
                          'original_spacing': spacing,
                          'original_size': scan.shape,
                          "origin": origin,
                          "direction": direction})
        ret = {
            "#image": scan.astype(np.int16),
            "#lobe_reference": lobe.astype(np.uint8),
            "#lesion_reference": lesion.astype(np.uint8),
            "meta": base_dict
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


class COPDGeneSubtyping(Dataset):
    ON_PREMISE_ROOT = None

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'SeriesInstanceUID')
        return sorted(list(scan_selected.keys()))

    def __init__(self, archive_path, series_uids,
                 transforms=None, keep_sorted=True, crop_border=5, emphysema_th=-950):
        super(COPDGeneSubtyping, self).__init__()
        self.archive_path = archive_path
        self.keep_sorted = keep_sorted
        self.transforms = transforms
        self.emphysema_th = emphysema_th
        self.lobe_path = archive_path + '/derived/seg-lobes-copdgene-approved_Lobes/mha/'
        meta_csv = archive_path + '/meta/ctss.csv'

        self.meta, _ = read_csv_in_dict(meta_csv, "SeriesInstanceUID")
        self.crop_border = crop_border
        if not self.keep_sorted:
            self.series_uids = random.sample(series_uids, len(series_uids))
        else:
            self.series_uids = sorted(series_uids)

        self.subtyping_labels = {}

        for series_uid in series_uids:
            cle = int(float(self.meta[series_uid]['CT_Visual_Emph_Severity_P1']))
            pse = int(float(self.meta[series_uid]['CT_Visual_Emph_Paraseptal_P1']))

            self.subtyping_labels[series_uid] = {
                "cle": cle,
                "pse": pse
            }

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, index):
        d = self.get_data(self.series_uids[index])
        return d

    def read_image(self, path):
        sitk_image = sitk.ReadImage(path)
        spacing = sitk_image.GetSpacing()[::-1]
        origin = sitk_image.GetOrigin()[::-1]
        direction = np.asarray(sitk_image.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        scan = sitk.GetArrayFromImage(sitk_image)
        return scan, origin, spacing, direction

    def get_lobe(self, series_uid):
        reference_file_path = self.lobe_path + f"/{series_uid}.mha"
        if self.ON_PREMISE_ROOT is not None:
            target_path = os.path.join(self.ON_PREMISE_ROOT, "lobes")
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            on_premise_path = os.path.join(target_path, "{}.mha".format(series_uid))
            try:
                if not os.path.exists(on_premise_path):
                    on_premise_path = shutil.copyfile(reference_file_path, on_premise_path)
                f_ = self.read_image(on_premise_path)
            except Exception as e:
                print(
                    "loading file or copying  at {} failed with {},"
                    " now read from {}.".format(on_premise_path, e, reference_file_path))
                f_ = self.read_image(reference_file_path)
            return f_
        return self.read_image(reference_file_path)

    def get_scan(self, series_uid):
        series_scan_path = self.archive_path + f"/{series_uid}.mha"
        if self.ON_PREMISE_ROOT is not None:
            if not os.path.exists(self.ON_PREMISE_ROOT):
                os.makedirs(self.ON_PREMISE_ROOT)
            on_premise_path = os.path.join(self.ON_PREMISE_ROOT, "{}.mha".format(series_uid))
            try:
                if not os.path.exists(on_premise_path):
                    # print("copying file to on premise path {}.".format(on_premise_path))
                    on_premise_path = shutil.copyfile(series_scan_path, on_premise_path)
                f_ = self.read_image(on_premise_path)
            except Exception as e:
                print(
                    "loading file or copying  at {} failed with {},"
                    " now read from {}.".format(on_premise_path, e, series_scan_path))
                f_ = self.read_image(series_scan_path)
            return f_

        return self.read_image(series_scan_path)

    def get_data(self, series_uid):
        scan, origin, spacing, direction = self.get_scan(series_uid)
        original_size = scan.shape
        lobe, origin_, spacing_, direction_ = self.get_lobe(series_uid)
        lung = lobe > 0

        assert lobe.shape == scan.shape
        slices = find_crops(lung, spacing, self.crop_border)
        scan = scan[slices]
        lung = lung[slices].astype(np.uint8)
        lobe = lobe[slices].astype(np.uint8)
        es = (np.logical_and(scan < self.emphysema_th, lung > 0)).astype(np.uint8)

        base_dict = {
            'uid': series_uid,
        }
        base_dict.update({"size": scan.shape,
                          "spacing": spacing,
                          "crop_slices": slices,
                          'LAA': self.emphysema_th,
                          'original_spacing': spacing,
                          'original_size': original_size,
                          "origin": origin,
                          "direction": direction,
                          "cle": self.subtyping_labels[series_uid]['cle'],
                          "pse": self.subtyping_labels[series_uid]['pse'],
                          })
        ret = {
            "#image": scan.astype(np.int16),
            "#lobe_reference": lobe.astype(np.uint8),
            "#lesion_reference": es.astype(np.uint8),
            "meta": base_dict
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


class TestDataset(Dataset):

    def __init__(self, scan_path, lobe_path, transforms=None, keep_sorted=True,
                 crop_border=5):
        super(TestDataset, self).__init__()
        self.scan_path = scan_path
        self.lobe_path = lobe_path
        self.crop_border = crop_border
        self.transforms = transforms
        all_scans = glob.glob(self.scan_path + '/*.mha', recursive=False)
        if keep_sorted:
            self.series_uids = sorted([Path(scan_file).stem for scan_file in all_scans])
        else:
            self.series_uids = [Path(scan_file).stem for scan_file in all_scans]

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, index):
        d = self.get_data(self.series_uids[index])
        return d

    def read_image(self, path):
        sitk_image = sitk.ReadImage(path)
        spacing = sitk_image.GetSpacing()[::-1]
        origin = sitk_image.GetOrigin()[::-1]
        direction = np.asarray(sitk_image.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        scan = sitk.GetArrayFromImage(sitk_image)
        return scan, origin, spacing, direction

    def get_data(self, series_uid):
        scan, origin, spacing, direction = self.read_image(self.scan_path + f'/{series_uid}.mha')
        original_size = scan.shape
        lobe, origin_, spacing_, direction_ = self.read_image(self.lobe_path + f'/{series_uid}.mha')
        assert lobe.shape == scan.shape

        base_dict = {
            'uid': series_uid,
        }
        base_dict.update({"size": scan.shape,
                          "spacing": spacing,
                          'original_spacing': spacing,
                          'original_size': original_size,
                          "origin": origin,
                          "direction": direction,
                          })
        ret = {
            "#image": scan.astype(np.int16),
            "#lobe_reference": lobe.astype(np.uint8),
            "meta": base_dict
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret



class RadboudCOVID(Dataset):
    ON_PREMISE_ROOT = None

    ctss_cut_off = {
        (0.0, 0.01): 0,
        (0.01, 0.05): 1,
        (0.05, 0.25): 2,
        (0.25, 0.5): 3,
        (0.5, 0.75): 4,
        (0.75, 1.00001): 5
    }

    metric_k_mapping = {
        1: "lul [0-5]",
        2: "lll [0-5]",
        3: "rul [0-5]",
        4: "rll [0-5]",
        5: "rml [0-5]"
    }

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict_double(csv_file, ['patientid', 'study'])
        uids = [f"{pid}_{sid}" for pid, sid in scan_selected.keys()]
        return sorted(uids)

    def __init__(self, archive_path, uids,
                 transforms=None, keep_sorted=True, use_masked_scan=True,
                 crop_border=5, task='wss'):
        super(RadboudCOVID, self).__init__()
        self.transforms = transforms
        self.keep_sorted = keep_sorted
        self.archive_path = archive_path
        self.crop_border = crop_border
        self.use_masked_scan = use_masked_scan
        self.lobe_folder = archive_path + f'/{task}/lobes/'
        self.pseudo_vessel_folder = archive_path + f'/{task}/pseudo_vessels/'
        self.lesion_folder = archive_path + f'/{task}/lesion/'
        self.scan_folder = archive_path + f'/{task}/images/'
        csv_file = archive_path + f'/wss_all.csv'
        self.all_metas, _ = read_csv_in_dict_double(csv_file, ['patientid', 'study'])
        self.uids = uids
        print("dataset containing: {} cases.".format(len(self.uids)), flush=True)

        if keep_sorted:
            self.uids = sorted(self.uids)
        else:
            self.uids = random.sample(self.uids, len(self.uids))
        scan_files = glob.glob(self.scan_folder + '/*.mha', recursive=False)
        self.scan_path_map = {Path(scan_file).stem
                              : scan_file for scan_file in scan_files}

        lobe_files = glob.glob(self.lobe_folder + '/*.mha', recursive=False)
        self.lobe_path_map = {Path(ref_file).stem
                              : ref_file for ref_file in lobe_files}

        lesion_files = glob.glob(self.lesion_folder + '/*.mha', recursive=False)
        self.lesion_path_map = {Path(ref_file).stem
                              : ref_file for ref_file in lesion_files}

        pseudo_vessel_files = glob.glob(self.pseudo_vessel_folder + '/*.mha', recursive=False)
        self.pseudo_vessel_path_map = {Path(ref_file).stem
                                : ref_file for ref_file in pseudo_vessel_files}

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, scan_index):
        uid = self.uids[scan_index]
        ret = self.get_data(uid)
        return ret

    def get_scan(self, scan_name):
        scan_path = self.scan_path_map[scan_name]
        if self.ON_PREMISE_ROOT is not None:
            on_premise_scan_path = os.path.join(self.ON_PREMISE_ROOT, "{}".format(self.__class__.__name__))
            if not os.path.exists(on_premise_scan_path):
                os.makedirs(on_premise_scan_path)
            on_premise_path = os.path.join(on_premise_scan_path, "{}.mha".format(scan_name))
            try:
                if not os.path.exists(on_premise_path):
                    on_premise_path = shutil.copyfile(scan_path, on_premise_path)
                f_ = sitk.ReadImage(on_premise_path)
            except Exception as e:
                print(
                    "loading file or copying  at {} failed with {},"
                    " now read from {}.".format(on_premise_path, e, scan_path))
                f_ = sitk.ReadImage(scan_path)
        else:
            f_ = sitk.ReadImage(scan_path)
        spacing = f_.GetSpacing()
        direction = f_.GetDirection()
        origin = f_.GetOrigin()
        return sitk.GetArrayFromImage(f_), origin[::-1], \
               np.asarray(direction).reshape(3, 3)[::-1].flatten().tolist(), \
               spacing[::-1]

    def get_mask(self, path_map, ref_name):
        if (ref_name not in path_map.keys()) or \
                (not os.path.exists(path_map[ref_name])):
            return None
        file_path = path_map[ref_name]
        return sitk.ReadImage(file_path)

    def get_data(self, uid):
        pid, sid = uid.split("_")
        scan, origin, direction, spacing = self.get_scan(uid)
        original_size = scan.shape
        original_spacing = spacing
        series_meta = self.all_metas[(pid, sid)]
        lobe = sitk.GetArrayFromImage(self.get_mask(self.lobe_path_map, uid)).astype(np.uint8)
        lesion = sitk.GetArrayFromImage(self.get_mask(self.lesion_path_map, uid)).astype(np.uint8)
        v_sitk = self.get_mask(self.pseudo_vessel_path_map, uid)
        if v_sitk is None:
            vessel = np.zeros_like(lobe)
        else:
            vessel = sitk.GetArrayFromImage(v_sitk).astype(np.uint8)

        if self.use_masked_scan:
            scan[lobe == 0] = -2048
        base_dict = {
            'uid': uid,
            'patient_meta': series_meta
        }
        base_dict.update({"size": scan.shape,
                          "spacing": spacing,
                          'original_spacing': original_spacing,
                          'original_size': original_size,
                          "original_origin": origin,
                          "original_direction": direction,
                          "origin": origin,
                          "direction": direction,
                          })

        ret = {
            "#image": scan.astype(np.int16),
            "#lobe_reference": lobe.astype(np.uint8),
            "#lesion_reference": lesion.astype(np.uint8),
            "#vessel_reference": vessel.astype(np.uint8),
            "meta": base_dict
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


class RadboudCOVIDLobeVesselChunk(RadboudCOVID):
    ON_PREMISE_ROOT = None

    def __init__(self, archive_path, case_uids,
                 transforms=None, keep_sorted=True,
                 crop_border=5, task='wss_chunk'):
        super(RadboudCOVID, self).__init__()
        self.transforms = transforms
        self.keep_sorted = keep_sorted
        self.archive_path = archive_path
        self.crop_border = crop_border
        self.lobe_folder = archive_path + f'/{task}/lobes/'
        self.scan_folder = archive_path + f'/{task}/images/'
        csv_file = archive_path + f'/{task}/memo.csv'
        self.all_metas, _ = read_csv_in_dict(csv_file, "uid")
        self.case_uids = case_uids
        print("dataset containing: {} cases.".format(len(self.case_uids)), flush=True)

        self.uids = [uid for uid in self.all_metas.keys()
                     if f"{self.all_metas[uid]['patientid']}_{self.all_metas[uid]['study']}" in self.case_uids]
        print("dataset containing: {} chunks.".format(len(self.uids)), flush=True)
        if keep_sorted:
            self.uids = sorted(self.uids)
        else:
            self.uids = random.sample(self.uids, len(self.uids))
        scan_files = glob.glob(self.scan_folder + '/*.mha', recursive=False)
        self.scan_path_map = {Path(scan_file).stem
                              : scan_file for scan_file in scan_files}

        lobe_files = glob.glob(self.lobe_folder + '/*.mha', recursive=False)
        self.lobe_path_map = {Path(ref_file).stem
                              : ref_file for ref_file in lobe_files}

        pseudo_vessel_files = glob.glob(self.pseudo_vessel_folder + '/*.mha', recursive=False)
        self.pseudo_vessel_path_map = {Path(ref_file).stem
                                : ref_file for ref_file in pseudo_vessel_files}

    def get_data(self, lobe_wise_uid):
        lobe_id = lobe_wise_uid.rpartition('-')[-1]
        scan, origin, direction, spacing = self.get_scan(lobe_wise_uid)
        original_size = scan.shape
        original_spacing = spacing
        meta = self.all_metas[lobe_wise_uid]
        ctss = int(meta['ctss'])
        lobe = sitk.GetArrayFromImage(self.get_mask(self.lobe_path_map, lobe_wise_uid)).astype(np.uint8)
        vessel = sitk.GetArrayFromImage(self.get_mask(self.pseudo_vessel_folder, lobe_wise_uid)).astype(np.uint8)

        vessel = np.logical_and(vessel > 0, lobe > 0)
        w_scan = windowing(scan, to_span=(0, 1))
        _, th = binary_cam(w_scan[lobe > 0], 0.75)
        lesion_candidate = (w_scan > th) & (lobe > 0)

        base_dict = {
            'uid': lobe_wise_uid,
            'patient_meta': meta
        }
        base_dict.update({"size": scan.shape,
                          "spacing": spacing,
                          'original_spacing': original_spacing,
                          'original_size': original_size,
                          "original_origin": origin,
                          "original_direction": direction,
                          "origin": origin,
                          "direction": direction,
                          })
        ret = {
            "#image": scan.astype(np.int16),
            "#pseudo_lesion_reference": lesion_candidate.astype(np.uint8),
            "#lobe_reference": lobe.astype(np.uint8),
            "#vessel_reference": vessel.astype(np.uint8),
            "ctss": ctss,
            "lobe_id": lobe_id,
            "meta": base_dict
        }

        if self.transforms:
            ret = self.transforms(ret)
        return ret