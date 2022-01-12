
import os
import time
import random
import glob
import re
import pickle
import copy
from pathlib import Path
from contextlib import contextmanager
import warnings
from collections.abc import Iterable

import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from .utils import interpolate_sample, augmentNoise, spatial_augment, LabelDisturbanceMode

@contextmanager
def torch_manual_seeded(seed):
    saved_state = torch.get_rng_state()
    yield
    torch.set_rng_state(saved_state)

class CrossmodaHybridIdLoader(Dataset):
    def __init__(self,
        base_dir, domain, state,
        ensure_labeled_pairs=True, use_additional_data=False, resample=True,
        size:tuple=(96,96,60), normalize:bool=True,
        max_load_num=None, crop_3d_w_dim_range=None, crop_2d_slices_gt_num_threshold=None,
        modified_3d_label_override=None, prevent_disturbance=False,
        use_2d_normal_to=None, flip_r_samples=True, pre_interpolation_factor=2.,
        debug=False
    ):

        """
        Function to create Dataset structure with crossMoDa data.
        The function allows to use different preproccessing steps of the crossMoDa data set
        and using additinal data from TCIA database.
        The data can also be resampled to a desired size and normalized to mean=0 and std=1.

        Parameters:
                base_dir (os.Pathlike): provide the directory which contains "L1..." to "L4..." directories
                domain (str): choose which domain to load. Can be set to "source", "target" or "validation". Source are ceT1, target and validation hrT2 images.

                state (str): state of preprocessing:    "l1" = original data,
                                                        "l2" = resampled data @ 0.5mm,
                                                        "l3" = center-cropped data,
                                                        "l4" = image specific crops for desired anatomy

                ensure_labeled_pairs (bool): Only images with corresponding labels will be loaded (default: True)

                use_additional_data (bool): set to True to use additional data from TCIA (default: False)

                resample (bool): set to False to disable resampling to desired size (default: True)

                size (tuple): 3d-tuple(int) to which the data is resampled. Unused if resample=False. (default: (96,96,60)).
                    WARNING: choosing large sizes or not resampling can lead to excess memory usage

                normalize (bool): set to False to disable normalization to mean=0, std=1 for each image (default: True)
                max_load_num (int): maximum number of pairs to load (uses first max_load_num samples for either images and labels found)
                crop_3d_w_dim_range (tuple): Tuple of ints defining the range to which dimension W of (D,H,W) is cropped
                use_2d_normal_to (bool):

        Returns:
                torch.utils.data.Dataset containing CrossMoDa data

        Useful Links:
        CrossMoDa challenge:
        https://crossmoda.grand-challenge.org/

        ToDos:
            extend to other preprocessing states

        Example:
            dataset = CrossMoDa_source('original')

            data = dataset.get_data()

        """
        self.label_tags = ['background', 'tumour']
        self.use_2d_normal_to = use_2d_normal_to
        self.crop_2d_slices_gt_num_threshold = crop_2d_slices_gt_num_threshold
        self.prevent_disturbance = prevent_disturbance
        self.do_augment = False
        self.use_modified = False
        self.disturbed_idxs = []
        self.augment_at_collate = False
        self.pre_interpolation_factor = pre_interpolation_factor

        #define finished preprocessing states here with subpath and default size
        states = {
            'l1':('L1_original/', (512,512,160)),
            'l2':('L2_resampled_05mm/', (420,420,360)),
            'l3':('L3_coarse_fixed_crop/', (128,128,192)),
            'l4':('L4_fine_localized_crop/', (128,128,128))
        }
        t0 = time.time()
        #choose directory with data according to chosen preprocessing state
        if state not in states: raise Exception("Unknown state. Choose one of: "+str(states.keys))

        state_dir = states[state.lower()][0] #get sub directory

        if not resample: size = states[state.lower()][1] #set size to default defined at top of file

        path = base_dir + state_dir

        #get file list
        if domain.lower() =="ceT1" or domain.lower() =="source":
            directory = "source_training_labeled/"
            add_directory = "__additional_data_source_domain__"
            domain = "ceT1"

        elif domain.lower() =="hrT2" or domain.lower() =="target":
            directory = "target_training_unlabeled/"
            add_directory = "__additional_data_target_domain__"
            domain = "hrT2"

        elif domain.lower() =="validation":
            directory = "target_validation_unlabeled/"

        else:
            raise Exception("Unknown domain. Choose either 'source', 'target' or 'validation'")

        files = sorted(glob.glob(os.path.join(path+directory , "*.nii.gz")))

        if domain == "hrT2":
            files = files+sorted(glob.glob(os.path.join(path+"__omitted_labels_target_training__" , "*.nii.gz")))

        if domain.lower() == "validation":
            files = files+sorted(glob.glob(os.path.join(path+"__omitted_labels_target_validation__" , "*.nii.gz")))

        if use_additional_data and domain.lower() != "validation": #add additional data to file list
            files = files+sorted(glob.glob(os.path.join(path+add_directory , "*.nii.gz")))
            files = [i for i in files if "additionalLabel" not in i] #remove additional label files


        # First read filepaths
        self.img_paths = {}
        self.label_paths = {}

        if debug:
            files = files[:10]


        for _path in files:

            numeric_id = int(re.findall(r'\d+', os.path.basename(_path))[0])
            if "_l.nii.gz" in _path or "_l_Label.nii.gz" in _path:
                lr_id = 'l'
            elif "_r.nii.gz" in _path or "_r_Label.nii.gz" in _path:
                lr_id = 'r'
            else:
                lr_id = ""

            # Generate crossmoda id like 004r
            crossmoda_id = f"{numeric_id:03d}{lr_id}"

            # Skip file if we do not have modified labels for it when external labels were provided
            if modified_3d_label_override:
                if crossmoda_id not in \
                    [self.extract_short_3d_id(var_id) for var_id in modified_3d_label_override.keys()]:
                    continue

            if "Label" in _path:
                self.label_paths[crossmoda_id] = _path

            elif domain in _path:
                self.img_paths[crossmoda_id] = _path

        if ensure_labeled_pairs:
            pair_idxs = set(self.img_paths).intersection(set(self.label_paths))
            self.label_paths = {_id: _path for _id, _path in self.label_paths.items() if _id in pair_idxs}
            self.img_paths = {_id: _path for _id, _path in self.img_paths.items() if _id in pair_idxs}


        # Populate data
        self.img_data_3d = {}
        self.label_data_3d = {}
        self.modified_label_data_3d = {}

        self.img_data_2d = {}
        self.label_data_2d = {}
        self.modified_label_data_2d = {}

        #load data

        print("Loading CrossMoDa {} images and labels...".format(domain))
        id_paths_to_load = list(self.label_paths.items()) + list(self.img_paths.items())

        description = f"{len(self.img_paths)} images, {len(self.label_paths)} labels"

        for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
            # tqdm.write(f"Loading {f}")
            if "Label" in _file:
                tmp = torch.from_numpy(nib.load(_file).get_fdata())

                if resample: #resample image to specified size
                    tmp = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=size,mode='nearest').squeeze()

                if tmp.shape != size: #for size missmatch use symmetric padding with 0
                    difs = [size[0]-tmp.size(0),size[1]-tmp.size(1),size[2]-tmp.size(2)]
                    pad = (difs[-1]//2,difs[-1]-difs[-1]//2,difs[-2]//2,difs[-2]-difs[-2]//2,difs[-3]//2,difs[-3]-difs[-3]//2)
                    tmp = F.pad(tmp,pad)

                if crop_3d_w_dim_range:
                    tmp = tmp[..., crop_3d_w_dim_range[0]:crop_3d_w_dim_range[1]]

                # Only use tumour class, remove TODO
                tmp[tmp==2] = 0
                self.label_data_3d[_3d_id] = tmp.long()

            elif domain in _file:
                tmp = torch.from_numpy(nib.load(_file).get_fdata())

                if resample: #resample image to specified size
                    tmp = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=size,mode='trilinear',align_corners=False).squeeze()

                if tmp.shape != size: #for size missmatch use symmetric padding with 0
                    difs = [size[0]-tmp.size(0),size[1]-tmp.size(1),size[2]-tmp.size(2)]
                    pad = (difs[-1]//2,difs[-1]-difs[-1]//2,difs[-2]//2,difs[-2]-difs[-2]//2,difs[-3]//2,difs[-3]-difs[-3]//2)
                    tmp = F.pad(tmp,pad)

                if crop_3d_w_dim_range:
                    tmp = tmp[..., crop_3d_w_dim_range[0]:crop_3d_w_dim_range[1]]

                if normalize: #normalize image to zero mean and unit std
                    tmp = (tmp - tmp.mean()) / tmp.std()

                self.img_data_3d[_3d_id] = tmp

        # Initialize 3d modified labels as unmodified labels
        for label_id in self.label_data_3d.keys():
            self.modified_label_data_3d[label_id] = self.label_data_3d[label_id]

        # Now inject externally overriden labels (if any)
        if modified_3d_label_override:
            stored_3d_ids = list(self.label_data_3d.keys())

            # Delete all modified labels which have no base data keys
            unmatched_keys = [key for key in modified_3d_label_override.keys() \
                if self.extract_short_3d_id(key) not in stored_3d_ids]
            for del_key in unmatched_keys:
                del modified_3d_label_override[del_key]
            if len(stored_3d_ids) != len(modified_3d_label_override.keys()):
                print(f"Expanding label data with modified_3d_label_override from {len(stored_3d_ids)} to {len(modified_3d_label_override.keys())} labels")

            for _mod_3d_id, modified_label in modified_3d_label_override.items():
                tmp = modified_label

                if resample: #resample image to specified size
                    tmp = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=size,mode='nearest').squeeze()

                if tmp.shape != size: #for size missmatch use symmetric padding with 0
                    difs = [size[0]-tmp.size(0),size[1]-tmp.size(1),size[2]-tmp.size(2)]
                    pad = (difs[-1]//2,difs[-1]-difs[-1]//2,difs[-2]//2,difs[-2]-difs[-2]//2,difs[-3]//2,difs[-3]-difs[-3]//2)
                    tmp = F.pad(tmp,pad)

                if crop_3d_w_dim_range:
                    tmp = tmp[..., crop_3d_w_dim_range[0]:crop_3d_w_dim_range[1]]

                # Only use tumour class, remove TODO
                tmp[tmp==2] = 0
                self.modified_label_data_3d[_mod_3d_id] = tmp.long()

                # Now expand original _3d_ids with _mod_3d_id
                _3d_id = self.extract_short_3d_id(_mod_3d_id)
                self.img_paths[_mod_3d_id] = self.img_paths[_3d_id]
                self.label_paths[_mod_3d_id] = self.label_paths[_3d_id]
                self.img_data_3d[_mod_3d_id] = self.img_data_3d[_3d_id]
                self.label_data_3d[_mod_3d_id] = self.label_data_3d[_3d_id]

            # Delete original 3d_ids as they got expanded
            for del_id in stored_3d_ids:
                del self.img_paths[del_id]
                del self.label_paths[del_id]
                del self.img_data_3d[del_id]
                del self.label_data_3d[del_id]

        # Postprocessing of 3d volumes
        orig_3d_num = len(self.label_data_3d.keys())

        for _3d_id in list(self.label_data_3d.keys()):
            if self.label_data_3d[_3d_id].unique().numel() != 2: #TODO use 3 classes again
                del self.img_data_3d[_3d_id]
                del self.label_data_3d[_3d_id]
                del self.modified_label_data_3d[_3d_id]
            elif "r" in _3d_id:
                self.img_data_3d[_3d_id] = self.img_data_3d[_3d_id].flip(dims=(1,))
                self.label_data_3d[_3d_id] = self.label_data_3d[_3d_id].flip(dims=(1,))
                self.modified_label_data_3d[_3d_id] = self.modified_label_data_3d[_3d_id].flip(dims=(1,))

        if ensure_labeled_pairs:
            labelled_keys = set(self.label_data_3d.keys())
            unlabelled_imgs = set(self.img_data_3d.keys()) - labelled_keys
            unlabelled_modified_labels = set([self.extract_3d_id(key) for key in self.modified_label_data_3d.keys()]) - labelled_keys

            for del_key in unlabelled_imgs:
                del self.img_data_3d[del_key]
            for del_key in unlabelled_modified_labels:
                del self.modified_label_data_3d[del_key]

        if max_load_num:
            for del_key in sorted(list(self.img_data_3d.keys()))[max_load_num:]:
                del self.img_data_3d[del_key]
            for del_key in sorted(list(self.label_data_3d.keys()))[max_load_num:]:
                del self.label_data_3d[del_key]
            for del_key in sorted(list(self.modified_label_data_3d.keys()))[max_load_num:]:
                del self.modified_label_data_3d[del_key]

        postprocessed_3d_num = len(self.label_data_3d.keys())
        print(f"Removed {orig_3d_num - postprocessed_3d_num} 3D images in postprocessing")
        #check for consistency
        print(f"Equal image and label numbers: {set(self.img_data_3d)==set(self.label_data_3d)==set(self.modified_label_data_3d)} ({len(self.img_data_3d)})")

        img_stack = torch.stack(list(self.img_data_3d.values()), dim=0)
        img_mean, img_std = img_stack.mean(), img_stack.std()

        label_stack = torch.stack(list(self.label_data_3d.values()), dim=0)

        print("Image shape: {}, mean.: {:.2f}, std.: {:.2f}".format(img_stack.shape, img_mean, img_std))
        print("Label shape: {}, max.: {}".format(label_stack.shape,torch.max(label_stack)))

        if use_2d_normal_to:
            if use_2d_normal_to == "D":
                slice_dim = -3
            if use_2d_normal_to == "H":
                slice_dim = -2
            if use_2d_normal_to == "W":
                slice_dim = -1

            for _3d_id, image in self.img_data_3d.items():
                for idx, img_slc in [(slice_idx, image.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(image.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.img_data_2d[f"{_3d_id}{use_2d_normal_to}{idx:03d}"] = img_slc

            for _3d_id, label in self.label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.label_data_2d[f"{_3d_id}{use_2d_normal_to}{idx:03d}"] = lbl_slc

            for _3d_id, label in self.modified_label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.modified_label_data_2d[f"{_3d_id}{use_2d_normal_to}{idx:03d}"] = lbl_slc

        # Postprocessing of 2d slices
        orig_2d_num = len(self.label_data_2d.keys())

        for key, label in list(self.label_data_2d.items()):
            uniq_vals = label.unique()

            if sum(label[label > 0]) < self.crop_2d_slices_gt_num_threshold:
                # Delete 2D slices with less than n gt-pixels (but keep 3d data)
                del self.img_data_2d[key]
                del self.label_data_2d[key]
                del self.modified_label_data_2d[key]

        postprocessed_2d_num = len(self.label_data_2d.keys())
        print(f"Removed {orig_2d_num - postprocessed_2d_num} of {orig_2d_num} 2D slices in postprocessing")
        print("Data import finished.")
        print(f"CrossMoDa loader will yield {'2D' if self.use_2d_normal_to else '3D'} samples")

    def extract_3d_id(self, _input):
        # Match sth like 100r:var020
        return "".join(re.findall(r'^(\d{3}[lr])(:var\d{3})?', _input)[0])

    def extract_short_3d_id(self, _input):
        # Match sth like 100r:var020 and returns 100r
        return re.findall(r'^\d{3}[lr]', _input)[0]

    def get_short_3d_ids(self):
        return [self.extract_short_3d_id(_id) for _id in self.get_3d_ids()]

    def get_3d_ids(self):
        return sorted(list(
            set(self.img_data_3d.keys())
            .union(set(self.label_data_3d.keys()))
        ))

    def get_2d_ids(self):
        assert self.use_2d(), "Dataloader does not provide 2D data."
        return sorted(list(
            set(self.img_data_2d.keys())
            .union(set(self.label_data_2d.keys()))
        ))

    def get_id_dicts(self, use_2d_override=None):
        all_3d_ids = self.get_3d_ids()
        id_dicts = []
        if self.use_2d(use_2d_override):
            for _2d_dataset_idx, _2d_id in enumerate(self.get_2d_ids()):
                _3d_id = _2d_id[:-4]
                id_dicts.append(
                    {
                        '2d_id': _2d_id,
                        '2d_dataset_idx': _2d_dataset_idx,
                        '3d_id': _3d_id,
                        '3d_dataset_idx': all_3d_ids.index(_3d_id),
                    }
                )
        else:
            for _3d_dataset_idx, _3d_id in enumerate(self.get_3d_ids()):
                id_dicts.append(
                    {
                        '3d_id': _3d_id,
                        '3d_dataset_idx': all_3d_ids.index(_3d_id),
                    }
                )

        return id_dicts

    def switch_2d_identifiers(self, _2d_identifiers):
        assert self.use_2d(), "Dataloader does not provide 2D data."

        if isinstance(_2d_identifiers, (torch.Tensor, np.ndarray)):
            _2d_identifiers = _2d_identifiers.tolist()
        elif not isinstance(_2d_identifiers, Iterable) or isinstance(_2d_identifiers, str):
            _2d_identifiers = [_2d_identifiers]

        _ids = self.get_2d_ids()
        if all([isinstance(elem, int) for elem in _2d_identifiers]):
            vals = [_ids[elem] for elem in _2d_identifiers]
        elif all([isinstance(elem, str) for elem in _2d_identifiers]):
            vals = [_ids.index(elem) for elem in _2d_identifiers]
        else:
            raise ValueError
        return vals[0] if len(vals) == 1 else vals

    def switch_3d_identifiers(self, _3d_identifiers):
        if isinstance(_3d_identifiers, (torch.Tensor, np.ndarray)):
            _3d_identifiers = _3d_identifiers.tolist()
        elif not isinstance(_3d_identifiers, Iterable) or isinstance(_3d_identifiers, str):
            _3d_identifiers = [_3d_identifiers]

        _ids = self.get_3d_ids()
        if all([isinstance(elem, int) for elem in _3d_identifiers]):
            vals = [_ids[elem] for elem in _3d_identifiers]
        elif all([isinstance(elem, str) for elem in _3d_identifiers]):
            vals = [_ids.index(elem) if elem in _ids else None for elem in _3d_identifiers]
        else:
            raise ValueError
        return vals[0] if len(vals) == 1 else vals

    def get_3d_from_2d_identifiers(self, _2d_identifiers, retrn='id'):
        assert self.use_2d(), "Dataloader does not provide 2D data."
        assert retrn in ['id', 'idx']

        if isinstance(_2d_identifiers, (torch.Tensor, np.ndarray)):
            _2d_identifiers = _2d_identifiers.tolist()
        elif not isinstance(_2d_identifiers, Iterable) or isinstance(_2d_identifiers, str):
            _2d_identifiers = [_2d_identifiers]

        if isinstance(_2d_identifiers[0], int):
            _2d_identifiers = self.switch_2d_identifiers(_2d_identifiers)

        vals = []
        for item in _2d_identifiers:
            _3d_id = self.extract_3d_id(item)
            if retrn == 'id':
                vals.append(_3d_id)
            elif retrn == 'idx':
                vals.append(self.switch_3d_identifiers(_3d_id))

        return vals[0] if len(vals) == 1 else vals

    def use_2d(self, override=None):
        if not self.use_2d_normal_to:
            return False
        elif override is not None:
            return override
        else:
            return True

    def __len__(self, use_2d_override=None):
        if self.use_2d(use_2d_override):
            return len(self.img_data_2d)
        return len(self.img_data_3d)

    def __getitem__(self, dataset_idx, use_2d_override=None):
        use_2d = self.use_2d(use_2d_override)
        if use_2d:
            all_ids = self.get_2d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_2d.get(_id, torch.tensor([]))
            label = self.label_data_2d.get(_id, torch.tensor([]))

            # For 2D crossmoda id cut last 4 "003rW100"
            _3d_id = self.get_3d_from_2d_identifiers(_id)
            image_path = self.img_paths[_3d_id]
            label_path = self.label_paths[_3d_id]

        else:
            all_ids = self.get_3d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_3d.get(_id, torch.tensor([]))
            label = self.label_data_3d.get(_id, torch.tensor([]))

            image_path = self.img_paths[_id]
            label_path = self.label_paths[_id]

        spat_augment_grid = []

        if self.use_modified:
            if use_2d:
                modified_label = self.modified_label_data_2d.get(_id, label.detach().clone())
            else:
                modified_label = self.modified_label_data_3d.get(_id, label.detach().clone())
        else:
            modified_label = label.detach().clone()

        b_image = image.unsqueeze(0).cuda()
        b_label = label.unsqueeze(0).cuda()
        b_modified_label = modified_label.unsqueeze(0).cuda()

        if self.do_augment and not self.augment_at_collate:
            b_image, b_label, b_spat_augment_grid = self.augment(
                b_image, b_label, use_2d, pre_interpolation_factor=self.pre_interpolation_factor
            )
            _, b_modified_label, _ = spatial_augment(
                b_label=b_modified_label, use_2d=use_2d, b_grid_override=b_spat_augment_grid,
                pre_interpolation_factor=self.pre_interpolation_factor
            )
            spat_augment_grid = b_spat_augment_grid.squeeze(0).detach().cpu().clone()

        elif not self.do_augment:
            b_image, b_label = interpolate_sample(b_image, b_label, 2., use_2d)
            _, b_modified_label = interpolate_sample(b_label=b_modified_label, scale_factor=2.,
                use_2d=use_2d)

        image = b_image.squeeze(0).cpu()
        label = b_label.squeeze(0).cpu()
        modified_label = b_modified_label.squeeze(0).cpu()

        if use_2d:
            assert image.dim() == label.dim() == 2
        else:
            assert image.dim() == label.dim() == 3

        return {
            'image': image,
            'label': label,
            'modified_label': modified_label,
            # if disturbance is off, modified label is equals label
            'dataset_idx': dataset_idx,
            'id': _id,
            'image_path': image_path,
            'label_path': label_path,
            'spat_augment_grid': spat_augment_grid
        }

    def get_3d_item(self, _3d_dataset_idx):
        return self.__getitem__(_3d_dataset_idx, use_2d_override=False)

    def get_data(self, use_2d_override=None):
        if self.use_2d(use_2d_override):
            img_stack = torch.stack(list(self.img_data_2d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_2d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_2d.values()), dim=0)
        else:
            img_stack = torch.stack(list(self.img_data_3d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_3d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_3d.values()), dim=0)

        return img_stack, label_stack, modified_label_stack

    def disturb_idxs(self, all_idxs, disturbance_mode, disturbance_strength=1., use_2d_override=None):
        if self.prevent_disturbance:
            warnings.warn("Disturbed idxs shall be set but disturbance is prevented for dataset.")
            return

        use_2d = self.use_2d(use_2d_override)

        if all_idxs is not None:
            if isinstance(all_idxs, (np.ndarray, torch.Tensor)):
                all_idxs = all_idxs.tolist()

            self.disturbed_idxs = all_idxs
        else:
            self.disturbed_idxs = []

        # Reset modified data
        for idx in range(self.__len__(use_2d_override=use_2d)):
            if use_2d:
                label_id = self.get_2d_ids()[idx]
                self.modified_label_data_2d[label_id] = self.label_data_2d[label_id]
            else:
                label_id = self.get_3d_ids()[idx]
                self.modified_label_data_3d[label_id] = self.label_data_3d[label_id]

            # Now apply disturbance
            if idx in self.disturbed_idxs:
                label = self.modified_label_data_2d[label_id].detach().clone()

                with torch_manual_seeded(idx):
                    if str(disturbance_mode)==str(LabelDisturbanceMode.FLIP_ROLL):
                        roll_strength = 10*disturbance_strength
                        if use_2d:
                            modified_label = \
                                torch.roll(
                                    label.transpose(-2,-1),
                                    (
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength)
                                    ),(-2,-1)
                                )
                        else:
                            modified_label = \
                                torch.roll(
                                    label.permute(1,2,0),
                                    (
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength),
                                        int(torch.randn(1)*roll_strength)
                                    ),(-3,-2,-1)
                                )

                    elif str(disturbance_mode)==str(LabelDisturbanceMode.AFFINE):
                        b_modified_label = label.unsqueeze(0).cuda()
                        _, b_modified_label, _ = spatial_augment(b_label=b_modified_label, use_2d=use_2d,
                            bspline_num_ctl_points=6, bspline_strength=0., bspline_probability=0.,
                            affine_strength=0.09*disturbance_strength,
                            add_affine_translation=0.18*disturbance_strength, affine_probability=1.)
                        modified_label = b_modified_label.squeeze(0).cpu()

                    else:
                        raise ValueError(f"Disturbance mode {disturbance_mode} is not implemented.")

                    if use_2d:
                        self.modified_label_data_2d[label_id] = modified_label
                    else:
                        self.modified_label_data_2d[label_id] = modified_label


    def train(self, augment=True, use_modified=True):
        self.do_augment = augment
        self.use_modified = use_modified

    def eval(self, augment=False, use_modified=False):
        self.train(augment, use_modified)

    def set_augment_at_collate(self, augment_at_collate=True):
        self.augment_at_collate = augment_at_collate

    def get_efficient_augmentation_collate_fn(self):
        use_2d = True if self.use_2d_normal_to else False

        def collate_closure(batch):
            batch = torch.utils.data._utils.collate.default_collate(batch)
            if self.augment_at_collate:
                # Augment the whole batch not just one sample
                b_image = batch['image'].cuda()
                b_label = batch['label'].cuda()
                b_image, b_label = self.augment(b_image, b_label, use_2d)
                batch['image'], batch['label'] = b_image.cpu(), b_label.cpu()

            return batch

        return collate_closure

    def augment(self, b_image, b_label, use_2d,
        noise_strength=0.05,
        bspline_num_ctl_points=6, bspline_strength=0.004, bspline_probability=.95,
        affine_strength=0.07, affine_probability=.45,
        pre_interpolation_factor=2.):

        if use_2d:
            assert b_image.dim() == b_label.dim() == 3, \
                f"Augmenting 2D. Input batch of image and " \
                f"label should be BxHxW but are {b_image.shape} and {b_label.shape}"
        else:
            assert b_image.dim() == b_label.dim() == 4, \
                f"Augmenting 3D. Input batch of image and " \
                f"label should be BxDxHxW but are {b_image.shape} and {b_label.shape}"

        b_image = augmentNoise(b_image, strength=noise_strength)
        b_image, b_label, b_spat_augment_grid = spatial_augment(
            b_image, b_label,
            bspline_num_ctl_points=bspline_num_ctl_points, bspline_strength=bspline_strength, bspline_probability=bspline_probability,
            affine_strength=affine_strength, affine_probability=affine_probability,
            pre_interpolation_factor=pre_interpolation_factor, use_2d=use_2d)

        b_label = b_label.long()

        return b_image, b_label, b_spat_augment_grid
