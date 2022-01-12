
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
from curriculum_deeplab.HybridIdLoader import HybridIdLoader



class CrossmodaHybridIdLoader(HybridIdLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_tags = ['background', 'tumour']



def get_crossmoda_data_load_closure(base_dir, domain, state, use_additional_data, size, resample, normalize, crop_3d_w_dim_range, ensure_labeled_pairs, debug):

    def extract_3d_id(_input):
        # Match sth like 100r:var020
        return "".join(re.findall(r'^(\d{3}[lr])(:var\d{3})?', _input)[0])

    def extract_short_3d_id(_input):
        # Match sth like 100r:var020 and returns 100r
        return re.findall(r'^\d{3}[lr]', _input)[0]

    def data_load_closure():
        """
        Function to load dataset with crossMoDa data.
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

        nonlocal base_dir
        nonlocal domain
        nonlocal state
        nonlocal use_additional_data
        nonlocal size
        nonlocal resample
        nonlocal normalize
        nonlocal crop_3d_w_dim_range
        nonlocal ensure_labeled_pairs
        nonlocal debug

        # Define finished preprocessing states here with subpath and default size
        states = {
            'l1':('L1_original/', (512,512,160)),
            'l2':('L2_resampled_05mm/', (420,420,360)),
            'l3':('L3_coarse_fixed_crop/', (128,128,192)),
            'l4':('L4_fine_localized_crop/', (128,128,128))
        }
        t0 = time.time()
        # Choose directory with data according to chosen preprocessing state
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
        img_paths = {}
        label_paths = {}

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

            if "Label" in _path:
                label_paths[crossmoda_id] = _path

            elif domain in _path:
                img_paths[crossmoda_id] = _path

        if ensure_labeled_pairs:
            pair_idxs = set(img_paths).intersection(set(label_paths))
            label_paths = {_id: _path for _id, _path in label_paths.items() if _id in pair_idxs}
            img_paths = {_id: _path for _id, _path in img_paths.items() if _id in pair_idxs}

        img_data_3d = {}
        label_data_3d = {}

        # Load data from files
        print("Loading CrossMoDa {} images and labels...".format(domain))
        id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

        description = f"{len(img_paths)} images, {len(label_paths)} labels"

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
                label_data_3d[_3d_id] = tmp.long()

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

                img_data_3d[_3d_id] = tmp

        return (img_paths, label_paths, img_data_3d, label_data_3d,
            extract_3d_id, extract_short_3d_id)

    return data_load_closure