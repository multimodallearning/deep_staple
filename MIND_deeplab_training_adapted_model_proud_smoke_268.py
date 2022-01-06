# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import time
import random
import glob
import re
from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars(select="* -4"))
import pickle
import copy
from pathlib import Path
from contextlib import contextmanager
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import KFold
from mdl_seg_class.metrics import dice3d, dice2d
from mdl_seg_class.visualization import visualize_seg
from curriculum_deeplab.mindssc import mindssc


print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))


# %%
def in_notebook():
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False

if in_notebook:
    THIS_SCRIPT_DIR = os.path.abspath('')
else:
    THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
print(f"Running in: {THIS_SCRIPT_DIR}")


# %%
def interpolate_sample(b_image=None, b_label=None, scale_factor=1.,
                       yield_2d=False):
    if yield_2d:
        scale = [scale_factor]*2
        im_mode = 'bilinear'
    else:
        scale = [scale_factor]*3
        im_mode = 'trilinear'

    if b_image is not None:
        b_image = F.interpolate(
            b_image.unsqueeze(1), scale_factor=scale, mode=im_mode, align_corners=True,
            recompute_scale_factor=False
        )
        b_image = b_image.squeeze(1)

    if b_label is not None:
        b_label = F.interpolate(
            b_label.unsqueeze(1).float(), scale_factor=scale, mode='nearest',
            recompute_scale_factor=False
        ).long()
        b_label = b_label.squeeze(1)

    return b_image, b_label



def dilate_label_class(b_label, class_max_idx, class_dilate_idx,
                       yield_2d, kernel_sz=3):

    if kernel_sz < 2:
        return b_label

    b_dilated_label = b_label

    b_onehot = torch.nn.functional.one_hot(b_label.long(), class_max_idx+1)
    class_slice = b_onehot[...,class_dilate_idx]

    if yield_2d:
        B, H, W = class_slice.shape
        kernel = torch.ones([kernel_sz,kernel_sz]).long()
        kernel = kernel.view(1,1,kernel_sz,kernel_sz)
        class_slice = torch.nn.functional.conv2d(
            class_slice.view(B,1,H,W), kernel, padding='same')

    else:
        B, D, H, W = class_slice.shape
        kernel = torch.ones([kernel_sz,kernel_sz,kernel_sz])
        kernel = kernel.long().view(1,1,kernel_sz,kernel_sz,kernel_sz)
        class_slice = torch.nn.functional.conv3d(
            class_slice.view(B,1,D,H,W), kernel, padding='same')

    dilated_class_slice = torch.clamp(class_slice.squeeze(0), 0, 1)
    b_dilated_label[dilated_class_slice.bool()] = class_dilate_idx

    return b_dilated_label


def get_batch_dice_per_class(b_dice, class_tags, exclude_bg=True) -> dict:
    score_dict = {}
    for cls_idx, cls_tag in enumerate(class_tags):
        if exclude_bg and cls_idx == 0:
            continue

        if torch.all(torch.isnan(b_dice[:,cls_idx])):
            score = float('nan')
        else:
            score = np.nanmean(b_dice[:,cls_idx]).item()

        score_dict[cls_tag] = score

    return score_dict

def get_batch_dice_over_all(b_dice, exclude_bg=True) -> float:

    start_idx = 1 if exclude_bg else 0
    if torch.all(torch.isnan(b_dice[:,start_idx:])):
        return float('nan')
    return np.nanmean(b_dice[:,start_idx:]).item()



def get_2d_stack_batch_size(b_input_size: torch.Size, stack_dim):
    assert len(b_input_size) == 5, f"Input size must be 5D: BxCxDxHxW but is {b_input_size}"
    if stack_dim == "D":
        return b_input_size[0]*b_input_size[2]
    if stack_dim == "H":
        return b_input_size[0]*b_input_size[3]
    if stack_dim == "W":
        return b_input_size[0]*b_input_size[4]
    else:
        raise ValueError(f"stack_dim '{stack_dim}' must be 'D' or 'H' or 'W'.")



def make_2d_stack_from_3d(b_input, stack_dim):
    assert b_input.dim() == 5, f"Input must be 5D: BxCxDxHxW but is {b_input.shape}"
    B, C, D, H, W = b_input.shape

    if stack_dim == "D":
        return b_input.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
    if stack_dim == "H":
        return b_input.permute(0, 3, 1, 2, 4).reshape(B*H, C, D, W)
    if stack_dim == "W":
        return b_input.permute(0, 4, 1, 2, 3).reshape(B*W, C, D, H)
    else:
        raise ValueError(f"stack_dim '{stack_dim}' must be 'D' or 'H' or 'W'.")



def make_3d_from_2d_stack(b_input, stack_dim, orig_stack_size):
    assert b_input.dim() == 4, f"Input must be 4D: (orig_batch_size/B)xCxSPAT1xSPAT0 but is {b_input.shape}"
    B, C, SPAT1, SPAT0 = b_input.shape
    b_input = b_input.reshape(orig_stack_size, int(B//orig_stack_size), C, SPAT1, SPAT0)

    if stack_dim == "D":
        return b_input.permute(0, 2, 1, 3, 4)
    if stack_dim == "H":
        return b_input.permute(0, 2, 3, 1, 4)
    if stack_dim == "W":
        return b_input.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"stack_dim is '{stack_dim}' but must be 'D' or 'H' or 'W'.")


# %%
def spatial_augment(b_image=None, b_label=None,
    bspline_num_ctl_points=6, bspline_strength=0.005, bspline_probability=.9,
    affine_strength=0.08, add_affine_translation=0., affine_probability=.45,
    pre_interpolation_factor=None,
    yield_2d=False,
    b_grid_override=None):

    """
    2D/3D b-spline augmentation on image and segmentation mini-batch on GPU.
    :input: b_image batch (torch.cuda.FloatTensor), b_label batch (torch.cuda.LongTensor)
    :return: augmented Bx(D)xHxW image batch (torch.cuda.FloatTensor),
    augmented Bx(D)xHxW seg batch (torch.cuda.LongTensor)
    """

    do_bspline = (np.random.rand() < bspline_probability)
    do_affine = (np.random.rand() < affine_probability)

    if pre_interpolation_factor:
        b_image, b_label = interpolate_sample(b_image, b_label, pre_interpolation_factor, yield_2d)

    KERNEL_SIZE = 3

    if b_image is None:
        common_shape = b_label.shape
        common_device = b_label.device

    elif b_label is None:
        common_shape = b_image.shape
        common_device = b_image.device
    else:
        assert b_image.shape == b_label.shape, \
            f"Image and label shapes must match but are {b_image.shape} and {b_label.shape}."
        common_shape = b_image.shape
        common_device = b_image.device

    if b_grid_override is None:
        if yield_2d:
            assert len(common_shape) == 3, \
                f"Augmenting 2D. Input batch " \
                f"should be BxHxW but is {common_shape}."
            B,H,W = common_shape

            identity = torch.eye(2,3).expand(B,2,3).to(common_device)
            id_grid = F.affine_grid(identity, torch.Size((B,2,H,W)),
                align_corners=False)

            grid = id_grid

            if do_bspline:
                bspline = torch.nn.Sequential(
                    nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
                ).to(common_device)
                # Add an extra *.5 factor to dim strength to make strength fit 3D case
                dim_strength = (torch.tensor([H,W]).float()*bspline_strength*.5).to(common_device)
                rand_control_points = dim_strength.view(1,2,1,1) * \
                    (
                        torch.randn(B, 2, bspline_num_ctl_points, bspline_num_ctl_points)
                    ).to(common_device)

                bspline_disp = bspline(rand_control_points)
                bspline_disp = torch.nn.functional.interpolate(
                    bspline_disp, size=(H,W), mode='bilinear', align_corners=True
                ).permute(0,2,3,1)

                grid += bspline_disp

            if do_affine:
                affine_matrix = (torch.eye(2,3).unsqueeze(0) + \
                    affine_strength * torch.randn(B,2,3)).to(common_device)
                # Add additional x,y offset
                alpha = np.random.rand() * 2 * np.pi
                offset_dir =  torch.tensor([np.cos(alpha), np.sin(alpha)])
                affine_matrix[:,:,-1] = add_affine_translation * offset_dir
                affine_disp = F.affine_grid(affine_matrix, torch.Size((B,1,H,W)),
                                        align_corners=False)
                grid += (affine_disp-id_grid)

        else:
            assert len(common_shape) == 4, \
                f"Augmenting 3D. Input batch " \
                f"should be BxDxHxW but is {common_shape}."
            B,D,H,W = common_shape

            identity = torch.eye(3,4).expand(B,3,4).to(common_device)
            id_grid = F.affine_grid(identity, torch.Size((B,3,D,H,W)),
                align_corners=False)

            grid = id_grid

            if do_bspline:
                bspline = torch.nn.Sequential(
                    nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
                ).to(b_image.device)
                dim_strength = (torch.tensor([D,H,W]).float()*bspline_strength).to(common_device)

                rand_control_points = dim_strength.view(1,3,1,1,1)  * \
                    (
                        torch.randn(B, 3, bspline_num_ctl_points, bspline_num_ctl_points, bspline_num_ctl_points)
                    ).to(b_image.device)

                bspline_disp = bspline(rand_control_points)

                bspline_disp = torch.nn.functional.interpolate(
                    bspline_disp, size=(D,H,W), mode='trilinear', align_corners=True
                ).permute(0,2,3,4,1)

                grid += bspline_disp

            if do_affine:
                affine_matrix = (torch.eye(3,4).unsqueeze(0) + \
                    affine_strength * torch.randn(B,3,4)).to(common_device)

                # Add additional x,y,z offset
                theta = np.random.rand() * 2 * np.pi
                phi = np.random.rand() * 2 * np.pi
                offset_dir =  torch.tensor([
                    np.cos(phi)*np.sin(theta),
                    np.sin(phi)*np.sin(theta),
                    np.cos(theta)])
                affine_matrix[:,:,-1] = add_affine_translation * offset_dir

                affine_disp = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)),
                                        align_corners=False)

                grid += (affine_disp-id_grid)
    else:
        # Override grid with external value
        grid = b_grid_override

    if b_image is not None:
        b_image_out = F.grid_sample(
            b_image.unsqueeze(1).float(), grid,
            padding_mode='border', align_corners=False)
        b_image_out = b_image_out.squeeze(1)
    else:
        b_image_out = None

    if b_label is not None:
        b_label_out = F.grid_sample(
            b_label.unsqueeze(1).float(), grid,
            mode='nearest', align_corners=False)
        b_label_out = b_label_out.squeeze(1).long()
    else:
        b_label_out = None

    b_out_grid = grid


    return b_image_out, b_label_out, b_out_grid



def augmentNoise(b_image, strength=0.05):
    return b_image + strength*torch.randn_like(b_image)


@contextmanager
def torch_manual_seeded(seed):
    saved_state = torch.get_rng_state()
    yield
    torch.set_rng_state(saved_state)



# %%
class CrossMoDa_Data(Dataset):
    def __init__(self,
        base_dir, domain, state,
        ensure_labeled_pairs=True, use_additional_data=False, resample=True,
        size:tuple=(96,96,60), normalize:bool=True,
        max_load_num=None, crop_3d_w_dim_range=None, crop_2d_slices_gt_num_threshold=None,
        modified_3d_label_override=None, prevent_disturbance=False,
        yield_2d_normal_to=None, flip_r_samples=True,
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
                yield_2d_normal_to (bool):

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
        self.yield_2d_normal_to = yield_2d_normal_to
        self.crop_2d_slices_gt_num_threshold = crop_2d_slices_gt_num_threshold
        self.prevent_disturbance = prevent_disturbance
        self.do_augment = False
        self.use_modified = False
        self.disturbed_idxs = []
        self.augment_at_collate = False

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
            files = files[:4]


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
                if crossmoda_id not in [var_id[:4] for var_id in modified_3d_label_override.keys()]:
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
            unmatched_keys = [key for key in modified_3d_label_override.keys() if key[:4] not in stored_3d_ids]
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
                _3d_id = _mod_3d_id[:4]
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
            unlabelled_modified_labels = set([key[:4] for key in self.modified_label_data_3d.keys()]) - labelled_keys

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

        if yield_2d_normal_to:
            if yield_2d_normal_to == "D":
                slice_dim = -3
            if yield_2d_normal_to == "H":
                slice_dim = -2
            if yield_2d_normal_to == "W":
                slice_dim = -1

            for _3d_id, image in self.img_data_3d.items():
                for idx, img_slc in [(slice_idx, image.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(image.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.img_data_2d[f"{_3d_id}{yield_2d_normal_to}{idx:03d}"] = img_slc

            for _3d_id, label in self.label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.label_data_2d[f"{_3d_id}{yield_2d_normal_to}{idx:03d}"] = lbl_slc

            for _3d_id, label in self.modified_label_data_3d.items():
                for idx, lbl_slc in [(slice_idx, label.select(slice_dim, slice_idx)) \
                                     for slice_idx in range(label.shape[slice_dim])]:
                    # Set data view for crossmoda id like "003rW100"
                    self.modified_label_data_2d[f"{_3d_id}{yield_2d_normal_to}{idx:03d}"] = lbl_slc

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
        print(f"CrossMoDa loader will yield {'2D' if self.yield_2d_normal_to else '3D'} samples")

    def get_3d_ids(self):
        return sorted(list(
            set(self.img_data_3d.keys())
            .union(set(self.label_data_3d.keys()))
        ))

    def get_2d_ids(self):
        return sorted(list(
            set(self.img_data_2d.keys())
            .union(set(self.label_data_2d.keys()))
        ))

    def get_id_dicts(self):

        all_3d_ids = self.get_3d_ids()
        id_dicts = []

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

        return id_dicts

    def __len__(self, yield_2d_override=None):
        if yield_2d_override == None:
            # Here getting 2D or 3D data length
            yield_2d = True if self.yield_2d_normal_to else False
        else:
            yield_2d = yield_2d_override

        if yield_2d:
            return len(self.img_data_2d)

        return len(self.img_data_3d)

    def __getitem__(self, dataset_idx, yield_2d_override=None):

        if yield_2d_override == None:
            # Here getting 2D or 3D data can be overridden
            yield_2d = True if self.yield_2d_normal_to else False
        else:
            yield_2d = yield_2d_override

        if yield_2d:
            all_ids = self.get_2d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_2d.get(_id, torch.tensor([]))
            label = self.label_data_2d.get(_id, torch.tensor([]))

            # For 2D crossmoda id cut last 4 "003rW100"
            image_path = self.img_paths[_id[:-4]]
            label_path = self.label_paths[_id[:-4]]

        else:
            all_ids = self.get_3d_ids()
            _id = all_ids[dataset_idx]
            image = self.img_data_3d.get(_id, torch.tensor([]))
            label = self.label_data_3d.get(_id, torch.tensor([]))

            image_path = self.img_paths[_id]
            label_path = self.label_paths[_id]

        spat_augment_grid = []

        if self.use_modified:
            if yield_2d:
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
                b_image, b_label, yield_2d, pre_interpolation_factor=2.
            )
            _, b_modified_label, _ = spatial_augment(
                b_label=b_modified_label, yield_2d=yield_2d, b_grid_override=b_spat_augment_grid,
                pre_interpolation_factor=2.
            )
            spat_augment_grid = b_spat_augment_grid.squeeze(0).detach().cpu().clone()

        elif not self.do_augment:
            b_image, b_label = interpolate_sample(b_image, b_label, 2., yield_2d)
            _, b_modified_label = interpolate_sample(b_label=b_modified_label, scale_factor=2.,
                yield_2d=yield_2d)

        image = b_image.squeeze(0).cpu()
        label = b_label.squeeze(0).cpu()
        modified_label = b_modified_label.squeeze(0).cpu()

        if yield_2d:
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
        return self.__getitem__(_3d_dataset_idx, yield_2d_override=False)

    def get_data(self, yield_2d_override=None):
        if yield_2d_override == None:
            # Here getting 2D or 3D data can be overridden
            yield_2d = True if self.yield_2d_normal_to else False
        else:
            yield_2d = yield_2d_override

        if yield_2d:
            img_stack = torch.stack(list(self.img_data_2d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_2d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_2d.values()), dim=0)
        else:
            img_stack = torch.stack(list(self.img_data_3d.values()), dim=0)
            label_stack = torch.stack(list(self.label_data_3d.values()), dim=0)
            modified_label_stack = torch.stack(list(self.modified_label_data_3d.values()), dim=0)

        return img_stack, label_stack, modified_label_stack

    def disturb_idxs(self, all_idxs, disturbance_mode, disturbance_strength=1., yield_2d_override=None):
        if self.prevent_disturbance:
            warnings.warn("Disturbed idxs shall be set but disturbance is prevented for dataset.")
            return

        if yield_2d_override == None:
            # Here getting 2D or 3D data can be overridden
            yield_2d = True if self.yield_2d_normal_to else False
        else:
            yield_2d = yield_2d_override

        if all_idxs is not None:
            if isinstance(all_idxs, (np.ndarray, torch.Tensor)):
                all_idxs = all_idxs.tolist()

            self.disturbed_idxs = all_idxs
        else:
            self.disturbed_idxs = []

        # Reset modified data
        for idx in range(self.__len__(yield_2d_override=yield_2d)):
            if yield_2d:
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
                        if yield_2d:
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
                        _, b_modified_label, _ = spatial_augment(b_label=b_modified_label, yield_2d=yield_2d,
                            bspline_num_ctl_points=6, bspline_strength=0., bspline_probability=0.,
                            affine_strength=0.09*disturbance_strength,
                            add_affine_translation=0.18*disturbance_strength, affine_probability=1.)
                        modified_label = b_modified_label.squeeze(0).cpu()

                    else:
                        raise ValueError(f"Disturbance mode {disturbance_mode} is not implemented.")

                    if yield_2d:
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
        yield_2d = True if self.yield_2d_normal_to else False

        def collate_closure(batch):
            batch = torch.utils.data._utils.collate.default_collate(batch)
            if self.augment_at_collate:
                # Augment the whole batch not just one sample
                b_image = batch['image'].cuda()
                b_label = batch['label'].cuda()
                b_image, b_label = self.augment(b_image, b_label, yield_2d)
                batch['image'], batch['label'] = b_image.cpu(), b_label.cpu()

            return batch

        return collate_closure

    def augment(self, b_image, b_label, yield_2d,
        noise_strength=0.05,
        bspline_num_ctl_points=6, bspline_strength=0.004, bspline_probability=.95,
        affine_strength=0.07, affine_probability=.45,
        pre_interpolation_factor=2.):

        if yield_2d:
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
            pre_interpolation_factor=2., yield_2d=yield_2d)

        b_label = b_label.long()

        return b_image, b_label, b_spat_augment_grid

# %%

from enum import Enum, auto
class LabelDisturbanceMode(Enum):
    FLIP_ROLL = auto()
    AFFINE = auto()

class DataParamMode(Enum):
    INSTANCE_PARAMS = auto()
    GRIDDED_INSTANCE_PARAMS = auto()
    DISABLED = auto()

class DotDict(dict):
    """dot.notation access to dictionary attributes
        See https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config_dict = DotDict({
    'num_folds': 3,
    'only_first_fold': True,
    # 'fold_override': 0,
    # 'checkpoint_epx': 0,

    'use_mind': False,
    'epochs': 40,

    'batch_size': 32,
    'val_batch_size': 1,

    'dataset': 'crossmoda',
    'reg_state': None,
    'train_set_max_len': None,
    'crop_3d_w_dim_range': (45, 95),
    'crop_2d_slices_gt_num_threshold': 0,
    'yield_2d_normal_to': "W",

    'lr': 0.0005,
    'use_cosine_annealing': True,

    # Data parameter config
    'data_param_mode': DataParamMode.INSTANCE_PARAMS,
    'init_inst_param': 0.0,
    'lr_inst_param': 0.1,
    'use_risk_regularization': True,

    'grid_size_y': 64,
    'grid_size_x': 64,
    # ),

    'save_every': 200,
    'mdl_save_prefix': 'data/models',

    'do_plot': False,
    'debug': False,
    'wandb_mode': 'disabled',
    'checkpoint_name': None,
    'do_sweep': False,

    'disturbance_mode': LabelDisturbanceMode.AFFINE,
    'disturbance_strength': 2.,
    'disturbed_percentage': .3,
    'start_disturbing_after_ep': 0,

    'start_dilate_kernel_sz': 1
})

# %%
def prepare_data(config):
    if config.reg_state:
        print("Loading registered data.")

        REG_STATES = ["combined", "best_1", "best_n", "multiple", "mix_combined_best", "best", "cummulate_combined_best"]
        if config.reg_state in REG_STATES:
            pass
        else:
            raise Exception(f"Unknown registration version. Choose one of {REG_STATES}")

        label_data_left = torch.load('./data/optimal_reg_left.pth')
        label_data_right = torch.load('./data/optimal_reg_right.pth')

        loaded_identifier = label_data_left['valid_left_t1'] + label_data_right['valid_right_t1']

        if config.reg_state == "mix_combined_best":
            perm = np.random.permutation(len(loaded_identifier))
            _clen = int(.5*len(loaded_identifier))
            best_choice = perm[:_clen]
            combined_choice = perm[_clen:]

            best_label_data = torch.cat([label_data_left['best_all'][:44], label_data_right['best_all'][:63]], dim=0)[best_choice]
            combined_label_data = torch.cat([label_data_left['combined_all'][:44], label_data_right['combined_all'][:63]], dim=0)[combined_choice]
            label_data = torch.zeros([107,128,128,128])
            label_data[best_choice] = best_label_data
            label_data[combined_choice] = combined_label_data
            loaded_identifier = [_id+':var000' for _id in loaded_identifier]

        elif config.reg_state == "cummulate_combined_best":
            best_label_data = torch.cat([label_data_left['best_all'][:44], label_data_right['best_all'][:63]], dim=0)
            combined_label_data = torch.cat([label_data_left['combined_all'][:44], label_data_right['combined_all'][:63]], dim=0)
            label_data = torch.cat([best_label_data, combined_label_data])
            loaded_identifier = [_id+':var000' for _id in loaded_identifier] + [_id+':var001' for _id in loaded_identifier]

        else:
            label_data = torch.cat([label_data_left[config.reg_state+'_all'][:44], label_data_right[config.reg_state+'_all'][:63]], dim=0)
            loaded_identifier = [_id+':var000' for _id in loaded_identifier]

        modified_3d_label_override = {}
        for idx, identifier in enumerate(loaded_identifier):
            nl_id = int(re.findall(r'\d+', identifier)[0])
            var_id = int(re.findall(r':var(\d+)$', identifier)[0])
            lr_id = re.findall(r'([lr])\.nii\.gz', identifier)[0]

            crossmoda_var_id = f"{nl_id:03d}{lr_id}:var{var_id:03d}"

            modified_3d_label_override[crossmoda_var_id] = label_data[idx]

        prevent_disturbance = True

    else:
        modified_3d_label_override = None
        prevent_disturbance = False

    if config.dataset == 'crossmoda':
        training_dataset = CrossMoDa_Data("/share/data_supergrover1/weihsbach/shared_data/tmp/CrossMoDa/",
            domain="source", state="l4", size=(128, 128, 128),
            ensure_labeled_pairs=True,
            max_load_num=config['train_set_max_len'],
            crop_3d_w_dim_range=config['crop_3d_w_dim_range'], crop_2d_slices_gt_num_threshold=config['crop_2d_slices_gt_num_threshold'],
            yield_2d_normal_to=config['yield_2d_normal_to'],
            modified_3d_label_override=modified_3d_label_override, prevent_disturbance=prevent_disturbance,
            debug=config['debug']
        )
        training_dataset.eval()
        print(f"Nonzero slices: " \
            f"{sum([b['label'].unique().numel() > 1 for b in training_dataset])/len(training_dataset)*100}%"
        )
        # validation_dataset = CrossMoDa_Data("/share/data_supergrover1/weihsbach/shared_data/tmp/CrossMoDa/",
        #     domain="validation", state="l4", ensure_labeled_pairs=True)
        # target_dataset = CrossMoDa_Data("/share/data_supergrover1/weihsbach/shared_data/tmp/CrossMoDa/",
        #     domain="target", state="l4", ensure_labeled_pairs=True)

    elif config['dataset'] == 'organmnist3d':
        training_dataset = WrapperOrganMNIST3D(
            split='train', root='./data/medmnist', download=True, normalize=True,
            max_load_num=300, crop_3d_w_dim_range=None,
            disturbed_idxs=None, yield_2d_normal_to='W'
        )
        print(training_dataset.mnist_set.info)
        print("Classes: ", training_dataset.label_tags)
        print("Samples: ", len(training_dataset))

    return training_dataset

# %%
if False:
    training_dataset = prepare_data(config_dict)
    _, all_labels, _ = training_dataset.get_data(yield_2d_override=False)
    print(all_labels.shape)
    sum_over_w = torch.sum(all_labels, dim=(0,1,2))
    plt.xlabel("W")
    plt.ylabel("ground truth>0")
    plt.plot(sum_over_w);

# %%
if config_dict['do_plot']:
    training_dataset = prepare_data(config_dict)
    # Print bare 2D data
    # print("Displaying 2D bare sample")
    # for img, label in zip(training_dataset.img_data_2d.values(),
    #                       training_dataset.label_data_2d.values()):
    #     display_seg(in_type="single_2D",
    #                 img=img.unsqueeze(0),
    #                 ground_truth=label,
    #                 crop_to_non_zero_gt=True,
    #                 alpha_gt = .3)

    # Print transformed 2D data
    training_dataset.train(use_modified=False, augment=True)
    print(training_dataset.disturbed_idxs)

    print("Displaying 2D training sample")
    for dist_stren in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
        print(dist_stren)
        training_dataset.disturb_idxs(list(range(0,20,2)),
            disturbance_mode=LabelDisturbanceMode.AFFINE,
            disturbance_strength=dist_stren
        )
        img_stack = []
        label_stack = []
        mod_label_stack = []

        for sample in (training_dataset[idx] for idx in range(20)):
            img_stack.append(sample['image'])
            label_stack.append(sample['label'])
            mod_label_stack.append(sample['modified_label'])

        # Change label num == hue shift for display
        img_stack = torch.stack(img_stack).unsqueeze(1)
        label_stack = torch.stack(label_stack)
        mod_label_stack = torch.stack(mod_label_stack)

        mod_label_stack*=4

        visualize_seg(in_type="batch_2D",
            img=img_stack,
            # ground_truth=label_stack,
            seg=(mod_label_stack-label_stack).abs(),
            # crop_to_non_zero_gt=True,
            crop_to_non_zero_seg=True,
            alpha_seg = .6,
            file_path=f'out{dist_stren}.png'
        )

    # Print transformed 3D data
    # training_dataset.train()
    # print("Displaying 3D training sample")
    # leng = 1# training_dataset.__len__(yield_2d_override=False)
    # for sample in (training_dataset.get_3d_item(idx) for idx in range(leng)):
    #     # training_dataset.set_dilate_kernel_size(1)
    #     visualize_seg(in_type="single_3D", reduce_dim="W",
    #                 img=sample['image'].unsqueeze(0),
    #                 ground_truth=sample['label'],
    #                 crop_to_non_zero_gt=True,
    #                 alpha_gt = .3)

#         # training_dataset.set_dilate_kernel_size(7)
#         display_seg(in_type="single_3D", reduce_dim="W",
#                     img=sample['image'].unsqueeze(0),
#                     ground_truth=sample['modified_label'],
#                     crop_to_non_zero_gt=True,
#                     alpha_gt = .3)

    for sidx in [0,]:
        print(f"Sample {sidx}:")

        training_dataset.eval()
        sample_eval = training_dataset.get_3d_item(sidx)

        visualize_seg(in_type="single_3D", reduce_dim="W",
                    img=sample_eval['image'].unsqueeze(0),
                    ground_truth=sample_eval['label'],
                    crop_to_non_zero_gt=True,
                    alpha_gt = .3)

        visualize_seg(in_type="single_3D", reduce_dim="W",
                    img=sample_eval['image'].unsqueeze(0),
                    ground_truth=sample_eval['label'],
                    crop_to_non_zero_gt=True,
                    alpha_gt = .0)

        training_dataset.train()
        print("Train sample with ground-truth overlay")
        sample_train = training_dataset.get_3d_item(sidx)
        print(sample_train['label'].unique())
        visualize_seg(in_type="single_3D", reduce_dim="W",
                    img=sample_train['image'].unsqueeze(0),
                    ground_truth=sample_train['label'],
                    crop_to_non_zero_gt=True,
                    alpha_gt=.3)

        print("Eval/train diff with diff overlay")
        visualize_seg(in_type="single_3D", reduce_dim="W",
                    img=(sample_eval['image'] - sample_train['image']).unsqueeze(0),
                    ground_truth=(sample_eval['label'] - sample_train['label']).clamp(min=0),
                    crop_to_non_zero_gt=True,
                    alpha_gt = .3)

    train_plotset = (training_dataset.get_3d_item(idx) for idx in (55, 81, 63))
    for sample in train_plotset:
        print(f"Sample {sample['dataset_idx']}:")
        display_seg(in_type="single_3D", reduce_dim="W",
            img=sample_eval['image'].unsqueeze(0),
            ground_truth=sample_eval['label'],
            crop_to_non_zero_gt=True,
            alpha_gt = .6)
        display_seg(in_type="single_3D", reduce_dim="W",
            img=sample_eval['image'].unsqueeze(0),
            ground_truth=sample_eval['label'],
            crop_to_non_zero_gt=True,
            alpha_gt = .0)

# %%
#Add functions to replace modules of a model

import functools
MOD_GET_FN = lambda self, key: self[int(key)] if isinstance(self, nn.Sequential) \
                                              else getattr(self, key)

def get_module(module, keychain):
    """Retrieves any module inside a pytorch module for a given keychain.
       module.named_ to retrieve valid keychains for layers.
    """

    return functools.reduce(MOD_GET_FN, keychain.split('.'), module)

def set_module(module, keychain, replacee):
    """Replaces any module inside a pytorch module for a given keychain with "replacee".
       Use get_named_layers_leaves(module) to retrieve valid keychains for layers.
    """

    key_list = keychain.split('.')
    root = functools.reduce(MOD_GET_FN, key_list[:-1], module)
    leaf = key_list[-1]
    if isinstance(root, nn.Sequential):
        root[int(leaf)] = replacee
    else:
        setattr(root, leaf, replacee)


# %%
def save_model(_path, **statefuls):
    _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()
    _path.mkdir(exist_ok=True, parents=True)

    for name, stful in statefuls.items():
        if stful != None:
            torch.save(stful.state_dict(), _path.joinpath(name+'.pth'))



def get_model(config, dataset_len, num_classes, _path=None, device='cpu'):
    _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    if config.use_mind:
        in_channels = 12
    else:
        in_channels = 1

    lraspp = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        pretrained=False, progress=True, num_classes=num_classes
    )
    set_module(lraspp, 'backbone.0.0',
        torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2),
                        padding=(1, 1), bias=False)
    )
    # set_module(lraspp, 'classifier.scale.2',
    #     torch.nn.Identity()
    # )
    lraspp.register_parameter('sigmoid_offset', nn.Parameter(torch.tensor([0.])))
    lraspp.to(device)
    print(f"Param count lraspp: {sum(p.numel() for p in lraspp.parameters())}")

    optimizer = torch.optim.AdamW(lraspp.parameters(), lr=config.lr)
    scaler = amp.GradScaler()

    # Add data paramters embedding and optimizer
    if config.data_param_mode == str(DataParamMode.INSTANCE_PARAMS):
        embedding = nn.Embedding(dataset_len, 1, sparse=True)
        # p_offset = torch.zeros(1, layout=torch.strided, requires_grad=True)
        # p_offset.grad = torch.sparse_coo_tensor([[0]], 1., size=(1,))

        # embedding.register_parameter('sigmoid_offset', nn.Parameter(torch.tensor([0.])))
        # embedding.sigmoid_offset.register_hook(lambda grad: torch.sparse_coo_tensor([[0]], grad, size=(1,)))
        embedding = embedding.to(device)

    elif config.data_param_mode == str(DataParamMode.GRIDDED_INSTANCE_PARAMS):
        embedding = nn.Embedding(dataset_len*config.grid_size_y*config.grid_size_x, 1, sparse=True).to(device)
    else:
        embedding = None


    if str(config.data_param_mode) != str(DataParamMode.DISABLED):
        optimizer_dp = torch.optim.SparseAdam(
            embedding.parameters(), lr=config.lr_inst_param,
            betas=(0.9, 0.999), eps=1e-08)
        torch.nn.init.normal_(embedding.weight.data, mean=config.init_inst_param, std=0.01)

        if _path and _path.is_dir():
            print(f"Loading embedding and dp_optimizer from {_path}")
            optimizer_dp.load_state_dict(torch.load(_path.joinpath('optimizer_dp.pth'), map_location=device))
            embedding.load_state_dict(torch.load(_path.joinpath('embedding.pth'), map_location=device))

        print(f"Param count embedding: {sum(p.numel() for p in embedding.parameters())}")

    else:
        optimizer_dp = None

    if _path and _path.is_dir():
        print(f"Loading lr-aspp model, optimizers and grad scalers from {_path}")
        lraspp.load_state_dict(torch.load(_path.joinpath('lraspp.pth'), map_location=device))
        optimizer.load_state_dict(torch.load(_path.joinpath('optimizer.pth'), map_location=device))
        scaler.load_state_dict(torch.load(_path.joinpath('scaler.pth'), map_location=device))
    else:
        print("Generating fresh lr-aspp model, optimizer and grad scaler.")

    return (lraspp, optimizer, optimizer_dp, embedding, scaler)


# %%
def get_global_idx(fold_idx, epoch_idx, max_epochs):
    # Get global index e.g. 2250 for fold_idx=2, epoch_idx=250 @ max_epochs<1000
    return 10**len(str(int(max_epochs)))*fold_idx + epoch_idx



def log_data_parameters(log_path, parameter_idxs, parameters):
    data = [[idx, param] for (idx, param) in \
        zip(parameter_idxs, torch.exp(parameters).tolist())]

    table = wandb.Table(data=data, columns = ["parameter_idx", "value"])
    wandb.log({log_path:wandb.plot.bar(table, "parameter_idx", "value", title=log_path)})



def calc_inst_parameters_in_target_pos_ratio(dpm, disturbed_inst_idxs, target_pos='min'):

    assert target_pos == 'min' or target_pos == 'max', "Value of target_pos must be 'min' or 'max'."
    descending = False if target_pos == 'min' else True

    target_len = len(disturbed_inst_idxs)

    disturbed_params = dpm.get_parameter_list(inst_keys=disturbed_inst_idxs)
    all_params = sorted(dpm.get_parameter_list(inst_keys='all'), reverse=descending)
    target_param_ids = [id(param) for param in all_params[:target_len]]

    ratio = [1. for param in disturbed_params if id(param) in target_param_ids]
    ratio = sum(ratio)/target_len
    return ratio

def log_data_parameter_stats(log_path, epx, data_parameters):
    """Log stats for data parameters on wandb."""
    data_parameters = data_parameters.exp()
    wandb.log({f'{log_path}/highest': torch.max(data_parameters).item()}, step=epx)
    wandb.log({f'{log_path}/lowest': torch.min(data_parameters).item()}, step=epx)
    wandb.log({f'{log_path}/mean': torch.mean(data_parameters).item()}, step=epx)
    wandb.log({f'{log_path}/std': torch.std(data_parameters).item()}, step=epx)



def reset_determinism():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)




def log_class_dices(log_prefix, log_postfix, class_dices, log_idx):
    if not class_dices:
        return

    for cls_name in class_dices[0].keys():
        log_path = f"{log_prefix}{cls_name}{log_postfix}"

        cls_dices = list(map(lambda dct: dct[cls_name], class_dices))
        mean_per_class =np.nanmean(cls_dices)
        print(log_path, f"{mean_per_class*100:.2f}%")
        wandb.log({log_path: mean_per_class}, step=log_idx)


# %%
def map_embedding_idxs(idxs, grid_size_y, grid_size_x):
    with torch.no_grad():
        t_sz = grid_size_y * grid_size_x
        return ((idxs*t_sz).long().repeat(t_sz).view(t_sz, idxs.numel())+torch.tensor(range(t_sz)).to(idxs).view(t_sz,1)).permute(1,0).reshape(-1)

def inference_wrap(lraspp, b_img, use_mind=True):
    with torch.inference_mode():
        if use_mind:
            b_out = lraspp(
                mindssc(b_img.view(1,1,1,b_img.shape[-2],b_img.shape[-1]).float()).squeeze(2)
            )['out']
        else:
            b_out = lraspp(b_img.view(1,1,b_img.shape[-2],b_img.shape[-1]).float())['out']

        b_out = b_out.argmax(1)
        return b_out

def train_DL(run_name, config, training_dataset):
    reset_determinism()

    # Configure folds
    kf = KFold(n_splits=config.num_folds)
    kf.get_n_splits(training_dataset)

    fold_iter = enumerate(kf.split(training_dataset))

    if config.get('fold_override', None):
        selected_fold = config.get('fold_override', 0)
        fold_iter = list(fold_iter)[selected_fold:selected_fold+1]
    elif config.only_first_fold:
        fold_iter = list(fold_iter)[0:1]

    if config.wandb_mode != 'disabled':
        # Log dataset info
        training_dataset.eval()
        dataset_info = [[smp['dataset_idx'], smp['id'], smp['image_path'], smp['label_path']] \
                        for smp in training_dataset]
        wandb.log({'datasets/training_dataset':wandb.Table(columns=['dataset_idx', 'id', 'image', 'label'], data=dataset_info)}, step=0)

    fold_means_no_bg = []

    for fold_idx, (train_idxs, val_idxs) in fold_iter:
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)

        # Training happens in 2D, validation happens in 3D:
        # Read 2D dataset idxs which are used for training,
        # get their 3D super-ids by 3d dataset length
        # and substract these from all 3D ids to get val_3d_idxs
        trained_3d_dataset_idxs = {dct['3d_dataset_idx'] \
             for dct in training_dataset.get_id_dicts() if dct['2d_dataset_idx'] in train_idxs.tolist()}
        val_3d_idxs = set(range(training_dataset.__len__(yield_2d_override=False))) - trained_3d_dataset_idxs
        val_3d_long_ids = {dct['3d_id'] \
             for dct in training_dataset.get_id_dicts() if dct['3d_dataset_idx'] in val_3d_idxs}
        val_3d_short_ids = set([_id[:4] for _id in val_3d_long_ids])

        # Get only unique 3D val images and labels even if multiple
        # lables for one 3D image exist
        unique_img_val_3d_ids = []
        for short_id in val_3d_short_ids:
            for long_id in val_3d_long_ids:
                if short_id in long_id:
                    unique_img_val_3d_ids.append(long_id)
                    break

        print("Will run validation with these 3D samples:", val_3d_idxs)

        _, _, all_modified_segs = training_dataset.get_data()

        non_empty_train_idxs = train_idxs[(all_modified_segs[train_idxs].sum(dim=(-2,-1)) > 0)]

        ### Disturb dataset (only non-emtpy idxs)###
        proposed_disturbed_idxs = np.random.choice(non_empty_train_idxs, size=int(len(non_empty_train_idxs)*config.disturbed_percentage), replace=False)
        proposed_disturbed_idxs = torch.tensor(proposed_disturbed_idxs)
        training_dataset.disturb_idxs(proposed_disturbed_idxs,
            disturbance_mode=config.disturbance_mode,
            disturbance_strength=config.disturbance_strength
        )

        disturbed_bool_vect = torch.zeros(len(training_dataset))
        disturbed_bool_vect[training_dataset.disturbed_idxs] = 1.

        clean_idxs = train_idxs[np.isin(train_idxs, training_dataset.disturbed_idxs, invert=True)]
        print("Disturbed indexes:", sorted(training_dataset.disturbed_idxs))

        if clean_idxs.numel() < 200:
            print(f"Clean indexes: {sorted(clean_idxs.tolist())}")

        wandb.log({f'datasets/disturbed_idxs_fold{fold_idx}':wandb.Table(columns=['train_idxs'], data=[[idx] for idx in training_dataset.disturbed_idxs])},
            step=get_global_idx(fold_idx, 0, config.epochs))

        ### Configure MIND ###
        if config.use_mind:
            in_channels = 12
        else:
            in_channels = 1

        class_weights = 1/(torch.bincount(all_modified_segs.reshape(-1).long())).float().pow(.35)
        class_weights /= class_weights.mean()

        ### Add train sampler and dataloaders ##
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        # val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
            sampler=train_subsampler, pin_memory=True, drop_last=False,
            # collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
        )

        training_dataset.set_augment_at_collate(False)

#         val_dataloader = DataLoader(training_dataset, batch_size=config.val_batch_size,
#                                     sampler=val_subsampler, pin_memory=True, drop_last=False)

        ### Get model, data parameters, optimizers for model and data parameters, as well as grad scaler ###
        epx_start = config.get('checkpoint_epx', 0)

        if config.checkpoint_name:
            # Load from checkpoint
            _path = f"{config.mdl_save_prefix}/{config.checkpoint_name}_fold{fold_idx}_epx{epx_start}"
        else:
            _path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx_start}"

        (lraspp, optimizer, optimizer_dp, embedding, scaler) = get_model(config, len(training_dataset), len(training_dataset.label_tags), _path=_path, device='cuda')

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2)

        if optimizer_dp:
            scheduler_dp = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer_dp, T_0=500, T_mult=2)
        else:
            scheduler_dp = None

        t0 = time.time()

        # Prepare corr coefficient scoring
        training_dataset.eval(use_modified=True)
        if str(config.data_param_mode) == str(DataParamMode.GRIDDED_INSTANCE_PARAMS):
            norm_label, mod_label = list(zip(*[(sample['label'], sample['modified_label']) \
                for sample in training_dataset]))

            union_norm_mod_label = torch.logical_or(torch.stack(norm_label), torch.stack(mod_label))
            union_norm_mod_label = union_norm_mod_label.cuda()

        for epx in range(epx_start, config.epochs):
            global_idx = get_global_idx(fold_idx, epx, config.epochs)

            lraspp.train()

            ### Disturb samples ###
            training_dataset.train(use_modified=(epx >= config.start_disturbing_after_ep))
            wandb.log({"use_modified": float(training_dataset.use_modified)}, step=global_idx)

            epx_losses = []
            dices = []
            class_dices = []

            # Load data
            for batch in train_dataloader:

                optimizer.zero_grad()
                if optimizer_dp:
                    optimizer_dp.zero_grad()

                b_img = batch['image']
                b_seg = batch['label']
                b_spat_aug_grid = batch['spat_augment_grid']

                b_seg_modified = batch['modified_label']
                b_idxs_dataset = batch['dataset_idx']
                b_img = b_img.float()

                b_img = b_img.cuda()
                b_seg_modified = b_seg_modified.cuda()
                b_idxs_dataset = b_idxs_dataset.cuda()
                b_seg = b_seg.cuda()
                class_weights = class_weights.cuda()
                b_spat_aug_grid = b_spat_aug_grid.cuda()

                if config.use_mind:
                    b_img = mindssc(b_img.unsqueeze(1).unsqueeze(1)).squeeze(2)
                else:
                    b_img = b_img.unsqueeze(1)
                ### Forward pass ###
                with amp.autocast(enabled=True):
                    assert b_img.dim() == 4, \
                        f"Input image for model must be 4D: BxCxHxW but is {b_img.shape}"

                    logits = lraspp(b_img)['out']

                    ### Calculate loss ###
                    assert logits.dim() == 4, \
                        f"Input shape for loss must be BxNUM_CLASSESxHxW but is {logits.shape}"
                    assert b_seg_modified.dim() == 3, \
                        f"Target shape for loss must be BxHxW but is {b_seg_modified.shape}"

                    if config.data_param_mode == str(DataParamMode.INSTANCE_PARAMS):

                        loss = nn.CrossEntropyLoss(reduction='none')(logits, b_seg_modified).mean((-1,-2))
                        bare_weight = embedding(b_idxs_dataset).squeeze()
                        weight = torch.sigmoid(bare_weight)
                        weight = weight/weight.mean()

                        # Prepare logits for scoring
                        logits_for_score = logits.argmax(1)

                        if config.use_risk_regularization:
                            p_pred_num = (logits_for_score > 0).sum(dim=(-2,-1)).detach()
                            risk_regularization = -weight*p_pred_num/(logits_for_score.shape[-2]*logits_for_score.shape[-1])
                            loss = (loss*weight).sum() + risk_regularization.sum()
                        else:
                            loss = (loss*weight).sum()

                        gt_num = (b_seg_modified > 0).sum(dim=(-2,-1))
                        metric = 1/(np.log(gt_num+np.exp(1))+np.exp(1))
                        corr_coeff = np.corrcoef((bare_weight/metric).cpu().detach(), dice.detach())[0,1]

                        print("dice vs. e_log_gt:", corr_coeff)

                    elif config.data_param_mode == str(DataParamMode.GRIDDED_INSTANCE_PARAMS):
                        loss = nn.CrossEntropyLoss(reduction='none')(logits, b_seg_modified)
                        m_dp_idxs = map_embedding_idxs(b_idxs_dataset, config.grid_size_y, config.grid_size_x)
                        weight = embedding(m_dp_idxs)
                        weight = weight.reshape(-1, config.grid_size_y, config.grid_size_x)
                        weight = weight.unsqueeze(1)
                        weight = torch.nn.functional.interpolate(
                            weight,
                            size=(b_seg_modified.shape[-2:]),
                            mode='bilinear',
                            align_corners=True
                        )
                        weight = torch.sigmoid(weight)
                        weight = weight/weight.mean()
                        weight = F.grid_sample(weight, b_spat_aug_grid,
                            padding_mode='border', align_corners=False)
                        loss = (loss.unsqueeze(1)*weight).sum()

                        # Prepare logits for scoring
                        logits_for_score = (logits*weight).argmax(1)

                    else:
                        loss = nn.CrossEntropyLoss(class_weights)(logits, b_seg_modified)
                        # Prepare logits for scoring
                        logits_for_score = logits.argmax(1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)

                if str(config.data_param_mode) != str(DataParamMode.DISABLED) and epx > 10:
                    scaler.step(optimizer_dp)

                scaler.update()

                epx_losses.append(loss.item())

                # Calculate dice score
                b_dice = dice2d(
                    torch.nn.functional.one_hot(logits_for_score, len(training_dataset.label_tags)),
                    torch.nn.functional.one_hot(b_seg, len(training_dataset.label_tags)), # Calculate dice score with original segmentation (no disturbance)
                    one_hot_torch_style=True
                )

                dices.append(get_batch_dice_over_all(
                    b_dice, exclude_bg=True))
                class_dices.append(get_batch_dice_per_class(
                    b_dice, training_dataset.label_tags, exclude_bg=True))

                ###  Scheduler management ###
                if config.use_cosine_annealing:
                    scheduler.step()
                    # if scheduler_dp:
                    #     scheduler_dp.step()
                    # if epx == config.epochs//2:
                    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    #         optimizer, T_0=500, T_mult=2)
                    #     if optimizer_dp:
                    #         scheduler_dp = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    #             optimizer_dp, T_0=500, T_mult=2)
                if config.debug:
                    break

            ### Logging ###
            print(f"### Log epoch {epx} @ {time.time()-t0:.2f}s")
            print("### Training")
            ### Log wandb data ###
            # Log the epoch idx per fold - so we can recover the diagram by setting
            # ref_epoch_idx as x-axis in wandb interface
            wandb.log({"ref_epoch_idx": epx}, step=global_idx)

            mean_loss = torch.tensor(epx_losses).mean()
            wandb.log({f'losses/loss_fold{fold_idx}': mean_loss}, step=global_idx)

            mean_dice = np.nanmean(dices)
            print(f'dice_mean_wo_bg_fold{fold_idx}', f"{mean_dice*100:.2f}%")
            wandb.log({f'scores/dice_mean_wo_bg_fold{fold_idx}': mean_dice}, step=global_idx)

            log_class_dices("scores/dice_mean_", f"_fold{fold_idx}", class_dices, global_idx)

            # Log data parameters of disturbed samples
            if len(training_dataset.disturbed_idxs) > 0 and str(config.data_param_mode) != str(DataParamMode.DISABLED):
                if str(config.data_param_mode) == str(DataParamMode.GRIDDED_INSTANCE_PARAMS):
                    m_tr_idxs = map_embedding_idxs(train_idxs,
                        config.grid_size_y, config.grid_size_x
                    ).cuda()
                    masks = union_norm_mod_label[train_idxs].float()
                    masked_weights = torch.nn.functional.interpolate(
                        embedding(m_tr_idxs).view(-1,1,config.grid_size_y, config.grid_size_x),
                        size=(masks.shape[-2:]),
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1) * masks
                    masked_weights[masked_weights==0.] = float('nan')

                    corr_coeff = np.corrcoef(
                        np.nanmean(masked_weights.detach().cpu(), axis=(-2,-1)),
                        disturbed_bool_vect[train_idxs].cpu().numpy()
                    )[0,1]

                elif str(config.data_param_mode) == str(DataParamMode.INSTANCE_PARAMS):
                    corr_coeff = np.corrcoef(
                        embedding(train_idxs.cuda()).detach().cpu().view(train_idxs.numel()).numpy(),
                        disturbed_bool_vect[train_idxs].cpu().numpy()
                    )[0,1]

                wandb.log(
                    {f'data_parameters/corr_coeff_fold{fold_idx}': corr_coeff},
                    step=global_idx
                )
                print(f'data_parameters/corr_coeff_fold{fold_idx}', f"{corr_coeff:.2f}")

            if str(config.data_param_mode) != str(DataParamMode.DISABLED):
                log_data_parameter_stats(f'data_parameters/iter_stats_fold{fold_idx}', global_idx, embedding.weight.data)

            if (epx % config.save_every == 0 and epx != 0) \
                or (epx+1 == config.epochs):
                _path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx}"
                save_model(
                    _path,
                    lraspp=lraspp,
                    optimizer=optimizer, optimizer_dp=optimizer_dp,
                    scheduler=scheduler, scheduler_dp=scheduler_dp,
                    embedding=embedding,
                    scaler=scaler)
                (lraspp, optimizer, optimizer_dp, embedding, scaler) = get_model(config, len(training_dataset), len(training_dataset.label_tags), _path=_path, device='cuda')

            print()
            print("### Validation")
            lraspp.eval()
            training_dataset.eval()

            val_dices = []
            val_class_dices = []

            with amp.autocast(enabled=True):
                with torch.no_grad():
                    for val_idx in val_3d_idxs:
                        val_sample = training_dataset.get_3d_item(val_idx)
                        stack_dim = training_dataset.yield_2d_normal_to
                        # Create batch out of single val sample
                        b_val_img = val_sample['image'].unsqueeze(0)
                        b_val_seg = val_sample['label'].unsqueeze(0)

                        B = b_val_img.shape[0]

                        b_val_img = b_val_img.unsqueeze(1).float().cuda()
                        b_val_seg = b_val_seg.cuda()
                        b_val_img_2d = make_2d_stack_from_3d(b_val_img, stack_dim=stack_dim)

                        if config.use_mind:
                            b_val_img_2d = mindssc(b_val_img_2d.unsqueeze(1)).squeeze(2)

                        output_val = lraspp(b_val_img_2d)['out']

                        # Prepare logits for scoring
                        # Scoring happens in 3D again - unstack batch tensor again to stack of 3D
                        val_logits_for_score = output_val.argmax(1)
                        val_logits_for_score_3d = make_3d_from_2d_stack(
                            val_logits_for_score.unsqueeze(1), stack_dim, B
                        ).squeeze(1)

                        b_val_dice = dice3d(
                            torch.nn.functional.one_hot(val_logits_for_score_3d, len(training_dataset.label_tags)),
                            torch.nn.functional.one_hot(b_val_seg, len(training_dataset.label_tags)),
                            one_hot_torch_style=True
                        )

                        # Get mean score over batch
                        val_dices.append(get_batch_dice_over_all(
                            b_val_dice, exclude_bg=True))

                        val_class_dices.append(get_batch_dice_per_class(
                            b_val_dice, training_dataset.label_tags, exclude_bg=True))

                        if config.do_plot:
                            print(f"Validation 3D image label/ground-truth {val_3d_idxs}")
                            print(get_batch_dice_over_all(
                            b_val_dice, exclude_bg=False))
                            # display_all_seg_slices(b_seg.unsqueeze(1), logits_for_score)
                            display_seg(in_type="single_3D",
                                reduce_dim="W",
                                img=val_sample['image'].unsqueeze(0).cpu(),
                                seg=val_logits_for_score_3d.squeeze(0).cpu(), # CHECK TODO
                                ground_truth=b_val_seg.squeeze(0).cpu(),
                                crop_to_non_zero_seg=True,
                                crop_to_non_zero_gt=True,
                                alpha_seg=.3,
                                alpha_gt=.0
                            )

                    mean_val_dice = np.nanmean(val_dices)
                    print(f'val_dice_mean_wo_bg_fold{fold_idx}', f"{mean_val_dice*100:.2f}%")
                    wandb.log({f'scores/val_dice_mean_wo_bg_fold{fold_idx}': mean_val_dice}, step=global_idx)
                    log_class_dices("scores/val_dice_mean_", f"_fold{fold_idx}", val_class_dices, global_idx)

            print()
            # End of training loop

            if config.debug:
                break

        if str(config.data_param_mode) != str(DataParamMode.DISABLED):
            # Write sample data

            training_dataset.eval(use_modified=True)
            all_idxs = torch.tensor(range(len(training_dataset))).cuda()
            train_label_snapshot_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}/train_label_snapshot.pth")
            seg_viz_out_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}/data_parameter_weighted_samples.png")

            train_label_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            if str(config.data_param_mode) == str(DataParamMode.INSTANCE_PARAMS):
                dp_weights = embedding(all_idxs)
                save_data = []
                data_generator = zip(
                    dp_weights[train_idxs], \
                    disturbed_bool_vect[train_idxs],
                    torch.utils.data.Subset(training_dataset, train_idxs)
                )

                for dp_weight, disturb_flg, sample in data_generator:
                    data_tuple = ( \
                        dp_weight,
                        bool(disturb_flg.item()),
                        sample['id'],
                        sample['dataset_idx'],
                        sample['image'],
                        sample['label'],
                        sample['modified_label'],
                        inference_wrap(lraspp, sample['image'].cuda(), use_mind=config.use_mind)
                    )
                    save_data.append(data_tuple)

                save_data = sorted(save_data, key=lambda tpl: tpl[0])
                (dp_weight, disturb_flags,
                 d_ids, dataset_idxs, _2d_imgs,
                 _2d_labels, _2d_modified_labels, _2d_predictions) = zip(*save_data)

                dp_weight = torch.stack(dp_weight)
                dataset_idxs = torch.stack(dataset_idxs)
                _2d_imgs = torch.stack(_2d_imgs)
                _2d_labels = torch.stack(_2d_labels)
                _2d_modified_labels = torch.stack(_2d_modified_labels)
                _2d_predictions = torch.stack(_2d_predictions)

                torch.save(
                    {
                        'data_parameters': dp_weight.cpu(),
                        'disturb_flags': disturb_flags,
                        'd_ids': d_ids,
                        'dataset_idxs': dataset_idxs.cpu(),
                        'labels': _2d_labels.cpu().to_sparse(),
                        'modified_labels': _2d_modified_labels.cpu().to_sparse(),
                        'train_predictions': _2d_predictions.cpu().to_sparse(),
                    },
                    train_label_snapshot_path
                )

            elif str(config.data_param_mode) == str(DataParamMode.GRIDDED_INSTANCE_PARAMS):
                # Log histogram of clean and disturbed parameters
                m_all_idxs = map_embedding_idxs(all_idxs,
                        config.grid_size_y, config.grid_size_x
                ).cuda()
                masks = union_norm_mod_label[all_idxs].float()
                all_weights = torch.nn.functional.interpolate(
                    embedding(m_all_idxs).view(-1,1,config.grid_size_y, config.grid_size_x),
                    size=(masks.shape[-2:]),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

                masked_weights = all_weights * masks
                masked_weights[masked_weights==0.] = float('nan')
                dp_weights = np.nanmean(masked_weights.detach().cpu(), axis=(-2,-1))

                weightmap_out_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}/data_parameter_weightmap.png")

                save_data = []
                data_generator = zip(
                    dp_weights[train_idxs],
                    all_weights[train_idxs],
                    disturbed_bool_vect[train_idxs],
                    torch.utils.data.Subset(training_dataset,train_idxs)
                )

                for dp_weight, weightmap, disturb_flg, sample in data_generator:
                    data_tuple = (
                        dp_weight,
                        weightmap,
                        bool(disturb_flg.item()),
                        sample['id'],
                        sample['dataset_idx'],
                        sample['image'],
                        sample['label'],
                        sample['modified_label']
                    )
                    save_data.append(data_tuple)

                save_data = sorted(save_data, key=lambda tpl: tpl[0])

                (dp_weight, dp_weightmap, disturb_flags,
                 d_ids, dataset_idxs, _2d_imgs,
                 _2d_labels, _2d_modified_labels) = zip(*save_data)

                dp_weight = torch.stack(dp_weight)
                dp_weightmap = torch.stack(dp_weightmap)
                dataset_idxs = torch.stack(dataset_idxs)
                _2d_imgs = torch.stack(_2d_imgs)
                _2d_labels = torch.stack(_2d_labels)
                _2d_modified_labels = torch.stack(_2d_modified_labels)
                _2d_predictions = torch.stack(_2d_predictions)

                torch.save(
                    {
                        'data_parameters': dp_weight.cpu(),
                        'data_parameter_weightmaps': dp_weightmap.cpu(),
                        'disturb_flags': disturb_flags,
                        'd_ids': d_ids,
                        'dataset_idxs': dataset_idxs.cpu(),
                        'labels': _2d_labels.cpu().to_sparse(),
                        'modified_labels': _2d_modified_labels.cpu().to_sparse(),
                        'train_predictions': _2d_predictions.cpu().to_sparse(),
                    },
                    train_label_snapshot_path
                )

                print("Writing weight map image.")
                weightmap_out_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}_data_parameter_weightmap.png")
                visualize_seg(in_type="batch_2D",
                    img=torch.stack(dp_weightmap).unsqueeze(1),
                    seg=torch.stack(_2d_modified_labels),
                    alpha_seg = 0.,
                    n_per_row=70,
                    overlay_text=overlay_text_list,
                    annotate_color=(0,255,255),
                    frame_elements=disturb_flags,
                    file_path=weightmap_out_path,
                )

            if len(training_dataset.disturbed_idxs) > 0:
                # Log histogram
                separated_params = list(zip(dp_weights[clean_idxs], dp_weights[training_dataset.disturbed_idxs]))
                s_table = wandb.Table(columns=['clean_idxs', 'disturbed_idxs'], data=separated_params)
                fields = {"primary_bins" : "clean_idxs", "secondary_bins" : "disturbed_idxs", "title" : "Data parameter composite histogram"}
                composite_histogram = wandb.plot_table(vega_spec_name="rap1ide/composite_histogram", data_table=s_table, fields=fields)
                wandb.log({f"data_parameters/separated_params_fold_{fold_idx}": composite_histogram})

            # Write out data of modified and un-modified labels and an overview image
            print("Writing train sample image.")
            # overlay text example: d_idx=0, dp_i=1.00, dist? False
            overlay_text_list = [f"id:{d_id} dp:{instance_p.item():.2f}" \
                for d_id, instance_p, disturb_flg in zip(d_ids, dp_weight, disturb_flags)]

            visualize_seg(in_type="batch_2D",
                img=interpolate_sample(b_label=_2d_labels, scale_factor=.5, yield_2d=True)[1].unsqueeze(1),
                seg=interpolate_sample(b_label=4*_2d_predictions.squeeze(1), scale_factor=.5, yield_2d=True)[1],
                ground_truth=interpolate_sample(b_label=_2d_modified_labels, scale_factor=.5, yield_2d=True)[1],
                crop_to_non_zero_seg=False,
                alpha_seg = .5,
                alpha_gt = .5,
                n_per_row=70,
                overlay_text=overlay_text_list,
                annotate_color=(0,255,255),
                frame_elements=disturb_flags,
                file_path=seg_viz_out_path,
            )
        # End of fold loop


# %%
# Config overrides
# config_dict['wandb_mode'] = 'disabled'
# config_dict['debug'] = True
# Model loading
# config_dict['checkpoint_name'] = 'treasured-water-717'
# # config_dict['fold_override'] = 0
# config_dict['checkpoint_epx'] = 39

# Define sweep override dict
sweep_config_dict = dict(
    method='grid',
    metric=dict(goal='maximize', name='scores/val_dice_mean_tumour_fold0'),
    parameters=dict(
        # disturbance_mode=dict(
        #     values=[
        #        'LabelDisturbanceMode.AFFINE',
        #     ]
        # ),
        #     values=[
        #        'LabelDisturbanceMode.AFFINE',
        #     ]
        # ),
        # reg_state=dict(
        #     values=['best','combined']
        # ),
        # disturbance_strength=dict(
        #     values=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        # ),
        # disturbed_percentage=dict(
        #     values=[0.3, 0.6]
        # ),
        data_param_mode=dict(
            values=[
                DataParamMode.INSTANCE_PARAMS,
                DataParamMode.DISABLED,
            ]
        ),
        # use_risk_regularization=dict(
        #     values=[False, True]
        # )
    )
)

# %%
def normal_run():
    with wandb.init(project="curriculum_deeplab", group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:

        run_name = run.name
        config = wandb.config
        training_dataset = prepare_data(config)
        train_DL(run_name, config, training_dataset)

def sweep_run():
    with wandb.init() as run:
        run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        )

        run_name = run.name
        config = wandb.config
        training_dataset = prepare_data(config)
        train_DL(run_name, config, training_dataset)


if config_dict['do_sweep']:
    # Integrate all config_dict entries into sweep_dict.parameters -> sweep overrides config_dict
    cp_config_dict = copy.deepcopy(config_dict)
    # cp_config_dict.update(copy.deepcopy(sweep_config_dict['parameters']))
    for del_key in sweep_config_dict['parameters'].keys():
        if del_key in cp_config_dict:
            del cp_config_dict[del_key]
    merged_sweep_config_dict = copy.deepcopy(sweep_config_dict)
    # merged_sweep_config_dict.update(cp_config_dict)
    for key, value in cp_config_dict.items():
        merged_sweep_config_dict['parameters'][key] = dict(value=value)
    # Convert enum values in parameters to string. They will be identified by their numerical index otherwise
    for key, param_dict in merged_sweep_config_dict['parameters'].items():
        if 'value' in param_dict and isinstance(param_dict['value'], Enum):
            param_dict['value'] = str(param_dict['value'])
        if 'values' in param_dict:
            param_dict['values'] = [str(elem) if isinstance(elem, Enum) else elem for elem in param_dict['values']]

        merged_sweep_config_dict['parameters'][key] = param_dict

    sweep_id = wandb.sweep(merged_sweep_config_dict, project="curriculum_deeplab")
    wandb.agent(sweep_id, function=sweep_run)

else:
    normal_run()


# %%
# def inference_DL(run_name, config, inf_dataset):

#     score_dicts = []

#     fold_iter = range(config.num_folds)
#     if config_dict['only_first_fold']:
#         fold_iter = fold_iter[0:1]

#     for fold_idx in fold_iter:
#         lraspp, *_ = load_model(f"{config.mdl_save_prefix}_fold{fold_idx}", config, len(validation_dataset))

#         lraspp.eval()
#         inf_dataset.eval()
#         stack_dim = config.yield_2d_normal_to

#         inf_dices = []
#         inf_dices_tumour = []
#         inf_dices_cochlea = []

#         for inf_sample in inf_dataset:
#             global_idx = get_global_idx(fold_idx, sample_idx, config.epochs)
#             crossmoda_id = sample['crossmoda_id']
#             with amp.autocast(enabled=True):
#                 with torch.no_grad():

#                     # Create batch out of single val sample
#                     b_inf_img = inf_sample['image'].unsqueeze(0)
#                     b_inf_seg = inf_sample['label'].unsqueeze(0)

#                     B = b_inf_img.shape[0]

#                     b_inf_img = b_inf_img.unsqueeze(1).float().cuda()
#                     b_inf_seg = b_inf_seg.cuda()
#                     b_inf_img_2d = make_2d_stack_from_3d(b_inf_img, stack_dim=stack_dim)

#                     if config.use_mind:
#                         b_inf_img_2d = mindssc(b_inf_img_2d.unsqueeze(1)).squeeze(2)

#                     output_inf = lraspp(b_inf_img_2d)['out']

#                     # Prepare logits for scoring
#                     # Scoring happens in 3D again - unstack batch tensor again to stack of 3D
#                     inf_logits_for_score = make_3d_from_2d_stack(output_inf, stack_dim, B)
#                     inf_logits_for_score = inf_logits_for_score.argmax(1)

#                     inf_dice = dice3d(
#                         torch.nn.functional.one_hot(inf_logits_for_score, 3),
#                         torch.nn.functional.one_hot(b_inf_seg, 3),
#                         one_hot_torch_style=True
#                     )
#                     inf_dices.append(get_batch_dice_wo_bg(inf_dice))
#                     inf_dices_tumour.append(get_batch_dice_tumour(inf_dice))
#                     inf_dices_cochlea.append(get_batch_dice_cochlea(inf_dice))

#                     if config.do_plot:
#                         print("Inference 3D image label/ground-truth")
#                         print(inf_dice)
#                         # display_all_seg_slices(b_seg.unsqueeze(1), logits_for_score)
#                         display_seg(in_type="single_3D",
#                             reduce_dim="W",
#                             img=inf_sample['image'].unsqueeze(0).cpu(),
#                             seg=inf_logits_for_score.squeeze(0).cpu(),
#                             ground_truth=b_inf_seg.squeeze(0).cpu(),
#                             crop_to_non_zero_seg=True,
#                             crop_to_non_zero_gt=True,
#                             alpha_seg=.4,
#                             alpha_gt=.2
#                         )

#             if config.debug:
#                 break

#         mean_inf_dice = np.nanmean(inf_dices)
#         mean_inf_dice_tumour = np.nanmean(inf_dices_tumour)
#         mean_inf_dice_cochlea = np.nanmean(inf_dices_cochlea)

#         print(f'inf_dice_mean_wo_bg_fold{fold_idx}', f"{mean_inf_dice*100:.2f}%")
#         print(f'inf_dice_mean_tumour_fold{fold_idx}', f"{mean_inf_dice_tumour*100:.2f}%")
#         print(f'inf_dice_mean_cochlea_fold{fold_idx}', f"{mean_inf_dice_cochlea*100:.2f}%")
#         wandb.log({f'scores/inf_dice_mean_wo_bg_fold{fold_idx}': mean_inf_dice}, step=global_idx)
#         wandb.log({f'scores/inf_dice_mean_tumour_fold{fold_idx}': mean_inf_dice_tumour}, step=global_idx)
#         wandb.log({f'scores/inf_dice_mean_cochlea_fold{fold_idx}': mean_inf_dice_cochlea}, step=global_idx)

#         # Store data for inter-fold scoring
#         class_dice_list = inf_dices.tolist()[0]
#         for class_idx, class_dice in enumerate(class_dice_list):
#             score_dicts.append(
#                 {
#                     'fold_idx': fold_idx,
#                     'crossmoda_id': crossmoda_id,
#                     'class_idx': class_idx,
#                     'class_dice': class_dice,
#                 }
#             )

#     mean_oa_inf_dice = np.nanmean(torch.tensor([score['class_dice'] for score in score_dicts]))
#     print(f"Mean dice over all folds, classes and samples: {mean_oa_inf_dice*100:.2f}%")
#     wandb.log({'scores/mean_dice_all_folds_samples_classes': mean_oa_inf_dice}, step=global_idx)

#     return score_dicts


# %%
# folds_scores = []
# run = wandb.init(project="curriculum_deeplab", name=run_name, group=f"testing", job_type="test",
#         config=config_dict, settings=wandb.Settings(start_method="thread"),
#         mode=config_dict['wandb_mode']
# )
# config = wandb.config
# score_dicts = inference_DL(run_name, config, validation_dataset)
# folds_scores.append(score_dicts)
# wandb.finish()

# %%
