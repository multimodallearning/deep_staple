import sys
import argparse

from tqdm import tqdm
from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
import re
import curriculum_deeplab.utils.nifti_sets as nsets

REF_SPACING = 0.5*torch.ones(3,)
REF_SHAPE = torch.tensor([420,420,360])

TUMOUR_BBOX_LEFT = torch.tensor(
    [[186, 165, 7],
     [314, 293, 199]])

TUMOUR_BBOX_RIGHT = torch.tensor(
    [[100, 169, 11],
     [228, 297, 203]])

BBOX_REF_SHAPE = np.array([128,128,128])

# def kpts_world(kpts_pt, shape, align_corners=None):
#     device = kpts_pt.device
#     D, H, W = shape

#     if not align_corners:
#         kpts_pt /= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
#     kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1)).flip(-1)

#     return kpts_world_

def get_nifti(img, spacing):
    affine = np.eye(4)
    affine[0,0] = -spacing[0]
    affine[1,1] = -spacing[1]
    affine[2,2] = spacing[2]
    return nib.Nifti1Image(img, affine)

def adjust_spacing_and_get_shapes(ni_data, ref_spacing, device='cpu'):

    spacing = ni_data.header.get_zooms()

    shape = torch.tensor(ni_data.shape)
    spacing = torch.tensor(ni_data.header.get_zooms())
    scale_factor = spacing / ref_spacing
    new_shape = (shape * scale_factor).round().long()
    new_scale_factor = new_shape / shape
    new_spacing = spacing / new_scale_factor

    fdata = torch.from_numpy(ni_data.get_fdata()).unsqueeze(0).unsqueeze(0).float().to(device)

    return get_nifti(fdata[0,0].cpu().numpy(), spacing), new_spacing, new_shape


def interpolate_and_pad(ni_data, ref_spacing, ref_shape, is_label, device='cpu'):

    shape = torch.tensor(ni_data.shape)
    spacing = torch.tensor(ni_data.header.get_zooms())
    scale_factor = spacing / ref_spacing
    new_shape = (shape * scale_factor).round().long()
    new_scale_factor = new_shape / shape
    new_spacing = spacing / new_scale_factor

    fdata = torch.from_numpy(ni_data.get_fdata()).unsqueeze(0).unsqueeze(0).float().to(device)

    if not is_label:
        fdata = F.interpolate(fdata, new_shape.tolist(), mode='trilinear', align_corners=True)
    else:
        fdata = (F.interpolate(F.one_hot(fdata.long())[:, 0, :, :, :, :].permute(0, 4, 1, 2, 3).float(), new_shape.tolist(), mode='trilinear', align_corners=True) > 0.5).max(1)[1].unsqueeze(1).float()

    shape = torch.tensor(fdata.shape[2:])
    spacing = new_spacing

    if shape[0] < ref_shape[0]:
        pad = ref_shape[0] - shape[0]
        pad1 = (pad / 2).floor().int().item()
        pad2 = pad - pad1
        if not is_label:
            fdata = F.pad(fdata, [0, 0, 0, 0, pad1, pad2], mode='constant', value=fdata.min())
        else:
            label = F.pad(fdata, [0, 0, 0, 0, pad1, pad2], mode='constant', value=0)

    if shape[1] < ref_shape[1]:
        pad = ref_shape[1] - shape[1]
        pad1 = (pad / 2).floor().int().item()
        pad2 = pad - pad1
        if not is_label:
            fdata = F.pad(fdata, [0, 0, pad1, pad2, 0, 0], mode='constant', value=fdata.min())
        else:
            fdata = F.pad(fdata, [0, 0, pad1, pad2, 0, 0], mode='constant', value=0)

    if shape[2] < ref_shape[2]:
        pad = ref_shape[2] - shape[2]
        pad1 = (pad / 2).floor().int().item()
        pad2 = pad - pad1
        if not is_label:
            fdata = F.pad(fdata, [pad1, pad2, 0, 0, 0, 0], mode='constant', value=fdata.min())
        else:
            fdata = F.pad(fdata, [pad1, pad2, 0, 0, 0, 0], mode='constant', value=0)

    shape = torch.tensor(fdata.shape[2:])
    new_spacing = spacing / (ref_shape / shape)

    return get_nifti(fdata[0,0].cpu().numpy(), new_spacing)

def split_lr_sides_fixed(ni_data, is_target_domain, device='cpu'):

    spacing = ni_data.header.get_zooms()

    fdata = ni_data.get_fdata()

    if is_target_domain:
        fdata_temp = np.zeros(fdata.shape)
        fdata_temp[:, :, :-40] = fdata[:, :, 40:]
        fdata = fdata_temp


    fdata_tumour_crop_l = fdata[TUMOUR_BBOX_LEFT[0,0]:TUMOUR_BBOX_LEFT[1,0],TUMOUR_BBOX_LEFT[0,1]:TUMOUR_BBOX_LEFT[1,1],TUMOUR_BBOX_LEFT[0,2]:TUMOUR_BBOX_LEFT[1,2]]
    fdata_tumour_crop_r = fdata[TUMOUR_BBOX_RIGHT[0,0]:TUMOUR_BBOX_RIGHT[1,0],TUMOUR_BBOX_RIGHT[0,1]:TUMOUR_BBOX_RIGHT[1,1],TUMOUR_BBOX_RIGHT[0,2]:TUMOUR_BBOX_RIGHT[1,2]]

    return get_nifti(fdata_tumour_crop_l, spacing), get_nifti(fdata_tumour_crop_r, spacing)


def apply_fine_crop(ni_image, ni_label, is_target_domain, lr_id, bbox_ref_shape, cochlea_centers, id_num):

    if id_num+lr_id in cochlea_centers:
        center = torch.from_numpy(cochlea_centers[id_num+lr_id])
        if lr_id == 'l':
            center += TUMOUR_BBOX_LEFT[0,:]
        else:
            center += TUMOUR_BBOX_RIGHT[0,:]

        if is_target_domain:
            center[2] += 40

        center = center.round()
        bbox = torch.stack((center - (bbox_ref_shape/2).floor(), center + (bbox_ref_shape/2).floor())).long()

        spacing = ni_image.header.get_zooms()

        img = ni_image.get_fdata()
        label = ni_label.get_fdata()

        img_tumour_crop = img[bbox[0,0]:bbox[1,0],bbox[0,1]:bbox[1,1],bbox[0,2]:bbox[1,2]]
        label_tumour_crop = label[bbox[0,0]:bbox[1,0],bbox[0,1]:bbox[1,1],bbox[0,2]:bbox[1,2]]

        return get_nifti(img_tumour_crop, spacing), get_nifti(label_tumour_crop, spacing)

    else:
        return None, None

def preprocess(base_dir, cochlea_centers_path, device='cpu'):

    subdirs = [
        '__omitted_labels_target_training__', '__omitted_labels_target_validation__',
        'source_training_labeled',
        'target_training_unlabeled', 'target_validation_unlabeled'
    ]

    print("Building L2 ...")
    for s_dir in subdirs:
        source_dir = Path(base_dir, s_dir)
        print(f"Processing {source_dir}")
        l1_nifti_paths = nsets.get_nifti_filepaths(
            source_dir,
            with_subdirs=True
        )
        # l1_nifti_paths = l1_nifti_paths[0:2] # DEBUG

        l2_nifti_paths = [_path.replace("L1_original", "L2_resampled_05mm") for _path in l1_nifti_paths]

        # Do not override L1 level (would reset world orientation for original data)
        # for _path in tqdm(nifti_paths):
        #     # Adjust spacing. Input is original data
        #     target_path = Path(_path.replace('L1_original', 'L1_original'))
        #     ni_data = nib.load(_path)
        #     ni_data = adjust_spacing(ni_data, REF_SPACING, device)


        for _path in tqdm(l1_nifti_paths):
            # Resample and pad. Input is original data
            target_path = Path(_path.replace('L1_original', 'L2_resampled_05mm'))
            is_label =  ("_Label" in _path)
            ni_data = nib.load(_path)
            ni_data = interpolate_and_pad(ni_data, REF_SPACING, REF_SHAPE, is_label, device=device)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(ni_data, target_path)

    print("Building L3 ...")
    for s_dir in subdirs:
        source_dir = Path(base_dir, s_dir)
        print(f"Processing {source_dir}")
        l1_nifti_paths = nsets.get_nifti_filepaths(
            source_dir,
            with_subdirs=True
        )
        # l1_nifti_paths = l1_nifti_paths[0:2] # DEBUG

        l2_nifti_paths = [_path.replace("L1_original", "L2_resampled_05mm") for _path in l1_nifti_paths]

        for _path in tqdm(l2_nifti_paths):
            # Save coarse crops. Input is resampled data at 0.5mm
            target_path = Path(_path.replace('L2_resampled_05mm', 'L3_coarse_fixed_crop'))
            is_label =  ("_Label" in _path)
            is_target_domain = ("hrT2" in _path)

            target_path_left = Path(str(target_path).replace('.nii.gz', '_l.nii.gz'))
            target_path_right = Path(str(target_path).replace('.nii.gz', '_r.nii.gz'))

            ni_data = nib.load(_path)
            ni_left, ni_right = split_lr_sides_fixed(ni_data, is_target_domain, device=device)
            target_path_left.parent.mkdir(parents=True, exist_ok=True)
            nib.save(ni_left, target_path_left)
            nib.save(ni_right, target_path_right)

    print("Building L4 ...")
    cochlea_centers = torch.load(cochlea_centers_path)

    for s_dir in subdirs:
        source_dir = Path(base_dir, s_dir)
        print(f"Processing {source_dir}")
        l1_nifti_paths = nsets.get_nifti_filepaths(
            source_dir,
            with_subdirs=True
        )
        # l1_nifti_paths = l1_nifti_paths[0:2] # DEBUG
        l2_nifti_paths = [_path.replace("L1_original", "L2_resampled_05mm") for _path in l1_nifti_paths]

        l2_label_paths = [_path for _path in l2_nifti_paths if "_Label" in _path]

        for label_path in tqdm(l2_label_paths):
            # Save fine crops. Input is coarse crop for cochlea, resampled data at 0.5mm and resampled data at 0.5mm for img and label
            image_path = label_path.replace('_Label', '')
            is_target_domain = ("hrT2" in label_path)

            if is_target_domain:
                image_path = image_path.replace('__omitted_labels_target_training__', 'target_training_unlabeled')
                image_path = image_path.replace('__omitted_labels_target_validation__', 'target_validation_unlabeled')

            ni_image = nib.load(image_path)
            ni_label = nib.load(label_path)
            target_path_image = Path(image_path.replace('L2_resampled_05mm', 'L4_fine_localized_crop'))
            target_path_label = Path(label_path.replace('L2_resampled_05mm', 'L4_fine_localized_crop'))
            id_num = re.match(r".*/crossmoda_([0-9]{1,3})_", label_path).group(1)

            target_path_image.parent.mkdir(parents=True, exist_ok=True)
            target_path_label.parent.mkdir(parents=True, exist_ok=True)

            for lr_id in ['l', 'r']:
                ni_image_cropped, ni_label_cropped = apply_fine_crop(ni_image, ni_label, is_target_domain, lr_id, BBOX_REF_SHAPE, cochlea_centers, id_num)
                target_path_image_side = Path(str(target_path_image).replace('.nii.gz', f'_{lr_id}.nii.gz'))
                target_path_label_side = Path(str(target_path_label).replace('.nii.gz', f'_{lr_id}.nii.gz'))
                if ni_image_cropped: nib.save(ni_image_cropped, target_path_image_side)
                if ni_label_cropped: nib.save(ni_label_cropped, target_path_label_side)


def main(argv):

    # L1_original
    # |- source_training_labeled/               # crossmoda_1_ceT1(_Label).nii.gz ... crossmoda_105_ceT1(_Label).nii.gz
    # |- target_training_unlabeled/             # crossmoda_106_hrT2.nii.gz ... crossmoda_210_hrT2.nii.gz
    # |- target_validation_unlabeled/           # crossmoda_211_hrT2.nii.gz ... crossmoda_242_hrT2.nii.gz
    # |- __omitted_labels_target_training__/    # crossmoda_106_hrT2_Label.nii.gz ... crossmoda_203_hrT2_Label.nii.gz
    # |- __omitted_labels_target_validation__/  # crossmoda_211_hrT2_Label.nii.gz ... crossmoda_242_hrT2_Label.nii.gz
    # |- __additional_data_source_domain__/     # crossmoda_106_ceT1(_Label).nii.gz ... crossmoda_242_ceT1(_Label).nii.gz and others
    # |- __additional_data_target_domain__/     # crossmoda_10_hrT2(_Label).nii.gz ... crossmoda_236_hrT2(_Label).nii.gz and others

    # L2_resampled_05mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/

    # L3_coarse_fixed_crop 128x128x192vox @ 0.5x0.5x0.5mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/

    # L4_fine_localized_crop 128x128x128vox @ 0.5x0.5x0.5mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", required=True)
    parser.add_argument("-c", "--cochlea-centers", required=True)
    parser.add_argument("-d", "--device", required=False, default="cpu")
    args = parser.parse_args(argv)

    base_dir = Path(args.input_folder)
    base_dir = base_dir.joinpath("L1_original")
    assert base_dir.is_dir(), f"Base directory '{base_dir}' does not exist."

    preprocess(base_dir, args.cochlea_centers, args.device)

if __name__ == "__main__":
    main(sys.argv[1:])
