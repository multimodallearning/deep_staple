import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from enum import Enum, auto
class LabelDisturbanceMode(Enum):
    FLIP_ROLL = auto()
    AFFINE = auto()

@contextmanager
def torch_manual_seeded(seed):
    saved_state = torch.get_rng_state()
    yield
    torch.set_rng_state(saved_state)

def ensure_dense(label):
    entered_sparse = label.is_sparse
    if entered_sparse:
        label = label.to_dense()

    return label, entered_sparse

def restore_sparsity(label, was_sparse):
    if was_sparse and not label.is_sparse:
        return label.to_sparse()
    return label

def dilate_label_class(b_label, class_max_idx, class_dilate_idx,
                       use_2d, kernel_sz=3):

    if kernel_sz < 2:
        return b_label

    b_dilated_label = b_label

    b_onehot = torch.nn.functional.one_hot(b_label.long(), class_max_idx+1)
    class_slice = b_onehot[...,class_dilate_idx]

    if use_2d:
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



# %%
def in_notebook():
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return False

def get_script_dir():
    if in_notebook:
        return os.path.abspath('')
    else:
        return os.path.dirname(os.path.realpath(__file__))


# %%
def interpolate_sample(b_image=None, b_label=None, scale_factor=1.,
                       use_2d=False):
    if use_2d:
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



def augmentNoise(b_image, strength=0.05):
    return b_image + strength*torch.randn_like(b_image)



def spatial_augment(b_image=None, b_label=None,
    bspline_num_ctl_points=6, bspline_strength=0.005, bspline_probability=.9,
    affine_strength=0.08, add_affine_translation=0., affine_probability=.45,
    pre_interpolation_factor=None,
    use_2d=False,
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
        b_image, b_label = interpolate_sample(b_image, b_label, pre_interpolation_factor, use_2d)

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
        if use_2d:
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
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool2d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
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
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2)),
                    torch.nn.AvgPool3d(KERNEL_SIZE,stride=1,padding=int(KERNEL_SIZE//2))
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
