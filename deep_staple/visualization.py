r"""This module contains useful plots for medical imaging"""
import functools
import warnings
from collections.abc import Iterable
import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.display import display
import PIL
from torchvision import transforms
import math
import torchvision.utils as vision_utils



def pil_images_from_img_tensor(s_tensor: torch.Tensor, scale_min_max=True) -> list[PIL.Image]:
    r"""
    Get a list of pil RGBA images from a 2D image stack tensor.
    Parameters:
            s_tensor (torch.Tensor): A 4D (S,C,H,W) tensor stack with
                C=1 grayscale channel or C=3 RGB channels
            scale_min_max (bool): The image values will be scaled
                to [0...255] (default: True)
    Returns:
            list(PIL.Image): List of RGBA pil images
    Example:
        pil_list = pil_images_from_img_tensor(torch.rand(2,1,60,80), scale_min_max=True)
    """

    assert s_tensor.dim() == 4, \
        f"Tensor must be stack of 2d with exposed channel dimension SxCxHxW but is {s_tensor.shape}"
    assert s_tensor.shape[1] in [1, 3] , \
        f"Tensor must have channel size C=1 for grayscale or C=3 for RGB img but has C={s_tensor.shape[1]}"

    s_tensor = s_tensor.detach().cpu()
    _min, _max = s_tensor.min(), s_tensor.max()

    if scale_min_max:
        if _max == _min:
            s_tensor = torch.zeros_like(s_tensor)
        else:
            s_tensor = s_tensor.sub(_min).div(_max-_min).mul(255)

    s_tensor = s_tensor.permute(0,2,3,1)
    if s_tensor.shape[-1] == 1:
        # Got a stack of grayscale images
        s_tensor = s_tensor.squeeze(-1)

    s_numpy = s_tensor.numpy()

    images = [PIL.Image.fromarray(numpy_rgb).convert('RGBA') for numpy_rgb in s_numpy]
    return images



def pil_images_from_onehot_seg(s_tensor_onehot, onehot_colormap, alpha) -> list[PIL.Image]:
    r"""
    Get a list of pil RGBA images from a 2D onehot-encoded label stack tensor.
    Parameters:
            s_tensor_onehot (torch.Tensor): A 4D (S,H,W,E) tensor stack with
                E=num_classes onehot-encoded channels
            onehot_colormap (dict): A color map dict containing class_id keys
                and tuple of RGB values. Non existent class_ids or class_id:None
                mappings will not be colored
            alpha (float): A value of [0.0...1.0] controlling the transparency of
                applied colors ranging from 0.0 (fully transparent) to 1.0 (fully visible)
    Returns:
            list(PIL.Image): List of RGBA pil images
    Example:
            onehot_colormap = {0: None, 1: (255,0,0), 2: (0,255,0)}
            seg_onehot_stack = torch.nn.functional.one_hot(
                (torch.rand(2,60,80) * 3).to(dtype=torch.int64), 3
            )
            pil_list = pil_images_from_onehot_seg(seg_onehot_stack, onehot_colormap, alpha=.8)
    """

    assert s_tensor_onehot.dim() == 4, \
        "Tensor must be stack of 2d with onehot encoding: Dim = [S, H, W, E]"
    # S,H,W,E = b_tensor_onehot.shape
    s_tensor_onehot = s_tensor_onehot.detach().cpu()
    alpha_channel = (int(255.*alpha),)
    # Create RBG tensor with shape S,H,W,RGBA
    s_rgba_tensor = torch.stack([torch.zeros(s_tensor_onehot.shape[:-1])]*4, dim=-1).type(torch.uint8)

    for onehot_id, rgb_val in onehot_colormap.items():
        if isinstance(rgb_val, tuple):
            bhw_idx = s_tensor_onehot.argmax(dim=-1) == onehot_id
            s_rgba_tensor[bhw_idx] = torch.tensor(rgb_val + alpha_channel, dtype=torch.uint8)

    b_rgb_numpy = s_rgba_tensor.numpy()

    # Append a list of S,C,H,W pil images
    list_images = [PIL.Image.fromarray(numpy_rgb).convert('RGBA') for numpy_rgb in b_rgb_numpy]
    return list_images



def get_stacked_overlays(s_2d_img_tensor=None,
                         s_2d_seg_tensor_onehot=None, s_2d_gt_tensor_onehot=None,
                         onehot_colormap_seg=None, onehot_colormap_gt=None,
                         alpha_seg=0.5, alpha_gt=0.5) \
    -> tuple[list[PIL.Image], torch.tensor]:

    r"""
    Get a stack of image and label overlays.
    Parameters:
            s_2d_img_tensor (torch.Tensor): Properties see docstring of
                pil_images_from_img_tensor()
            s_2d_seg_tensor_onehot (torch.Tensor): Properties see docstring of
                pil_images_from_onehot_seg()
            s_2d_gt_tensor_onehot (torch.Tensor): Properties see docstring of
                pil_images_from_onehot_seg()
            onehot_colormap_seg (dict): Properties see docstring of
                pil_images_from_onehot_seg()
            onehot_colormap_gt (dict): Properties see docstring of
                pil_images_from_onehot_seg()
            alpha_seg (float): Properties see docstring of
                pil_images_from_onehot_seg()
            alpha_gt (float): Properties see docstring of
                pil_images_from_onehot_seg()

    Returns:
            tuple(list(PIL.Image), torch.Tensor): Tuple of overlayed RGBA pil images
                and a stacked torch.Tensor. Tensor will have dimensions
                of S,RGBA,H,W and value range of [0...1].
                Both tuple members contain the same info but have
                different formatting
    Example:
            img_stack = torch.rand(2,1,60,80)
            onehot_colormap = {0: None, 1: (255,0,0), 2: (0,255,0)}
            seg_onehot_stack = torch.nn.functional.one_hot(
                (torch.rand(2,60,80) * 3).to(dtype=torch.int64), 3
            )
            pil_list, stack_tensor = \
                get_stacked_overlays(img_stack, seg_onehot_stack, onehot_colormap, alpha=.8)
    """

    pil_imgs, pil_segs, pil_gts = [], [], []

    if s_2d_img_tensor is not None:
        pil_imgs = pil_images_from_img_tensor(s_2d_img_tensor, scale_min_max=True)
    if s_2d_seg_tensor_onehot is not None:
        pil_segs = pil_images_from_onehot_seg(s_2d_seg_tensor_onehot, onehot_colormap=onehot_colormap_seg, alpha=alpha_seg)
    if s_2d_gt_tensor_onehot is not None:
        pil_gts = pil_images_from_onehot_seg(s_2d_gt_tensor_onehot, onehot_colormap=onehot_colormap_gt, alpha=alpha_gt)

    all_pil_overlays = []

    # Extract pil data which is not []. May be List of list of 1,2 or 3 elements (img, seg, gt)
    rgb_data_items_list = zip(*[data for data in [pil_imgs, pil_segs, pil_gts] if data])

    for rgb_data_items in rgb_data_items_list:
        pil_overlay = functools.reduce(lambda d_a, d_b: PIL.Image.alpha_composite(d_a, d_b), rgb_data_items)
        all_pil_overlays.append(pil_overlay)

    tensor_overlays = torch.stack([transforms.functional.to_tensor(ovl) for ovl in all_pil_overlays], dim=0)
    return all_pil_overlays, tensor_overlays



def get_overlay_grid(s_2d_img_tensor=None,
                     s_2d_seg_tensor_onehot=None, s_2d_gt_tensor_onehot=None,
                     onehot_colormap_seg=None, onehot_colormap_gt=None,
                     alpha_seg=0.5, alpha_gt=0.5,
                     n_per_row=4) \
    -> tuple[PIL.Image, torch.tensor]:

    r"""
    Get a stack of image and label overlays.
    Parameters:
            s_2d_img_tensor (torch.Tensor): Properties see docstring of
                pil_images_from_img_tensor()
            s_2d_seg_tensor_onehot (torch.Tensor): Properties see docstring of
                pil_images_from_onehot_seg()
            s_2d_gt_tensor_onehot (torch.Tensor): Properties see docstring of
                pil_images_from_onehot_seg()
            onehot_colormap_seg (dict): Properties see docstring of
                pil_images_from_onehot_seg()
            onehot_colormap_gt (dict): Properties see docstring of
                pil_images_from_onehot_seg()
            alpha_seg (float): Properties see docstring of
                pil_images_from_onehot_seg()
            alpha_gt (float): Properties see docstring of
                pil_images_from_onehot_seg()
            n_per_row (int): Number of overlays contained in an image row (default: 4)

    Returns:
            tuple(PIL.Image, torch.Tensor): Tuple of overlayed RGBA pil image grid
                and a torch.Tensor. Both tuple members contain the same info but have
                different formatting
    Example:
            img_stack = torch.rand(6,1,60,80)
            onehot_colormap = {0: None, 1: (255,0,0), 2: (0,255,0)}
            seg_onehot_stack = torch.nn.functional.one_hot(
                (torch.rand(6,60,80) * 3).to(dtype=torch.int64), 3
            )
            pil_image, stack_tensor = \
                get_overlay_grid(img_stack, seg_onehot_stack, onehot_colormap_seg=onehot_colormap, alpha=.8, n_per_row=3)
    """
    if s_2d_img_tensor is not None:
        assert s_2d_img_tensor.dim() == 4, f"Image shape must be Sx1xHxW but is {s_2d_img_tensor.shape}"
        # Make shape of HxW equal to onehot order SxHxWxC so spatial shape can be checked easier
        S, C, H, W = s_2d_img_tensor.shape
        img_shape_repr = s_2d_img_tensor.view(S, H, W, C)
    else:
        img_shape_repr = None

    if s_2d_seg_tensor_onehot is not None:
        assert s_2d_seg_tensor_onehot.dim() == 4, f"Segmentation shape must be SxHxWxONEHOT but is {s_2d_seg_tensor_onehot.shape}"
    if s_2d_gt_tensor_onehot is not None:
        assert s_2d_gt_tensor_onehot.dim() == 4, f"Ground-truth shape must be SxHxWxONEHOT but is {s_2d_gt_tensor_onehot.shape}"

    all_spatial_shapes = [data.shape[1:3] for data in [img_shape_repr, s_2d_seg_tensor_onehot, s_2d_gt_tensor_onehot] if data is not None]
    assert len(set(all_spatial_shapes)) == 1, f"HxW of image, segmentation and ground-truth must match but are {all_spatial_shapes}"

    _, tensor_overlays = get_stacked_overlays(s_2d_img_tensor,
                                              s_2d_seg_tensor_onehot, s_2d_gt_tensor_onehot,
                                              onehot_colormap_seg, onehot_colormap_gt,
                                              alpha_seg=alpha_seg, alpha_gt=alpha_gt)
    grid_tensor = vision_utils.make_grid(tensor_overlays, nrow=n_per_row)
    return transforms.functional.to_pil_image(grid_tensor), grid_tensor



def get_cmap_dict(class_max_id, pyplot_map_name='gist_rainbow', no_color_zero_id=True) -> dict:

    r"""
    Get a discretized color map dict based on matplotlib.pyplot color maps.
    Parameters:
            class_max_id (int): Max idx of labels. E.g. for labels with two categories
                and backgroud set this to 2
            pyplot_map_name (string):
                See https://matplotlib.org/stable/tutorials/colors/colormaps.html
                (default: 'gist_rainbow')
            no_color_zero_id (bool): Id value 0 gets None color. Common setting when 0
                is the background id (default: True)

    Returns:
            dict: e.g. {0: None, 1: (255,0,0), 2:(0,255,255)}
    Example:
            color_map = get_cmap_dict(4, pyplot_map_name='rainbow')
    """

    cmap = plt.get_cmap(pyplot_map_name)
    cmap_dict = {}

    if no_color_zero_id:
        cmap_dict[0] = None
        num_ids = class_max_id
        id_offset = 1
    else:
        num_ids = class_max_id+1
        id_offset = 0

    discretized_map = (cmap((numpy.arange(num_ids)/float(num_ids)))*255).astype(numpy.int32)
    for onehot_idx, rgb_list in enumerate(discretized_map):
        cmap_dict[onehot_idx+id_offset] = tuple(rgb_list)[:3] # Extract only RGB not alpha

    return cmap_dict



def make_2d_stack_from_3d(b_input, stack_dim):
    # Return 2d stack, old batch dimension and new stack/batch dimension
    assert b_input.dim() == 5, f"Input must be 5D: BxCxDxHxW but is {b_input.shape}"
    B, C, D, H, W = b_input.shape

    if stack_dim == "D":
        return b_input.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W), B, B*D
    if stack_dim == "H":
        return b_input.permute(0, 3, 1, 2, 4).reshape(B*H, C, D, W), B, B*H
    if stack_dim == "W":
        return b_input.permute(0, 4, 1, 2, 3).reshape(B*W, C, D, H), B, B*W
    else:
        raise ValueError(f"stack_dim '{stack_dim}' must be 'D' or 'H' or 'W'.")



def visualize_seg(in_type, reduce_dim=None,
                img=None, seg=None, ground_truth=None,
                crop_to_non_zero_seg=False, crop_to_non_zero_gt=False,
                alpha_seg=.4, alpha_gt=.2,
                onehot_color_map=None, n_per_row=10,
                overlay_text=None, frame_elements=None, annotate_color=(255,255,255),
                file_path=None):

    r"""
    High level display of images, segmentations / ground-truth with alpha compositing.
    Parameters:
            in_type (str): Input data type.
                One of ["batch_3D", "batch_2D", "single_3D", "single_2D"]
            reduce_dim (str): Dimension to reduce when 3D data is passed.
                One of ["D", "H", "W"]
            img (torch.tensor): Image color or gray-scale.
                Applicable dimensions depend on in_type (default: None)
            seg (torch.tensor): Segmentation.
                Applicable dimensions depend on in_type  (default: None)
            ground_truth (torch.tensor): Ground-truth.
                Applicable dimensions depend on in_type  (default: None)
            crop_to_non_zero_seg (bool): If set slices with segmentations
                will be displayed (default: False)
            crop_to_non_zero_gt (bool): If set slices with ground-truth
                will be displayed (default: False)
            alpha_seg (float):
                Properties see docstring of pil_images_from_onehot_seg()
                (Default: 0.4)
            alpha_gt (float):
                Properties see docstring of pil_images_from_onehot_seg()
                (Default: 0.4)
            onehot_color_map (dict):
                Properties see docstring of pil_images_from_onehot_seg().
                (Default: None)
            n_per_row (int): Number of overlays contained in an image row. (default: 10)
            overlay_text (str,list): White text which is displayed at image corner. (default: None)
            frame_elements (list): Boolean list identifying which tiles will be marked with a frame. (default: None)
            annotate_color (tuple): Three tuple defining RGB annotation color. (default: (255,255,255)=white)
            file_path (os.Pathlike): Path to save output image to. (default: None)

    Returns:
            None
    Example:
            color_map = get_cmap_dict(4, pyplot_map_name='rainbow')
    """
    ALL_IN_TYPES = ["batch_3D", "batch_2D", "single_3D", "single_2D"]
    REDUCE_TYPES = ["D", "H", "W"]
    assert in_type in ALL_IN_TYPES, f"in_type needs to be one of '{ALL_IN_TYPES}'"

    assert reduce_dim in REDUCE_TYPES if "3D" in in_type else True, f"For '{in_type}' specify reduce_dim as one of {REDUCE_TYPES}"
    if reduce_dim in REDUCE_TYPES and "2D" in in_type:
        warnings.warn(f"For '{in_type}' 'reduce_dim' is specified but will not be used.")

    # Check overlay vars
    assert isinstance(overlay_text, (list,tuple,str)) or overlay_text is None, \
        f"overlay_text must be of type (list,tuple,str) or be None but is {type(overlay_text)}"
    assert isinstance(frame_elements, (list,tuple)) or frame_elements is None, \
        f"frame_elements must be of type (list,tuple) or be None but is {type(overlay_text)}"

    if isinstance(overlay_text, (list,tuple)):
        assert len(overlay_text) == img.shape[0]

    if isinstance(frame_elements, (list,tuple)):
        assert len(frame_elements) == img.shape[0]

    if "3D" in in_type:
        all_spatial_shapes = [data.shape[-3:] for data in [img, seg, ground_truth] if data is not None]
        assert len(set(all_spatial_shapes)) == 1, f"DxHxW of image, segmentation and ground-truth must match but are {all_spatial_shapes}"
    if "2D" in in_type:
        all_spatial_shapes = [data.shape[-2:] for data in [img, seg, ground_truth] if data is not None]
        assert len(set(all_spatial_shapes)) == 1, f"HxW of image, segmentation and ground-truth must match but are {all_spatial_shapes}"

    if in_type == "batch_3D":
        if img is not None:
            assert img.dim() == 5, "Image batch need to have dimensions BxCxDxHxW"
            img, old_b_size, new_b_size = make_2d_stack_from_3d(img, reduce_dim)
        if seg is not None:
            assert seg.dim() == 4, "Segmentation batch need to have dimensions BxDxHxW"
            seg, old_b_size, new_b_size = make_2d_stack_from_3d(seg.unsqueeze(1), reduce_dim)
            seg = seg.squeeze(1)
        if ground_truth is not None:
            assert ground_truth.dim() == 4, "Ground-truth batch need to have dimensions BxDxHxW"
            ground_truth, old_b_size, new_b_size = make_2d_stack_from_3d(ground_truth.unsqueeze(1), reduce_dim)
            ground_truth = ground_truth.squeeze(1)
        if overlay_text is not None:
            expanded_text = [[elem]*(new_b_size//old_b_size) for elem in overlay_text]
            expanded_text = [lst_elem for lst in expanded_text for lst_elem in lst]
            overlay_text = expanded_text
        if frame_elements is not None:
            expanded_frame_elements = [[elem]*(new_b_size//old_b_size) for elem in overlay_text]
            expanded_frame_elements = [lst_elem for lst in expanded_frame_elements for lst_elem in lst]
            frame_elements = expanded_frame_elements

    elif in_type == "batch_2D":
        if img is not None:
            assert img.dim() == 4, f"Image batch need to have dimensions BxCxHxW but is {img.shape}"
        if seg is not None:
            assert seg.dim() == 3, f"Segmentation batch need to have dimensions BxHxW but is {seg.shape}"
        if ground_truth is not None:
            assert ground_truth.dim() == 3, f"Ground-truth batch need to have dimensions BxHxW but is {ground_truth.shape}"

    elif in_type == "single_3D":
        if img is not None:
            assert img.dim() == 4, f"Image needs to have dimensions CxDxHxW but is {img.shape}"
            img = make_2d_stack_from_3d(img.unsqueeze(0), reduce_dim)

        if seg is not None:
            assert seg.dim() == 3, f"Segmentation needs to have dimensions DxHxW but is {seg.shape}"
            seg = make_2d_stack_from_3d(seg.unsqueeze(0).unsqueeze(0), reduce_dim).squeeze(1)

        if ground_truth is not None:
            assert ground_truth.dim() == 3, f"Ground-truth need to have dimensions DxHxW but is {ground_truth.shape}"
            ground_truth = make_2d_stack_from_3d(ground_truth.unsqueeze(0).unsqueeze(0), reduce_dim).squeeze(1)

    elif in_type == "single_2D":
        if img is not None:
            assert img.dim() == 3, f"Image needs to have dimensions CxHxW but is {img.shape}"
            img = img.unsqueeze(0)

        if seg is not None:
            assert seg.dim() == 2, f"Segmentation needs to have dimensions HxW but is {seg.shape}"
            seg = seg.unsqueeze(0)

        if ground_truth is not None:
            assert ground_truth.dim() == 2, f"Segmentation needs to have dimensions HxW but is {ground_truth.shape}"
            ground_truth = ground_truth.unsqueeze(0)

    # Will have img 1xCxSPAT1xSPAT0
    # Will have seg/gt 1xSPAT1xSPAT0

    if crop_to_non_zero_seg or crop_to_non_zero_gt:
        selected_idxs = torch.tensor([])

        if crop_to_non_zero_seg:
            idx_depth_with_segs, *_ = torch.nonzero(seg > 0, as_tuple=True)
            selected_idxs = torch.cat([selected_idxs, idx_depth_with_segs.cpu()])

        if crop_to_non_zero_gt:
            idx_depth_with_gt, *_ = torch.nonzero(ground_truth > 0, as_tuple=True)
            selected_idxs = torch.cat([selected_idxs, idx_depth_with_gt.cpu()])

    else:
        # Select all indices
        selected_idxs = torch.arange(img.shape[0])

    if selected_idxs.numel() > 0:
        selected_idxs = selected_idxs.long().unique()

        if not onehot_color_map:
            class_max_id = torch.tensor([data.max() for data in [seg, ground_truth] if not data is None]).max()
            onehot_color_map = get_cmap_dict(class_max_id)

        img_slices = img[selected_idxs] if img is not None else None
        onehot_seg_slices = torch.nn.functional.one_hot(seg[selected_idxs], len(onehot_color_map)) if seg is not None else None
        onehot_gt_slices = torch.nn.functional.one_hot(ground_truth[selected_idxs], len(onehot_color_map)) if ground_truth is not None else None

        pil_ov, _ = get_overlay_grid(
            img_slices,
            onehot_seg_slices,
            onehot_gt_slices,
            onehot_color_map, onehot_color_map,
            alpha_seg=alpha_seg, alpha_gt=alpha_gt,
            n_per_row=n_per_row,
        )

        # Prepare drawing on tiles
        draw = PIL.ImageDraw.Draw(pil_ov)
        IM_W, IM_H = pil_ov._size
        TILE_H = IM_H//(math.ceil(img_slices.shape[0]/n_per_row))
        TILE_W = IM_W//min(n_per_row, img_slices.shape[0])

        if overlay_text:
            font_path = matplotlib.font_manager.findfont(prop=None)
            font = PIL.ImageFont.truetype(font_path, 9)

            if isinstance(overlay_text, (list,tuple)):
                overlay_text = [overlay_text[sel] for sel in selected_idxs.tolist()]

                for tile_idx, tile_txt in enumerate(overlay_text):
                    tile_y = TILE_H*(tile_idx//n_per_row)+2
                    tile_x = TILE_W*(tile_idx%n_per_row)+2
                    draw.text((tile_x, tile_y), tile_txt, annotate_color, font=font)

            elif isinstance(overlay_text, str):
                # Draw single text str
                draw.text((0, 0), overlay_text, annotate_color, font=font)

        if frame_elements:
            frame_elements = [frame_elements[sel] for sel in selected_idxs.tolist()]

            for tile_idx, do_frame in enumerate(frame_elements):
                if do_frame:
                    tile_y = TILE_H*(tile_idx//n_per_row)+1
                    tile_x = TILE_W*(tile_idx%n_per_row)+1
                    draw.rectangle([tile_x, tile_y, tile_x+TILE_W-1, tile_y+TILE_H-1], fill=None, outline=annotate_color, width=1)

        if file_path:
            pil_ov.save(file_path)
        else:
            display(pil_ov)
    else:
        print("No slices with segmentations/ground-truth to display.")