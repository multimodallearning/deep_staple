# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import time
import random
import re
import warnings
import glob
import pickle
import copy
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

import functools
from enum import Enum, auto

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import scipy

import wandb
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import KFold

from deep_staple.metrics import dice3d, dice2d
from deep_staple.visualization import visualize_seg
from deep_staple.mindssc import mindssc
from deep_staple.CrossmodaHybridIdLoader import CrossmodaHybridIdLoader, get_crossmoda_data_load_closure
from deep_staple.MobileNet_LR_ASPP_3D import MobileNet_LRASPP_3D, MobileNet_ASPP_3D
from deep_staple.utils.torch_utils import get_batch_dice_per_class, get_batch_dice_over_all, get_2d_stack_batch_size, \
    make_2d_stack_from_3d, make_3d_from_2d_stack, interpolate_sample, dilate_label_class, ensure_dense, get_module, set_module, save_model, reset_determinism
from deep_staple.utils.common_utils import DotDict, DataParamMode, LabelDisturbanceMode, in_notebook, get_script_dir
from deep_staple.utils.log_utils import get_global_idx, log_data_parameter_stats, log_class_dices

print(torch.__version__)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

THIS_SCRIPT_DIR = get_script_dir(__file__)
print(f"Running in: {THIS_SCRIPT_DIR}")

# %%

config_dict = DotDict({
    'num_folds': 3,
    'only_first_fold': True,
    # 'fold_override': 0,
    # 'checkpoint_epx': 0,

    'use_mind': False,
    'epochs': 40,

    'batch_size': 8,
    'val_batch_size': 1,
    'use_2d_normal_to': None,

    'num_val_images': 20,
    'atlas_count': 1,

    'dataset': 'crossmoda',
    'dataset_directory': Path(THIS_SCRIPT_DIR, "data/crossmoda_dataset"),
    'reg_state': "acummulate_every_third_deeds_FT2_MT1",
    'train_set_max_len': None,
    'crop_3d_w_dim_range': (45, 95),
    'crop_2d_slices_gt_num_threshold': 0,

    'lr': 0.01,
    'use_scheduling': True,

    # Data parameter config
    'data_param_mode': DataParamMode.INSTANCE_PARAMS, # DataParamMode.DISABLED
    'init_inst_param': 0.0,
    'lr_inst_param': 0.1,
    'use_risk_regularization': True,
    'use_fixed_weighting': True,
    'use_ool_dp_loss': True,

    # Extended config for loading pretrained data
    'fixed_weight_file': None,
    'fixed_weight_min_quantile': None,
    'fixed_weight_min_value': None,
    'override_embedding_weights': False,

    'save_every': 200,
    'mdl_save_prefix': 'data/models',

    'debug': False,
    'wandb_mode': 'disabled', # e.g. online, disabled
    'do_sweep': False,

    'checkpoint_name': None,
    'fold_override': None,
    'checkpoint_epx': None,

    'do_plot': False,
    'save_dp_figures': False,
    'save_labels': False,

    # Disturbance settings
    'disturbance_mode': None, # LabelDisturbanceMode.FLIP_ROLL, LabelDisturbanceMode.AFFINE
    'disturbance_strength': 0.,
    'disturbed_percentage': 0.,
})



# %%
def prepare_data(config):

    assert os.path.isdir(config.dataset_directory), "Dataset directory does not exist."

    reset_determinism()
    if config.reg_state:
        print("Loading registered data.")

        if config.reg_state == "mix_combined_best":
            config.atlas_count = 1
            domain = 'source'
            label_data_left = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_left.pth"))
            label_data_right = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_right.pth"))
            loaded_identifier = label_data_left['valid_left_t1'] + label_data_right['valid_right_t1']

            perm = np.random.permutation(len(loaded_identifier))
            _clen = int(.5*len(loaded_identifier))
            best_choice = perm[:_clen]
            combined_choice = perm[_clen:]

            best_label_data = torch.cat([label_data_left['best_all'].to_dense()[:44], label_data_right['best_all'].to_dense()[:63]], dim=0)[best_choice]
            combined_label_data = torch.cat([label_data_left['combined_all'].to_dense()[:44], label_data_right['combined_all'].to_dense()[:63]], dim=0)[combined_choice]
            label_data = torch.zeros([107,128,128,128])
            label_data[best_choice] = best_label_data
            label_data[combined_choice] = combined_label_data
            var_identifier = ["mBST" if idx in best_choice else "mCMB" for idx in range(len(loaded_identifier))]
            loaded_identifier = [f"{_id}:{var_id}" for _id, var_id in zip(loaded_identifier, var_identifier)]

        elif config.reg_state == "acummulate_combined_best":
            config.atlas_count = 2
            domain = 'source'
            label_data_left = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_left.pth"))
            label_data_right = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_right.pth"))
            loaded_identifier = label_data_left['valid_left_t1'] + label_data_right['valid_right_t1']
            best_label_data = torch.cat([label_data_left['best_all'].to_dense()[:44], label_data_right['best_all'].to_dense()[:63]], dim=0)
            combined_label_data = torch.cat([label_data_left['combined_all'].to_dense()[:44], label_data_right['combined_all'].to_dense()[:63]], dim=0)
            label_data = torch.cat([best_label_data, combined_label_data])
            loaded_identifier = [_id+':mBST' for _id in loaded_identifier] + [_id+':mCMB' for _id in loaded_identifier]

        elif config.reg_state == "best":
            config.atlas_count = 1
            domain = 'source'
            label_data_left = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_left.pth"))
            label_data_right = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_right.pth"))
            loaded_identifier = label_data_left['valid_left_t1'] + label_data_right['valid_right_t1']
            label_data = torch.cat([label_data_left[config.reg_state+'_all'].to_dense()[:44], label_data_right[config.reg_state+'_all'].to_dense()[:63]], dim=0)
            postfix = 'mBST'
            loaded_identifier = [_id+':'+postfix for _id in loaded_identifier]

        elif config.reg_state == "combined":
            config.atlas_count = 1
            domain = 'source'
            label_data_left = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_left.pth"))
            label_data_right = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220113_crossmoda_optimal/optimal_reg_right.pth"))
            loaded_identifier = label_data_left['valid_left_t1'] + label_data_right['valid_right_t1']
            label_data = torch.cat([label_data_left[config.reg_state+'_all'].to_dense()[:44], label_data_right[config.reg_state+'_all'].to_dense()[:63]], dim=0)
            postfix = 'mCMB'
            loaded_identifier = [_id+':'+postfix for _id in loaded_identifier]

        elif config.reg_state == "acummulate_convex_adam_FT2_MT1":
            config.atlas_count = 10
            domain = 'target'
            bare_data = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220318_crossmoda_convex_adam_lr/crossmoda_convex_registered_new_convex.pth"))
            label_data = []
            loaded_identifier = []
            for fixed_id, moving_dict in bare_data.items():
                sorted_moving_dict = OrderedDict(moving_dict)
                for idx_mov, (moving_id, moving_sample) in enumerate(sorted_moving_dict.items()):
                    # Only use every third warped sample
                    if idx_mov % 3 == 0:
                        label_data.append(moving_sample['warped_label'].cpu())
                        loaded_identifier.append(f"{fixed_id}:m{moving_id}")

        elif config.reg_state == "acummulate_every_third_deeds_FT2_MT1":
            config.atlas_count = 10
            domain = 'target'
            bare_data = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220114_crossmoda_multiple_registrations/crossmoda_deeds_registered.pth"))
            label_data = []
            loaded_identifier = []
            for fixed_id, moving_dict in bare_data.items():
                sorted_moving_dict = OrderedDict(moving_dict)
                for idx_mov, (moving_id, moving_sample) in enumerate(sorted_moving_dict.items()):
                    # Only use every third warped sample
                    if idx_mov % 3 == 0:
                        label_data.append(moving_sample['warped_label'].cpu())
                        loaded_identifier.append(f"{fixed_id}:m{moving_id}")

        elif config.reg_state == "acummulate_every_deeds_FT2_MT1":
            config.atlas_count = 30
            domain = 'target'
            bare_data = torch.load(Path(THIS_SCRIPT_DIR, "./data_artifacts/20220114_crossmoda_multiple_registrations/crossmoda_deeds_registered.pth"))
            label_data = []
            loaded_identifier = []
            for fixed_id, moving_dict in bare_data.items():
                sorted_moving_dict = OrderedDict(moving_dict)
                for idx_mov, (moving_id, moving_sample) in enumerate(sorted_moving_dict.items()):
                    label_data.append(moving_sample['warped_label'].cpu())
                    loaded_identifier.append(f"{fixed_id}:m{moving_id}")

        else:
            raise ValueError()

        modified_3d_label_override = {}
        for idx, identifier in enumerate(loaded_identifier):
            # Find sth. like 100r:mBST or 100r:m001l
            nl_id, lr_id, m_id = re.findall(r'(\d{1,3})([lr]):m([A-Z0-9a-z]{3,4})$', identifier)[0]
            nl_id = int(nl_id)
            crossmoda_var_id = f"{nl_id:03d}{lr_id}:m{m_id}"
            modified_3d_label_override[crossmoda_var_id] = label_data[idx]

        prevent_disturbance = True

    else:
        domain = 'source'
        modified_3d_label_override = None
        prevent_disturbance = False

    if config.dataset == 'crossmoda':
        # Use double size in 2D prediction, normal size in 3D
        pre_interpolation_factor = 2. if config.use_2d_normal_to is not None else 1.5
        clsre = get_crossmoda_data_load_closure(
            base_dir=str(config.dataset_directory),
            domain=domain, state='l4', use_additional_data=False,
            size=(128,128,128), resample=True, normalize=True, crop_3d_w_dim_range=config.crop_3d_w_dim_range,
            ensure_labeled_pairs=True, modified_3d_label_override=modified_3d_label_override,
            debug=config.debug
        )
        training_dataset = CrossmodaHybridIdLoader(
            clsre,
            size=(128,128,128), resample=True, normalize=True, crop_3d_w_dim_range=config.crop_3d_w_dim_range,
            ensure_labeled_pairs=True,
            max_load_3d_num=config.train_set_max_len,
            prevent_disturbance=prevent_disturbance,
            use_2d_normal_to=config.use_2d_normal_to,
            crop_2d_slices_gt_num_threshold=config.crop_2d_slices_gt_num_threshold,
            pre_interpolation_factor=pre_interpolation_factor,
            fixed_weight_file=config.fixed_weight_file, fixed_weight_min_quantile=config.fixed_weight_min_quantile, fixed_weight_min_value=config.fixed_weight_min_value,
        )

    return training_dataset

# %%
if config_dict['do_plot'] and False:
    # Plot label voxel W-dim distribution
    training_dataset = prepare_data(config_dict)
    _, all_labels, _ = training_dataset.get_data(use_2d_override=False)
    print(all_labels.shape)
    sum_over_w = torch.sum(all_labels, dim=(0,1,2))
    plt.xlabel("W")
    plt.ylabel("ground truth>0")
    plt.plot(sum_over_w);


# %%
def save_parameter_figure(_path, title, text, parameters, reweighted_parameters, dices):
    # Show weights and weights with compensation
    fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=80)
    sc1 = axs[0].scatter(
        range(len(parameters)),
        parameters.cpu().detach(), c=dices,s=1, cmap='plasma', vmin=0., vmax=1.)
    sc2 = axs[1].scatter(
        range(len(reweighted_parameters)),
        reweighted_parameters.cpu().detach(), s=1,c=dices, cmap='plasma', vmin=0., vmax=1.)

    fig.suptitle(title, fontsize=14)
    fig.text(0, 0, text)
    axs[0].set_title('Bare parameters')
    axs[1].set_title('Reweighted parameters')
    axs[0].set_ylim(-10, 10)
    axs[1].set_ylim(-3, 1)
    plt.colorbar(sc2)
    plt.savefig(_path)
    plt.clf()
    plt.close()



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


# %%


# %%
if config_dict['do_plot']:
    training_dataset = prepare_data(config_dict)

    # Print transformed 2D data
    training_dataset.train(use_modified=True, augment=False)
    # print(training_dataset.disturbed_idxs)

    print("Displaying 2D training sample")

    img_stack = []
    label_stack = []
    mod_label_stack = []

    for sample in (training_dataset[idx] for idx in [500,590]):
        print(sample['id'])
        img_stack.append(sample['image'])
        label_stack.append(sample['label'])
        mod_label_stack.append(sample['modified_label'])

    # Change label num == hue shift for display
    img_stack = torch.stack(img_stack).unsqueeze(1)
    label_stack = torch.stack(label_stack)
    mod_label_stack = torch.stack(mod_label_stack)

    mod_label_stack*=4

    visualize_seg(in_type="batch_3D", reduce_dim="W",
        img=img_stack,
        # ground_truth=label_stack,
        seg=(mod_label_stack-label_stack).abs(),
        # crop_to_non_zero_gt=True,
        crop_to_non_zero_seg=True,
        alpha_seg = .5
    )




def get_model(config, dataset_len, num_classes, THIS_SCRIPT_DIR, _path=None, device='cpu'):
    _path = Path(THIS_SCRIPT_DIR).joinpath(_path).resolve()

    if config.use_mind:
        in_channels = 12
    else:
        in_channels = 1

    if config.use_2d_normal_to is not None:
        # Use vanilla torch model
        lraspp = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            pretrained=False, progress=True, num_classes=num_classes
        )
        set_module(lraspp, 'backbone.0.0',
            torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2),
                            padding=(1, 1), bias=False)
        )
    else:
        # Use custom 3d model
        lraspp = MobileNet_LRASPP_3D(
            in_num=in_channels, num_classes=num_classes,
            use_checkpointing=True
        )

    # lraspp.register_parameter('sigmoid_offset', nn.Parameter(torch.tensor([0.])))
    lraspp.to(device)
    print(f"Param count lraspp: {sum(p.numel() for p in lraspp.parameters())}")

    optimizer = torch.optim.AdamW(lraspp.parameters(), lr=config.lr)
    scaler = amp.GradScaler()

    if config.use_2d_normal_to is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)
    else:
        # Use ExponentialLR in 3D
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.99)

    # Add data paramters embedding and optimizer
    if config.data_param_mode == str(DataParamMode.INSTANCE_PARAMS):
        embedding = nn.Embedding(dataset_len, 1, sparse=True)
        embedding = embedding.to(device)

        # Init embedding values
        #
        if config.override_embedding_weights:
            fixed_weightdata = torch.load(config.fixed_weight_file)
            fixed_weights = fixed_weightdata['data_parameters']
            fixed_d_ids = fixed_weightdata['d_ids']
            if config.use_2d_normal_to is not None:
                corresp_dataset_idxs = [training_dataset.get_2d_ids().index(_id) for _id in fixed_d_ids]
            else:
                corresp_dataset_idxs = [training_dataset.get_3d_ids().index(_id) for _id in fixed_d_ids]
            embedding_weight_tensor = torch.zeros_like(embedding.weight)
            embedding_weight_tensor[corresp_dataset_idxs] = fixed_weights.view(-1,1).cuda()
            embedding = nn.Embedding(len(training_dataset), 1, sparse=True, _weight=embedding_weight_tensor)

        elif _path and _path.is_dir():
            embedding.load_state_dict(torch.load(_path.joinpath('embedding.pth'), map_location=device))
        else:
            torch.nn.init.normal_(embedding.weight.data, mean=config.init_inst_param, std=0.00)

        print(f"Param count embedding: {sum(p.numel() for p in embedding.parameters())}")

        optimizer_dp = torch.optim.SparseAdam(
            embedding.parameters(), lr=config.lr_inst_param,
            betas=(0.9, 0.999), eps=1e-08)
        scaler_dp =  amp.GradScaler()

        if _path and _path.is_dir():
            print(f"Loading dp_optimizer and scaler_dp from {_path}")
            optimizer_dp.load_state_dict(torch.load(_path.joinpath('optimizer_dp.pth'), map_location=device))
            scaler_dp.load_state_dict(torch.load(_path.joinpath('scaler_dp.pth'), map_location=device))

    else:
        embedding = None
        optimizer_dp = None
        scaler_dp = None

    if _path and _path.is_dir():
        print(f"Loading lr-aspp model, optimizers and grad scalers from {_path}")
        lraspp.load_state_dict(torch.load(_path.joinpath('lraspp.pth'), map_location=device))
        optimizer.load_state_dict(torch.load(_path.joinpath('optimizer.pth'), map_location=device))
        scheduler.load_state_dict(torch.load(_path.joinpath('scheduler.pth'), map_location=device))
        scaler.load_state_dict(torch.load(_path.joinpath('scaler.pth'), map_location=device))
    else:
        print("Generating fresh lr-aspp model, optimizer and grad scaler.")

    return (lraspp, optimizer, scheduler, optimizer_dp, embedding, scaler, scaler_dp)



# %%
def inference_wrap(lraspp, img, use_2d, use_mind):
    with torch.inference_mode():
        b_img = img.unsqueeze(0).unsqueeze(0).float()
        if use_2d and use_mind:
            # MIND 2D, in Bx1x1xHxW, out BxMINDxHxW
            b_img = mindssc(b_img.unsqueeze(0)).squeeze(2)
        elif not use_2d and use_mind:
            # MIND 3D in Bx1xDxHxW out BxMINDxDxHxW
            b_img = mindssc(b_img)
        elif use_2d or not use_2d:
            # 2D Bx1xHxW
            # 3D out Bx1xDxHxW
            pass

        b_out = lraspp(b_img)['out']
        b_out = b_out.argmax(1)
        return b_out



def train_DL(run_name, config, training_dataset):
    reset_determinism()

    # Configure folds
    kf = KFold(n_splits=config.num_folds)
    # kf.get_n_splits(training_dataset.__len__(use_2d_override=False))
    fold_iter = enumerate(kf.split(range(training_dataset.__len__(use_2d_override=False))))

    if config.get('fold_override', None):
        selected_fold = config.get('fold_override', 0)
        fold_iter = list(fold_iter)[selected_fold:selected_fold+1]
    elif config.only_first_fold:
        fold_iter = list(fold_iter)[0:1]

    if config.wandb_mode != 'disabled':
        warnings.warn("Logging of dataset file paths is disabled.")
        # # Log dataset info
        # training_dataset.eval()
        # dataset_info = [[smp['dataset_idx'], smp['id'], smp['image_path'], smp['label_path']] \
        #                 for smp in training_dataset]
        # wandb.log({'datasets/training_dataset':wandb.Table(columns=['dataset_idx', 'id', 'image', 'label'], data=dataset_info)}, step=0)

    if config.use_2d_normal_to is not None:
        n_dims = (-2,-1)
    else:
        n_dims = (-3,-2,-1)

    fold_means_no_bg = []

    for fold_idx, (train_idxs, val_idxs) in fold_iter:
        train_idxs = torch.tensor(train_idxs)
        val_idxs = torch.tensor(val_idxs)
        all_3d_ids = training_dataset.get_3d_ids()

        if config.debug:
            num_val_images = 2
            atlas_count = 1
        else:
            num_val_images = config.num_val_images
            atlas_count = config.atlas_count

        if config.use_2d_normal_to is not None:
            # Override idxs
            all_3d_ids = training_dataset.get_3d_ids()

            val_3d_idxs = torch.tensor(list(range(0, num_val_images*atlas_count, atlas_count)))
            val_3d_ids = training_dataset.switch_3d_identifiers(val_3d_idxs)

            train_3d_idxs = list(range(num_val_images*atlas_count, len(all_3d_ids)))

            # Get corresponding 2D idxs
            train_2d_ids = []
            dcts = training_dataset.get_id_dicts()
            for id_dict in dcts:
                _2d_id = id_dict['2d_id']
                _3d_idx = id_dict['3d_dataset_idx']
                if _2d_id in training_dataset.label_data_2d.keys() and _3d_idx in train_3d_idxs:
                    train_2d_ids.append(_2d_id)

            train_2d_idxs = training_dataset.switch_2d_identifiers(train_2d_ids)
            train_idxs = torch.tensor(train_2d_idxs)

        else:
            val_3d_idxs = torch.tensor(list(range(0, num_val_images*atlas_count, atlas_count)))
            val_3d_ids = training_dataset.switch_3d_identifiers(val_3d_idxs)

            train_3d_idxs = list(range(num_val_images*atlas_count, len(all_3d_ids)))
            train_idxs = torch.tensor(train_3d_idxs)

        print(f"Will run validation with these 3D samples (#{len(val_3d_ids)}):", sorted(val_3d_ids))

        _, _, all_modified_segs = training_dataset.get_data()

        if config.disturbed_percentage > 0.:
            with torch.no_grad():
                non_empty_train_idxs = [(all_modified_segs[train_idxs].sum(dim=n_dims) > 0)]

            ### Disturb dataset (only non-emtpy idxs)###
            proposed_disturbed_idxs = np.random.choice(non_empty_train_idxs, size=int(len(non_empty_train_idxs)*config.disturbed_percentage), replace=False)
            proposed_disturbed_idxs = torch.tensor(proposed_disturbed_idxs)
            training_dataset.disturb_idxs(proposed_disturbed_idxs,
                disturbance_mode=config.disturbance_mode,
                disturbance_strength=config.disturbance_strength
            )
            disturbed_bool_vect = torch.zeros(len(training_dataset))
            disturbed_bool_vect[training_dataset.disturbed_idxs] = 1.

        else:
            disturbed_bool_vect = torch.zeros(len(training_dataset))

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

        ### Add train sampler and dataloaders ##
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
        # val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size,
            sampler=train_subsampler, pin_memory=False, drop_last=False,
            # collate_fn=training_dataset.get_efficient_augmentation_collate_fn()
        )

        # training_dataset.set_augment_at_collate(True) # This function does not work as expected. Scores get worse.

        ### Get model, data parameters, optimizers for model and data parameters, as well as grad scaler ###
        if 'checkpoint_epx' in config and config['checkpoint_epx'] is not None:
            epx_start = config['checkpoint_epx']
        else:
            epx_start = 0

        if config.checkpoint_name:
            # Load from checkpoint
            _path = f"{config.mdl_save_prefix}/{config.checkpoint_name}_fold{fold_idx}_epx{epx_start}"
        else:
            _path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx_start}"

        (lraspp, optimizer, scheduler, optimizer_dp, embedding, scaler, scaler_dp) = get_model(config, len(training_dataset), len(training_dataset.label_tags),
            THIS_SCRIPT_DIR=THIS_SCRIPT_DIR, _path=_path, device='cuda')

        t_start = time.time()

        dice_func = dice2d if config.use_2d_normal_to is not None else dice3d

        bn_count = torch.zeros([len(training_dataset.label_tags)], device=all_modified_segs.device)
        wise_dice = torch.zeros([len(training_dataset), len(training_dataset.label_tags)])
        gt_num = torch.zeros([len(training_dataset)])

        with torch.no_grad():
            print("Fetching training metrics for samples.")
            # _, wise_lbls, mod_lbls = training_dataset.get_data()
            training_dataset.eval(use_modified=True)
            for sample in tqdm((training_dataset[idx] for idx in train_idxs), desc="metric:", total=len(train_idxs)):
                d_idxs = sample['dataset_idx']
                wise_label, mod_label = sample['label'], sample['modified_label']
                mod_label = mod_label.cuda()
                wise_label = wise_label.cuda()
                mod_label, _ = ensure_dense(mod_label)

                dsc = dice_func(
                    torch.nn.functional.one_hot(wise_label.unsqueeze(0), len(training_dataset.label_tags)),
                    torch.nn.functional.one_hot(mod_label.unsqueeze(0), len(training_dataset.label_tags)),
                    one_hot_torch_style=True, nan_for_unlabeled_target=False
                )
                bn_count += torch.bincount(mod_label.reshape(-1).long(), minlength=len(training_dataset.label_tags)).cpu()
                wise_dice[d_idxs] = dsc.cpu()
                gt_num[d_idxs] = (mod_label > 0).sum(dim=n_dims).float().cpu()

            class_weights = 1 / (bn_count).float().pow(.35)
            class_weights /= class_weights.mean()

            fixed_weighting = (gt_num+np.exp(1)).log()+np.exp(1)

        class_weights = class_weights.cuda()
        fixed_weighting = fixed_weighting.cuda()

        for epx in range(epx_start, config.epochs):
            global_idx = get_global_idx(fold_idx, epx, config.epochs)

            lraspp.train()

            ### Disturb samples ###
            training_dataset.train(use_modified=True)

            epx_losses = []
            dices = []
            class_dices = []

            # Load data
            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="batch:", total=len(train_dataloader)):

                optimizer.zero_grad()
                if optimizer_dp:
                    optimizer_dp.zero_grad()

                b_img = batch['image']
                b_seg = batch['label']

                b_seg_modified = batch['modified_label']
                b_idxs_dataset = batch['dataset_idx']
                b_img = b_img.float()

                b_img = b_img.cuda()
                b_seg_modified = b_seg_modified.cuda()
                b_idxs_dataset = b_idxs_dataset.cuda()
                b_seg = b_seg.cuda()

                if training_dataset.use_2d() and config.use_mind:
                    # MIND 2D, in Bx1x1xHxW, out BxMINDxHxW
                    b_img = mindssc(b_img.unsqueeze(1).unsqueeze(1)).squeeze(2)
                elif not training_dataset.use_2d() and config.use_mind:
                    # MIND 3D
                    b_img = mindssc(b_img.unsqueeze(1))
                else:
                    b_img = b_img.unsqueeze(1)

                ### Forward pass ###
                with amp.autocast(enabled=True):
                    assert b_img.dim() == len(n_dims)+2, \
                        f"Input image for model must be {len(n_dims)+2}D: BxCxSPATIAL but is {b_img.shape}"
                    for param in lraspp.parameters():
                        param.requires_grad = True

                    lraspp.use_checkpointing = True
                    logits = lraspp(b_img)['out']

                    ### Calculate loss ###
                    assert logits.dim() == len(n_dims)+2, \
                        f"Input shape for loss must be BxNUM_CLASSESxSPATIAL but is {logits.shape}"
                    assert b_seg_modified.dim() == len(n_dims)+1, \
                        f"Target shape for loss must be BxSPATIAL but is {b_seg_modified.shape}"

                    ce_loss = nn.CrossEntropyLoss(class_weights)(logits, b_seg_modified)

                    if config.data_param_mode == str(DataParamMode.DISABLED) or config.use_ool_dp_loss:
                        scaler.scale(ce_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    if config.data_param_mode == str(DataParamMode.INSTANCE_PARAMS):
                        if config.use_ool_dp_loss:
                            # Run second consecutive forward pass
                            for param in lraspp.parameters():
                                param.requires_grad = False
                            lraspp.use_checkpointing = False
                            dp_logits = lraspp(b_img)['out']

                        else:
                            # Do not run a second forward pass
                            for param in lraspp.parameters():
                                param.requires_grad = True
                            lraspp.use_checkpointing = True
                            dp_logits = logits

                        dp_loss = nn.CrossEntropyLoss(reduction='none')(dp_logits, b_seg_modified)
                        dp_loss = dp_loss.mean(n_dims)

                        bare_weight = embedding(b_idxs_dataset).squeeze()

                        weight = torch.sigmoid(bare_weight)
                        weight = weight/weight.mean()

                        # This improves scores significantly: Reweight with log(gt_numel)
                        if config.use_fixed_weighting:
                            weight = weight/fixed_weighting[b_idxs_dataset]

                        if config.use_risk_regularization:
                            p_pred_num = (dp_logits.argmax(1) > 0).sum(dim=n_dims).detach()
                            if config.use_2d_normal_to is not None:
                                risk_regularization = -weight*p_pred_num/(dp_logits.shape[-2]*dp_logits.shape[-1])
                            else:
                                risk_regularization = -weight*p_pred_num/(dp_logits.shape[-3]*dp_logits.shape[-2]*dp_logits.shape[-1])

                            dp_loss = (dp_loss*weight).sum() + risk_regularization.sum()
                        else:
                            dp_loss = (dp_loss*weight).sum()

                if str(config.data_param_mode) != str(DataParamMode.DISABLED):
                    scaler_dp.scale(dp_loss).backward()

                    if config.use_ool_dp_loss:
                        # LRASPP already stepped.
                        if not config.override_embedding_weights:
                            scaler_dp.step(optimizer_dp)
                            scaler_dp.update()
                    else:
                        scaler_dp.step(optimizer)
                        if not config.override_embedding_weights:
                            scaler_dp.step(optimizer_dp)
                        scaler_dp.update()

                    epx_losses.append(dp_loss.item())
                else:
                    epx_losses.append(ce_loss.item())

                logits_for_score = logits.argmax(1)

                # Calculate dice score
                b_dice = dice_func(
                    torch.nn.functional.one_hot(logits_for_score, len(training_dataset.label_tags)),
                    torch.nn.functional.one_hot(b_seg, len(training_dataset.label_tags)), # Calculate dice score with original segmentation (no disturbance)
                    one_hot_torch_style=True
                )

                dices.append(get_batch_dice_over_all(
                    b_dice, exclude_bg=True))
                class_dices.append(get_batch_dice_per_class(
                    b_dice, training_dataset.label_tags, exclude_bg=True))

                ###  Scheduler management ###
                if config.use_scheduling and epx % atlas_count == 0:
                    scheduler.step()

                if str(config.data_param_mode) != str(DataParamMode.DISABLED) and batch_idx % 10 == 0 and config.save_dp_figures:
                    # Output data parameter figure
                    train_params = embedding.weight[train_idxs].squeeze()
                    # order = np.argsort(train_params.cpu().detach()) # Order by DP value
                    order = torch.arange(len(train_params))
                    pearson_corr_coeff = np.corrcoef(train_params.cpu().detach(), wise_dice[train_idxs][:,1].cpu().detach())[0,1]
                    dp_figure_path = Path(f"data/output_figures/{wandb.run.name}_fold{fold_idx}/dp_figure_epx{epx:03d}_batch{batch_idx:03d}.png")
                    dp_figure_path.parent.mkdir(parents=True, exist_ok=True)
                    save_parameter_figure(dp_figure_path, wandb.run.name, f"corr. coeff. DP vs. dice(expert label, train gt): {pearson_corr_coeff:4f}",
                        train_params[order], train_params[order]/fixed_weighting[train_idxs][order], dices=wise_dice[train_idxs][:,1][order])

                if config.debug:
                    break

            ### Logging ###
            print(f"### Log epoch {epx} @ {time.time()-t_start:.2f}s")
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
            if str(config.data_param_mode) != str(DataParamMode.DISABLED):
                # Calculate dice score corr coeff (unknown to network)
                train_params = embedding.weight[train_idxs].squeeze()
                order = np.argsort(train_params.cpu().detach())
                pearson_corr_coeff = np.corrcoef(train_params[order].cpu().detach(), wise_dice[train_idxs][:,1][order].cpu().detach())[0,1]
                spearman_corr_coeff, spearman_p = scipy.stats.spearmanr(train_params[order].cpu().detach(), wise_dice[train_idxs][:,1][order].cpu().detach())

                wandb.log(
                    {f'data_parameters/pearson_corr_coeff_fold{fold_idx}': pearson_corr_coeff},
                    step=global_idx
                )
                wandb.log(
                    {f'data_parameters/spearman_corr_coeff_fold{fold_idx}': spearman_corr_coeff},
                    step=global_idx
                )
                wandb.log(
                    {f'data_parameters/spearman_p_fold{fold_idx}': spearman_p},
                    step=global_idx
                )
                print(f'data_parameters/pearson_corr_coeff_fold{fold_idx}', f"{pearson_corr_coeff:.2f}")
                print(f'data_parameters/spearman_corr_coeff_fold{fold_idx}', f"{spearman_corr_coeff:.2f}")
                print(f'data_parameters/spearman_p_fold{fold_idx}', f"{spearman_p:.5f}")

                # Log stats of data parameters and figure
                log_data_parameter_stats(f'data_parameters/iter_stats_fold{fold_idx}', global_idx, embedding.weight.data)

            if (epx % config.save_every == 0 and epx != 0) \
                or (epx+1 == config.epochs):
                _path = f"{config.mdl_save_prefix}/{wandb.run.name}_fold{fold_idx}_epx{epx}"
                save_model(
                    _path,
                    lraspp=lraspp,
                    optimizer=optimizer, optimizer_dp=optimizer_dp,
                    scheduler=scheduler,
                    embedding=embedding,
                    scaler=scaler,
                    scaler_dp=scaler_dp)

                (lraspp, optimizer, optimizer_dp, embedding, scaler) = \
                    get_model(
                        config, len(training_dataset),
                        len(training_dataset.label_tags),
                        THIS_SCRIPT_DIR=THIS_SCRIPT_DIR,
                        _path=_path, device='cuda')

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
                        stack_dim = training_dataset.use_2d_normal_to
                        # Create batch out of single val sample
                        b_val_img = val_sample['image'].unsqueeze(0)
                        b_val_seg = val_sample['label'].unsqueeze(0)

                        B = b_val_img.shape[0]

                        b_val_img = b_val_img.unsqueeze(1).float().cuda()
                        b_val_seg = b_val_seg.cuda()

                        if training_dataset.use_2d():
                            b_val_img_2d = make_2d_stack_from_3d(b_val_img, stack_dim=training_dataset.use_2d_normal_to)

                            if config.use_mind:
                                # MIND 2D model, in Bx1x1xHxW, out BxMINDxHxW
                                b_val_img_2d = mindssc(b_val_img_2d.unsqueeze(1)).squeeze(2)

                            output_val = lraspp(b_val_img_2d)['out']
                            val_logits_for_score = output_val.argmax(1)
                            # Prepare logits for scoring
                            # Scoring happens in 3D again - unstack batch tensor again to stack of 3D
                            val_logits_for_score = make_3d_from_2d_stack(
                                val_logits_for_score.unsqueeze(1), stack_dim, B
                            ).squeeze(1)

                        else:
                            if config.use_mind:
                                # MIND 3D model shape BxMINDxDxHxW
                                b_val_img = mindssc(b_val_img)
                            else:
                                # 3D model shape Bx1xDxHxW
                                pass

                            output_val = lraspp(b_val_img)['out']
                            val_logits_for_score = output_val.argmax(1)

                        b_val_dice = dice3d(
                            torch.nn.functional.one_hot(val_logits_for_score, len(training_dataset.label_tags)),
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
                                seg=val_logits_for_score_3d.squeeze(0).cpu(),
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

        if str(config.data_param_mode) == str(DataParamMode.INSTANCE_PARAMS):
            # Write sample data
            save_dict = {}

            training_dataset.eval(use_modified=True)
            all_idxs = torch.tensor(range(len(training_dataset))).cuda()
            train_label_snapshot_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}/train_label_snapshot.pth")
            seg_viz_out_path = Path(THIS_SCRIPT_DIR).joinpath(f"data/output/{wandb.run.name}_fold{fold_idx}_epx{epx}/data_parameter_weighted_samples.png")

            train_label_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

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
                    # sample['image'],
                    sample['label'].to_sparse(),
                    sample['modified_label'].to_sparse(),
                    inference_wrap(lraspp, sample['image'].cuda(), use_2d=training_dataset.use_2d(), use_mind=config.use_mind).to_sparse()
                )
                save_data.append(data_tuple)

            save_data = sorted(save_data, key=lambda tpl: tpl[0])
            (dp_weight, disturb_flags,
                d_ids, dataset_idxs,
            #  _imgs,
                _labels, _modified_labels, _predictions) = zip(*save_data)

            dp_weight = torch.stack(dp_weight)
            dataset_idxs = torch.stack(dataset_idxs)

            save_dict.update(
                {
                    'data_parameters': dp_weight.cpu(),
                    'disturb_flags': disturb_flags,
                    'd_ids': d_ids,
                    'dataset_idxs': dataset_idxs.cpu(),
                }
            )

            if config.save_labels:
                _labels = torch.stack(_labels)
                _modified_labels = torch.stack(_modified_labels)
                _predictions = torch.stack(_predictions)
                save_dict.update(
                    {
                        'labels': _labels.cpu(),
                        'modified_labels': _modified_labels.cpu(),
                        'train_predictions': _predictions.cpu()
                    }
                )

            print(f"Writing data parameters output to '{train_label_snapshot_path}'")
            torch.save(save_dict, train_label_snapshot_path)

            if len(training_dataset.disturbed_idxs) > 0:
                # Log histogram
                separated_params = list(zip(dp_weights[clean_idxs], dp_weights[training_dataset.disturbed_idxs]))
                s_table = wandb.Table(columns=['clean_idxs', 'disturbed_idxs'], data=separated_params)
                fields = {"primary_bins": "clean_idxs", "secondary_bins": "disturbed_idxs", "title": "Data parameter composite histogram"}
                composite_histogram = wandb.plot_table(vega_spec_name="rap1ide/composite_histogram", data_table=s_table, fields=fields)
                wandb.log({f"data_parameters/separated_params_fold_{fold_idx}": composite_histogram})

            # Write out data of modified and un-modified labels and an overview image

            if training_dataset.use_2d():
                reduce_dim = None
                in_type = "batch_2D"
                skip_writeout = len(training_dataset) > 3000 # Restrict dataset size to be visualized
            else:
                reduce_dim = "W"
                in_type = "batch_3D"
                skip_writeout = len(training_dataset) > 150 # Restrict dataset size to be visualized
            skip_writeout = True

            if not skip_writeout:
                print("Writing train sample image.")
                # overlay text example: d_idx=0, dp_i=1.00, dist? False
                overlay_text_list = [f"id:{d_id} dp:{instance_p.item():.2f}" \
                    for d_id, instance_p, disturb_flg in zip(d_ids, dp_weight, disturb_flags)]

                use_2d = training_dataset.use_2d()
                scf = 1/training_dataset.pre_interpolation_factor

                show_img = interpolate_sample(b_label=_labels.to_dense(), scale_factor=scf, use_2d=use_2d)[1].unsqueeze(1)
                show_seg = interpolate_sample(b_label=_predictions.to_dense().squeeze(1), scale_factor=scf, use_2d=use_2d)[1]
                show_gt = interpolate_sample(b_label=_modified_labels.to_dense(), scale_factor=scf, use_2d=use_2d)[1]

                visualize_seg(in_type=in_type, reduce_dim=reduce_dim,
                    img=show_img, # Expert label in BW
                    seg=4*show_seg, # Prediction in blue
                    ground_truth=show_gt, # Modified label in red
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
# config_dict['checkpoint_name'] = 'ethereal-serenity-1138'
# config_dict['fold_override'] = 0
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
        # disturbance_strength=dict(
        #     values=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        # ),
        # disturbed_percentage=dict(
        #     values=[0.3, 0.6]
        # ),
        # data_param_mode=dict(
        #     values=[
        #         DataParamMode.INSTANCE_PARAMS,
        #         DataParamMode.DISABLED,
        #     ]
        # ),
        use_risk_regularization=dict(
            values=[False, True]
        ),
        use_fixed_weighting=dict(
            values=[False, True]
        ),
        # fixed_weight_min_quantile=dict(
        #     values=[0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        # ),
    )
)

# %%
def normal_run():
    with wandb.init(project="deep_staple", group="training", job_type="train",
            config=config_dict, settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        ) as run:

        run_name = run.name
        print("Running", run_name)
        training_dataset = prepare_data(config_dict)
        config = wandb.config

        train_DL(run_name, config, training_dataset)

def sweep_run():
    with wandb.init() as run:
        run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            mode=config_dict['wandb_mode']
        )

        run_name = run.name
        print("Running", run_name)
        training_dataset = prepare_data(config)
        config = wandb.config

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

    sweep_id = wandb.sweep(merged_sweep_config_dict, project="deep_staple")
    wandb.agent(sweep_id, function=sweep_run)

else:
    normal_run()

# %%
if not in_notebook():
    sys.exit(0)

# %%
# Do any postprocessing / visualization in notebook here
