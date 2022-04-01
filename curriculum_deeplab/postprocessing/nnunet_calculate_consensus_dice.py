
import sys

# sys.path.append('/home/ckruse/PythonProjects')
# sys.path.append('/home/ckruse/MDL_Repositories')
import os
import torch
import glob
import nibabel as nib
import subprocess
from mdl_seg_class.metrics import dice3d
from pathlib import Path
# from UtilityFunctions.dice import dice_coeff

# raw_data_path = Path(os.environ['nnUNet_raw_data_base']).joinpath('nnUNet_raw_data')
# task_folders = glob.glob(str(raw_data_path) + "Task55*")

# cases = ['400_deeds', '400_convex_adam']
# current_case = cases[1]

# INPUT_FOLDER = Path(f'/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/{current_case}/val_labels')

# Run prediction all trainers and folders
# for t_folder in task_folders:
#     TASK_NAME_OR_ID = re.match(r"^Task[0-9]{3}_", t_folder)
#     CONFIGURATION = '3d_fullres'
#     all_trainer_classes = ['nnUNetTrainerV2', 'nnUNetTrainerV2_insaneDA']
#     for TRAINER_CLASS in all_trainer_classes:
#         OUTPUT_FOLDER = Path(os.environ['nnUNet_inference']).joinpath(f"T{TASK_NAME_OR_ID}{}")
#         subprocess.call(
#             ['nnUNet_predict', '-i', INPUT_FOLDER, '-o', OUTPUT_FOLDER, '-t', TASK_NAME_OR_ID,
#             '-m', CONFIGURATION, '-f', 'all', '-tr', TRAINER_CLASS]
#         )


task_no = 563

if task_no == 555:
    path_gt = ''
    path_target = ''

elif task_no == 556:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task556_CM_consensus_random_convex_adam/'

elif task_no == 557:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task557_CM_consensus_dp_convex_adam/'

elif task_no == 558:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task558_CM_consensus_staple_convex_adam/'

elif task_no == 559:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task559_CM_consensus_all_convex_adam/'

elif task_no == 560:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = ''

elif task_no == 561:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_deeds/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task561_CM_domain_adaptation_insane_moving_deeds/'

elif task_no == 562:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task562_CM_domain_adaptation_insane_moving_convex_adam/'

elif task_no == 563:
    path_gt = '/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/400_convex_adam/val_labels/'
    path_target = '/share/data_rechenknecht01_2/weihsbach/nnunet/nnUNet_inference_output/Task563_CM_consensus_expert_convex_adam/'


files = sorted(os.listdir(path_gt))
dice_target = torch.zeros(len(files))

all_dice_scores = []

for i, file in enumerate(files):
    gt_label = torch.from_numpy(nib.load(path_gt+file).get_fdata())
    target_label = torch.from_numpy(nib.load(path_target+file).get_fdata())

    one_hot_gt_label = torch.nn.functional.one_hot(gt_label.long(), num_classes=3).unsqueeze(0)
    one_hot_target_label = torch.nn.functional.one_hot(target_label.long(), num_classes=3).unsqueeze(0)

    dice_score = dice3d(one_hot_gt_label, one_hot_target_label, one_hot_torch_style=True)[0,1].item()
    all_dice_scores.append(dice_score)

print(task_no, torch.tensor(all_dice_scores).mean())