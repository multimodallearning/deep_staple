
import sys

sys.path.append('/home/ckruse/PythonProjects')
sys.path.append('/home/ckruse/MDL_Repositories')
import os
import torch
import nibabel as nib

from UtilityFunctions.dice import dice_coeff

path_gt = '/share/data_sam2/ckruse/CrossMoDa_testdata/L4_T2_consensus/val_labels/'
path_expert = '/share/data_sam2/ckruse/CrossMoDa_testdata/nnUNet_output/T550_L4_T2_val/'
path_random ='/share/data_sam2/ckruse/CrossMoDa_testdata/nnUNet_output/T551_L4_T2_val/'
path_simple ='/share/data_sam2/ckruse/CrossMoDa_testdata/nnUNet_output/T552_L4_T2_val/'
path_staple ='/share/data_sam2/ckruse/CrossMoDa_testdata/nnUNet_output/T553_L4_T2_val/'
path_all ='/share/data_sam2/ckruse/CrossMoDa_testdata/nnUNet_output/T554_L4_T2_val/'
files = sorted(os.listdir(path_gt))
dice_expert = torch.zeros(len(files))
dice_simple = torch.zeros(len(files))
dice_staple = torch.zeros(len(files))
dice_random = torch.zeros(len(files))
dice_all = torch.zeros(len(files))
for i,file in enumerate(files):
    label = torch.from_numpy(nib.load(path_gt+file).get_fdata())
    expert_label = torch.from_numpy(nib.load(path_expert+file).get_fdata())
    simple_consensus = torch.from_numpy(nib.load(path_simple+file).get_fdata())
    staple_consensus = torch.from_numpy(nib.load(path_staple+file).get_fdata())
    random_reg = torch.from_numpy(nib.load(path_random+file).get_fdata())
    all_reg = torch.from_numpy(nib.load(path_all+file).get_fdata())
    dice_expert[i] = dice_coeff(label,expert_label).item()
    dice_simple[i] = dice_coeff(label,simple_consensus).item()
    dice_staple[i] = dice_coeff(label,staple_consensus).item()
    dice_random[i] = dice_coeff(label,random_reg).item()
    dice_all[i] = dice_coeff(label,all_reg).item()
print('mean dice scores wenn trained on expert: {}, simple:{}, staple:{}, random: {}, all: {}'.format(dice_expert.mean(),dice_simple.mean(),dice_staple.mean(),dice_random.mean(),dice_all.mean()))