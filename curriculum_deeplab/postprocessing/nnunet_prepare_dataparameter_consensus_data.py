
import sys
sys.path.append('/home/ckruse/PythonProjects')
sys.path.append('/home/ckruse/MDL_Repositories')
import torch
import nibabel as nib
import random

from UtilityFunctions.dice import dice_coeff


out_path = '/share/data_sam2/ckruse/CrossMoDa_testdata/L4_T2_consensus/'

all_data = torch.load('/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220125_consensus/network_dataset_path_dict_400.pth')
consensus_dicts = torch.load("/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220125_consensus/consensus_dict_400.pth")
#print(consensus_dicts)
count = 0
for img_file in all_data['train_images']:
    label_file = img_file.replace('target_training_unlabeled','__omitted_labels_target_training__')
    label_file = label_file.replace('.nii.gz','_Label.nii.gz')
    file_id = img_file.split('/')[-1]
    file_id = file_id.replace('_hrT2_','').split('_')[-1].split('.')[0]
    label_dict = consensus_dicts[file_id]
    rndm_keys = list(label_dict.keys())
    rndm_keys.remove("simple_consensus")#
    rndm_keys.remove("staple_consensus")
    rndm_keys.remove("expert_label")
    rndm_keys.remove("prediction")

    rnd_ind = random.randint(0, len(rndm_keys)-1)

    rnd_key = rndm_keys[rnd_ind]

    expert_label = label_dict['expert_label'].to_dense()
    simple_consensus = label_dict['simple_consensus'].to_dense()
    staple_consensus = label_dict['staple_consensus'].to_dense()
    random_reg = label_dict[rnd_key]['warped_label'].to_dense()
    label = torch.from_numpy(nib.load(label_file).get_fdata())
    org_img = nib.load(img_file)
    image = torch.from_numpy(org_img.get_fdata())
    if 'r' in file_id:
        expert_label = expert_label.fliplr()
        simple_consensus = simple_consensus.fliplr()
        staple_consensus = staple_consensus.fliplr()
        random_reg = random_reg.fliplr()

    label = label[:,:,45:95]
    image = image[:,:,45:95]

    label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='nearest').squeeze()
    image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='trilinear').squeeze()

    img_out                 = nib.Nifti1Image(image             , org_img.affine, org_img.header)
    expert_out              = nib.Nifti1Image(expert_label      , org_img.affine, org_img.header)
    simple_consensus_out    = nib.Nifti1Image(simple_consensus  , org_img.affine, org_img.header)
    staple_consensus_out    = nib.Nifti1Image(staple_consensus  , org_img.affine, org_img.header)
    random_reg_out          = nib.Nifti1Image(random_reg        , org_img.affine, org_img.header)

    nib.save(img_out                , out_path + 'images/CrossMoDa_'+str(count).zfill(3)+'_0000.nii.gz')
    nib.save(expert_out             , out_path + 'expert_labels/CrossMoDa_'+str(count).zfill(3)+'.nii.gz')
    nib.save(simple_consensus_out   , out_path + 'simple_consensus/CrossMoDa_'+str(count).zfill(3)+'.nii.gz')
    nib.save(staple_consensus_out   , out_path + 'staple_consensus/CrossMoDa_'+str(count).zfill(3)+'.nii.gz')
    nib.save(random_reg_out         , out_path + 'random_reg/CrossMoDa_'+str(count).zfill(3)+'.nii.gz')
    count +=1
    dice_expert = dice_coeff(label,expert_label)
    dice_simple = dice_coeff(label,simple_consensus)
    dice_staple = dice_coeff(label,staple_consensus)
    dice_random =dice_coeff(label,random_reg)
    print('{}: expert: {}, simple: {}, staple:{}, random: {}'.format(file_id,dice_expert,dice_simple,dice_staple,dice_random))