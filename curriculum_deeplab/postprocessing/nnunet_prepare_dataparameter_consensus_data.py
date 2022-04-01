
import sys
import torch
import nibabel as nib
import random
from pathlib import Path
from mdl_seg_class.metrics import dice3d

cases = ['400_deeds', '400_convex_adam']

current_case = cases[0]

out_path = Path(f'/share/data_rechenknecht01_2/weihsbach/nnunet/tmp/{current_case}/')
for subfolder in ['all_images','all_reg','images', 'expert_labels', 'dp_consensus', 'staple_consensus', 'random_reg', 'val_images', 'val_labels']:
    Path(out_path.joinpath(subfolder)).mkdir(parents=True, exist_ok=True)

if current_case == '400_deeds':
    all_data = torch.load('/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220125_consensus/network_dataset_path_dict_400.pth')
    all_data['train_image_paths'] = {_path: _path for _path in all_data['train_images']}
    all_data['val_image_paths'] = {_path: _path for _path in all_data['val_images']}
    consensus_dicts = torch.load("/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220125_consensus/consensus_dict_400_deeds.pth")

elif current_case == '400_convex_adam':
    all_data = torch.load("/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220323_consensus_convex_adam/network_dataset_path_dict_train_400.pth")
    consensus_dicts = torch.load("/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220323_consensus_convex_adam/consensus_dict_400_convex_adam.pth")

count = 0

for img_file in all_data['train_image_paths'].values():
    label_file = img_file.replace('target_training_unlabeled','__omitted_labels_target_training__')
    label_file = label_file.replace('.nii.gz','_Label.nii.gz')
    file_id = img_file.split('/')[-1]
    file_id = file_id.replace('_hrT2_','').split('_')[-1].split('.')[0]
    label_dict = consensus_dicts[file_id]
    m_ids = list(label_dict.keys())
    m_ids.remove("dp_consensus")#
    m_ids.remove("staple_consensus")
    m_ids.remove("expert_label")
    m_ids.remove("prediction")
    m_ids.remove("image_path")
    m_ids.remove("dp_consensus_oracle_dice")
    m_ids.remove("staple_consensus_oracle_dice")

    rnd_ind = random.randint(0, len(m_ids)-1)

    rnd_key = m_ids[rnd_ind]

    expert_label = label_dict['expert_label'].to_dense()
    dp_consensus = label_dict['dp_consensus'].to_dense()
    staple_consensus = label_dict['staple_consensus'].to_dense()
    random_reg = label_dict[rnd_key]['warped_label'].to_dense()

    atlas_label = label_dict[m_ids[count % 10]]['warped_label'].to_dense()

    org_img = nib.load(img_file)
    label = torch.from_numpy(nib.load(label_file).get_fdata())
    image = torch.from_numpy(org_img.get_fdata())
    if 'r' in file_id:
        atlas_label = atlas_label.fliplr()
        expert_label = expert_label.fliplr()
        dp_consensus = dp_consensus.fliplr()
        staple_consensus = staple_consensus.fliplr()
        random_reg = random_reg.fliplr()

    label = label[:,:,45:95]
    image = image[:,:,45:95]

    label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='nearest').squeeze()
    image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='trilinear').squeeze()

    img_out                 = nib.Nifti1Image(image             , org_img.affine, org_img.header)
    atlas_out               = nib.Nifti1Image(atlas_label       , org_img.affine, org_img.header)
    expert_out              = nib.Nifti1Image(expert_label      , org_img.affine, org_img.header)
    simple_consensus_out    = nib.Nifti1Image(dp_consensus      , org_img.affine, org_img.header)
    staple_consensus_out    = nib.Nifti1Image(staple_consensus  , org_img.affine, org_img.header)
    random_reg_out          = nib.Nifti1Image(random_reg        , org_img.affine, org_img.header)

    nib.save(img_out, out_path.joinpath('all_images/CrossMoDa_'+str(count).zfill(3)+'_0000.nii.gz'))
    nib.save(atlas_out, out_path.joinpath('all_reg/CrossMoDa_'+str(count).zfill(3)+'.nii.gz'))

    if count % 10 == 0:
        # Here a new fixed id is drawn
        nib.save(img_out, out_path.joinpath('images/CrossMoDa_'+str(count//10).zfill(3)+'_0000.nii.gz'))
        nib.save(expert_out, out_path.joinpath('expert_labels/CrossMoDa_'+str(count//10).zfill(3)+'.nii.gz'))
        nib.save(simple_consensus_out, out_path.joinpath('dp_consensus/CrossMoDa_'+str(count//10).zfill(3)+'.nii.gz'))
        nib.save(staple_consensus_out, out_path.joinpath('staple_consensus/CrossMoDa_'+str(count//10).zfill(3)+'.nii.gz'))
        nib.save(random_reg_out, out_path.joinpath('random_reg/CrossMoDa_'+str(count//10).zfill(3)+'.nii.gz'))

    count += 1
    one_hot_label = torch.nn.functional.one_hot(label.long(), num_classes=3).unsqueeze(0)
    one_hot_expert_label = torch.nn.functional.one_hot(expert_label.long(), num_classes=3).unsqueeze(0)
    one_hot_dp_label = torch.nn.functional.one_hot(dp_consensus.long(), num_classes=3).unsqueeze(0)
    one_hot_staple_label = torch.nn.functional.one_hot(staple_consensus.long(), num_classes=3).unsqueeze(0)
    one_hot_random_label = torch.nn.functional.one_hot(random_reg.long(), num_classes=3).unsqueeze(0)

    dice_expert = dice3d(one_hot_label, one_hot_expert_label, one_hot_torch_style=True)[0,1].item()
    dice_dp = dice3d(one_hot_label, one_hot_dp_label, one_hot_torch_style=True)[0,1].item()
    dice_staple = dice3d(one_hot_label, one_hot_staple_label, one_hot_torch_style=True)[0,1].item()
    dice_random = dice3d(one_hot_label, one_hot_random_label, one_hot_torch_style=True)[0,1].item()

    print('{}: expert: {}, dp: {}, staple:{}, random: {}'.format(file_id,dice_expert,dice_dp,dice_staple,dice_random))

count = 0
for img_file in all_data['val_image_paths'].values():
    label_file = img_file.replace('target_training_unlabeled','__omitted_labels_target_training__')
    label_file = label_file.replace('.nii.gz','_Label.nii.gz')
    label = torch.from_numpy(nib.load(label_file).get_fdata())
    org_img = nib.load(img_file)
    image = torch.from_numpy(org_img.get_fdata())

    label = label[:,:,45:95]
    image = image[:,:,45:95]

    label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='nearest').squeeze()
    image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0),scale_factor = 2, mode='trilinear').squeeze()

    img_out                 = nib.Nifti1Image(image             , org_img.affine, org_img.header)
    label_out               = nib.Nifti1Image(label      , org_img.affine, org_img.header)

    nib.save(img_out, out_path.joinpath('val_images/CrossMoDa_'+str(count).zfill(3)+'_0000.nii.gz'))
    nib.save(label_out, out_path.joinpath('val_labels/CrossMoDa_'+str(count).zfill(3)+'.nii.gz'))
    count += 1
