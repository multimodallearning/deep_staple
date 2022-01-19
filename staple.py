# %%
# https://towardsdatascience.com/how-to-use-the-staple-algorithm-to-combine-multiple-image-segmentations-ce91ebeb451e
# packages
import nibabel as nib # https://nipy.org/nibabel/
import SimpleITK as sitk # https://simpleitk.org/
from matplotlib import pyplot as plt
import os, sys
from meidic_vtach_utils.run_on_recommended_cuda import get_cuda_environ_vars as get_vars
os.environ.update(get_vars(select="*"))
from collections import OrderedDict
import torch
import numpy as np

# %%
DATA_PATH = "/share/data_supergrover1/weihsbach/shared_data/important_data_artifacts/curriculum_deeplab/20220114_crossmoda_multiple_registrations/crossmoda_deeds_registered.pth"
bare_data = torch.load(DATA_PATH)

# %%
reg_state = "acummulate_deeds_FT2_MT1"

staple_filter = sitk.STAPLEImageFilter()
staple_filter.SetMaximumIterations(1000)
# sitk.ProcessObject.SetGlobalDefaultDebugOff()
FOREGROUND = 1.0
weight_data = {}
EVERY = 1
staple_filter.SetForegroundValue(FOREGROUND)
DEBUG = False
if reg_state == "acummulate_deeds_FT2_MT1":
    for fixed_id, moving_dict in bare_data.items():
        print('fixed_id', fixed_id)
        # print(moving_dict)
        sorted_moving_dict = OrderedDict(moving_dict)
        for slice_idx in range(45,95):
            print(len(weight_data))
            # if slice_idx == 62: continue
            moving_data = []
            selected_moving_ids = []

            for idx_mov, (moving_id, moving_sample) in enumerate(sorted_moving_dict.items()):
                if idx_mov % EVERY == 0:
                    moving_data.append(moving_sample['warped_label'].to_dense()[slice_idx].unsqueeze(-1).cpu())
                    moving_slice_id = f"{fixed_id}:m{moving_id}W{slice_idx-45:03d}"
                    selected_moving_ids.append(moving_slice_id)

            sitk_moving_data = [sitk.GetImageFromArray(reg_seg.numpy().astype(np.int16)) for reg_seg in moving_data]
            _ = staple_filter.Execute(sitk_moving_data)
            # staple_out = sitk.STAPLE(sitk_moving_data, FOREGROUND)
            # consensus = sitk.GetArrayFromImage(staple_out)

            specitivity = staple_filter.GetSpecificity()
            print("iters", staple_filter.GetElapsedIterations())
            sensitivity = staple_filter.GetSensitivity()
            # f_weight_dict = weight_data.get(fixed_id, {})
            # staple_consensus = sitk.GetArrayFromImage(staple_out)
            for moving_id, sens, spec in zip(selected_moving_ids, sensitivity, specitivity):
                weight_data[moving_id] = dict(sensitivity=sens, specitivity=spec)
            # weight_data[fixed_id] = f_weight_dict

        if DEBUG: break
else:
    raise ValueError()

weight_data['data_path'] = DATA_PATH
torch.save(weight_data, f"./data/staple_calc/{reg_state}_every_{EVERY}_2dslices.pth")
sys.exit(0)

# %%
dps = []
for key in weight_data.keys():
    if key == 'data_path': continue
    if np.isnan(weight_data[key]['sensitivity']):
        pass
    else:
        dps.append(weight_data[key]['sensitivity'])
plt.hist(dps)

# %%
network_scores = torch.load("/share/data_supergrover1/weihsbach/shared_data/tmp/curriculum_deeplab/data/output/noble-forest-1144_fold0_epx39/train_label_snapshot.pth")

# %%
d_ids = network_scores['d_ids']
network_parameter = network_scores['data_parameters']

# %%
set(d_ids) == set(weight_data.keys())

# %%
print(sorted(d_ids)[:10])
print(sorted(weight_data.keys())[:10])
bare_data.keys()



# %%
print(list(weight_data.keys()))

# %%
