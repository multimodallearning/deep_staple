import os
import time
import glob
import re
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from joblib import Memory

from deep_staple.utils.common_utils import DotDict
from deep_staple.hybrid_id_dataset import HybridIdDataset
from deep_staple.utils.torch_utils import ensure_dense, restore_sparsity


cache = Memory(location=os.environ['MMWHS_CACHE_PATH'])

class MMWHSDataset(HybridIdDataset):
    def __init__(self, *args, state='training',
        label_tags=(
            "background",
            "left_myocardium",
            "left_atrium",
            "left_ventricle",
            "right_atrium",
            "right_ventricle",
            "ascending_aorta",
            "pulmonary_artery"),
        **kwargs):
        self.state = state

        super().__init__(*args, state=state, label_tags=label_tags, **kwargs)

    def extract_3d_id(self, _input):
        # Match sth like 1001-HLA:mTS1
        items = re.findall(r'^(\d{4})-(HLA)(:m[A-Z0-9a-z]{3,4})?', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

    def extract_short_3d_id(self, _input):
        # Match sth like 1001-HLA:mTS1 and returns 1001-HLA
        items = re.findall(r'^(\d{4})-(HLA)', _input)[0]
        items = list(filter(None, items))
        return "-".join(items)

# @cache.cache(verbose=True)
def load_data(self_attributes: dict):
    # Use only specific attributes of a dict to have a cacheable function
    self = DotDict(self_attributes)

    IMAGE_ID = '_0000'
    t0 = time.time()

    if self.state.lower() == "training":
        img_directory = "Task651_MMWHS_MRI_HLA/imagesTr"
        label_directory = "Task651_MMWHS_MRI_HLA/labelsTr"

    elif self.state.lower() == "validation":
        img_directory = "Task651_MMWHS_MRI_HLA/imagesTs"
        label_directory = "Task651_MMWHS_MRI_HLA/labelsTs"

    else:
        raise Exception("Unknown data state. Choose either 'training or 'validation'")

    img_path = Path(self.base_dir, img_directory)
    label_path = Path(self.base_dir, label_directory)

    if self.crop_3d_region is not None:
        self.crop_3d_region = torch.from_numpy(self.crop_3d_region)

    # all_quality_labels = {}
    # with open(Path(data_path, "IQA.csv"), newline='') as csvfile:
    #     linereader = csv.reader(csvfile, delimiter=',')

    #     for sample_name, quality_label in list(linereader)[1:]:
    #         patient_id, secondary_id, cardiac_phase_id = re.findall(r'P(\d+)-(\d+)-([ESD]+)', sample_name)[0]
    #         patient_id, secondary_id = int(patient_id), int(secondary_id)
    #         mmwhs_id = f"{patient_id:03d}-{secondary_id:02d}-{cardiac_phase_id}"

    #         all_quality_labels[mmwhs_id] = int(quality_label) - 1 # We will get quality labels 0,1,2 not 1,2,3 as in the CMRxMotion guidelines

    files = sorted(list(img_path.glob("**/*.nii.gz")) + list(label_path.glob("**/*.nii.gz")))

    # First read filepaths
    img_paths = {}
    label_paths = {}

    if self.debug:
        files = files[:10]

    for _path in files:
        trailing_name = str(_path).split("/")[-1]
        # Extract ids from sth. like P001-1-ED-label.nii.gz
        patient_id, orient_id = re.findall(r'm(\d+)_(HLA).*?.nii.gz', trailing_name)[0]
        patient_id = int(patient_id)

        # Generate cmrxmotion id like 001-02-ES
        mmwhs_id = f"{patient_id:03d}-{orient_id}"

        if not IMAGE_ID in trailing_name:
            label_paths[mmwhs_id] = str(_path)
        else:
            img_paths[mmwhs_id] = str(_path)

    if self.ensure_labeled_pairs:
        pair_idxs = set(img_paths).intersection(set(label_paths))
        label_paths = {_id: _path for _id, _path in label_paths.items() if _id in pair_idxs}
        img_paths = {_id: _path for _id, _path in img_paths.items() if _id in pair_idxs}

    img_data_3d = {}
    label_data_3d = {}
    modified_label_data_3d = {}

    # Load data from files
    print(f"Loading MMWHS {self.state} images and labels...")
    id_paths_to_load = list(label_paths.items()) + list(img_paths.items())

    description = f"{len(img_paths)} images, {len(label_paths)} labels"

    for _3d_id, _file in tqdm(id_paths_to_load, desc=description):
        trailing_name = str(_file).split("/")[-1]
        tmp = torch.from_numpy(nib.load(_file).get_fdata()).squeeze()

        if not IMAGE_ID in trailing_name:
            resample_mode = 'nearest'
        else:
            resample_mode = 'trilinear'

        if self.do_resample:
            tmp = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=self.resample_size, mode=resample_mode).squeeze(0).squeeze(0)

            if tmp.shape != self.resample_size:
                difs = np.array(self.resample_size) - torch.tensor(tmp.shape)
                pad_before, pad_after = (difs/2).clamp(min=0).int(), (difs.int()-(difs/2).int()).clamp(min=0)
                tmp = F.pad(tmp, tuple(torch.stack([pad_before.flip(0), pad_after.flip(0)], dim=1).view(-1).tolist()))

        if self.crop_3d_region is not None:
            difs = self.crop_3d_region[:,1] - torch.tensor(tmp.shape)
            pad_before, pad_after = (difs/2).clamp(min=0).int(), (difs.int()-(difs/2).int()).clamp(min=0)
            tmp = F.pad(tmp, tuple(torch.stack([pad_before.flip(0), pad_after.flip(0)], dim=1).view(-1).tolist()))

            tmp = tmp[self.crop_3d_region[0,0]:self.crop_3d_region[0,1], :, :]
            tmp = tmp[:, self.crop_3d_region[1,0]:self.crop_3d_region[1,1], :]
            tmp = tmp[:, :, self.crop_3d_region[2,0]:self.crop_3d_region[2,1]]


        if not IMAGE_ID in trailing_name:
            label_data_3d[_3d_id] = tmp.long()
        else:
            if self.do_normalize: # Normalize image to zero mean and unit std
                tmp = (tmp - tmp.mean()) / tmp.std()

            img_data_3d[_3d_id] = tmp

    # Initialize 3d modified labels as unmodified labels
    for label_id in label_data_3d.keys():
        modified_label_data_3d[label_id] = label_data_3d[label_id]

    # Postprocessing of 3d volumes
    # None

    return dict(
        img_paths=img_paths,
        label_paths=label_paths,
        img_data_3d=img_data_3d,
        label_data_3d=label_data_3d,
        modified_label_data_3d=modified_label_data_3d,
    )