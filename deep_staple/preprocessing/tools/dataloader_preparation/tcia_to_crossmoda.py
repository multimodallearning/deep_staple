import sys
import os
import argparse
from pathlib import Path
import shutil
from csv import DictReader
import nibabel as nib
import numpy as np
import re
from tqdm import tqdm

import deep_staple.utils.nifti_sets as nsets

THIS_SCIPTS_DIR = os.path.dirname(os.path.realpath(__file__))

OTHER_LABELS = [
    'vol2016CISS', 'vol1y',
    'vol2.5', 'vol2.5y',
    'vol2y', 'vol3y',
    'vol2015', 'vol2018',
    'volT22y','vol1',
    'vol3m', 'vol16',
    'vol20mo', 'vol2016',
    'vol2017', 'vol2014', 'volt22y',
    '6m', 'test', 't1+13',
    'brainstem', '1y', 'men_ref',
    'men_ref', 'modiolus_ref', 'cochlea_c_ref',
    'cochlea_d_ref',
]

TUMOUR_LABELS = [
    'tv_ref', 'an_ref', 'tv', 'tumour_ref'
]
COCHLEA_LABELS = [
    'cochlea_ref',
]
ADDITIONAL_WORDS = TUMOUR_LABELS + COCHLEA_LABELS + OTHER_LABELS

SOURCE_RANGE = range(1, 106) # ceT1
TARGET_TRAINING_RANGE = range(106, 211) # hrT2
TARGET_VALIDATION_RANGE = range(211, 243) # hrT2

SUBDIR = "L1_original"

def format_lbl_types(lst):
    return [elem.rstrip('_ref').replace('.', '_') for elem in lst]

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", required=False)
    parser.add_argument("-o", "--output-folder", required=False)
    args = parser.parse_args(argv)

    base_dir = Path(args.output_folder, SUBDIR)
    assert not base_dir.is_dir(), f"Output directory '{base_dir}' exists. Please remove it to continue."

    nifti_paths = nsets.get_nifti_filepaths(
        args.input_folder,
        with_subdirs=True
    )

    # Translate from TCIA id space to crossmoda space
    mapping_file_path = Path(THIS_SCIPTS_DIR, "crossmoda_tcia_mapping.csv")
    mangling_dcts = []

    with open(mapping_file_path) as mapping_file:
        rdr = DictReader(mapping_file)
        for line in rdr:
            mangling_dcts.append(line)

    mangling_dcts = {int(entry['TCIA']): int(entry['CrossMoDa']) for entry in mangling_dcts}
    id_regex = re.compile(r"vs_gk_([0-9]{1,3})")

    for _path in tqdm(nifti_paths):
        id_dir, basename = Path(_path).parts[-2:]

        if 'T1.nii.gz' in basename:
            modality = 'ceT1'
        elif 'T2.nii.gz' in basename:
            modality = 'hrT2'

        tcia_num = int(id_regex.match(id_dir).group(1))

        lbl_types = [lbl_type for lbl_type in ADDITIONAL_WORDS if lbl_type in basename.lower()]
        lbl_types = format_lbl_types(lbl_types)
        lbl_types = list(set(lbl_types))

        lbl_string = "".join(lbl_types)

        if lbl_string in format_lbl_types(TUMOUR_LABELS):
            lbl_out_string = '_Label'
        elif lbl_string in format_lbl_types(COCHLEA_LABELS):
            lbl_out_string = '_Label'
        elif lbl_string != '':
            lbl_out_string = '_' + lbl_string + '_additionalLabel'
        elif lbl_string == '':
            lbl_out_string = ''
        else:
            raise ValueError()

        if tcia_num in mangling_dcts:
            crossmoda_num = mangling_dcts[tcia_num]
            new_fname = f"crossmoda_{crossmoda_num}_{modality}{lbl_out_string}.nii.gz"
        else:
            new_fname = f"tcia_id_{tcia_num}_{modality}{lbl_out_string}.nii.gz"

        # L1_original
        # |- source_training_labeled/               # crossmoda_1_ceT1(_Label).nii.gz ... crossmoda_105_ceT1(_Label).nii.gz
        # |- target_training_unlabeled/             # crossmoda_106_hrT2.nii.gz ... crossmoda_210_hrT2.nii.gz
        # |- target_validation_unlabeled/           # crossmoda_211_hrT2.nii.gz ... crossmoda_242_hrT2.nii.gz
        # |- __omitted_labels_target_training__/    # crossmoda_106_hrT2_Label.nii.gz ... crossmoda_203_hrT2_Label.nii.gz
        # |- __omitted_labels_target_validation__/  # crossmoda_211_hrT2_Label.nii.gz ... crossmoda_242_hrT2_Label.nii.gz
        # |- __additional_data_source_domain__/     # crossmoda_106_ceT1(_Label).nii.gz ... crossmoda_242_ceT1(_Label).nii.gz and others
        # |- __additional_data_target_domain__/     # crossmoda_10_hrT2(_Label).nii.gz ... crossmoda_236_hrT2(_Label).nii.gz and others

        target_dir = base_dir

        if modality == 'ceT1' and crossmoda_num in SOURCE_RANGE and not '_additionalLabel' in lbl_out_string:
            target_dir = target_dir.joinpath("source_training_labeled")
        elif modality == 'ceT1':
            target_dir = target_dir.joinpath("__additional_data_source_domain__")
        elif modality == 'hrT2' and crossmoda_num in TARGET_TRAINING_RANGE and lbl_out_string == '':
            target_dir = target_dir.joinpath("target_training_unlabeled")
        elif modality == 'hrT2' and crossmoda_num in TARGET_VALIDATION_RANGE and lbl_out_string == '':
            target_dir = target_dir.joinpath("target_validation_unlabeled")
        elif modality == 'hrT2' and crossmoda_num in TARGET_TRAINING_RANGE and lbl_out_string == '_Label':
            target_dir = target_dir.joinpath("__omitted_labels_target_training__")
        elif modality == 'hrT2' and crossmoda_num in TARGET_VALIDATION_RANGE and lbl_out_string == '_Label':
            target_dir = target_dir.joinpath("__omitted_labels_target_validation__")
        elif modality == 'hrT2':
            target_dir = target_dir.joinpath("__additional_data_target_domain__")
        else:
            raise ValueError

        target_dir.mkdir(parents=True, exist_ok=True)
        new_file = target_dir.joinpath(new_fname)

        if lbl_string in format_lbl_types(TUMOUR_LABELS) \
            or lbl_string in format_lbl_types(COCHLEA_LABELS):
            if os.path.isfile(new_file):
                ni_existing_label = nib.load(new_file)
                existing_label_data = ni_existing_label.get_fdata()
            else:
                existing_label_data = None

            ni_new = nib.load(_path)
            new_label_data = ni_new.get_fdata()

            # Aggregate tumour and cochlea label
            if lbl_string in format_lbl_types(TUMOUR_LABELS):
                new_label_data *= 1
            if lbl_string in format_lbl_types(COCHLEA_LABELS):
                new_label_data *= 2

            if existing_label_data is not None:
                out_data = existing_label_data + new_label_data
            else:
                out_data = new_label_data
            nib.save(nib.Nifti1Image(out_data, affine=ni_new.affine), new_file)
        else:
            shutil.copy(_path, new_file)

if __name__ == "__main__":
    main(sys.argv[1:])
