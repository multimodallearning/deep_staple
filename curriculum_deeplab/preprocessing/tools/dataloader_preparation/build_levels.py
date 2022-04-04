import sys
import argparse

SUBDIR = "L4_fine_localized_crop"

if __name__ == "__main__":
    main(sys.argv[1:])

def main(argv):

    # L1_original
    # |- source_training_labeled/               # crossmoda_1_ceT1(_Label).nii.gz ... crossmoda_105_ceT1(_Label).nii.gz
    # |- target_training_unlabeled/             # crossmoda_106_hrT2.nii.gz ... crossmoda_210_hrT2.nii.gz
    # |- target_validation_unlabeled/           # crossmoda_211_hrT2.nii.gz ... crossmoda_242_hrT2.nii.gz
    # |- __omitted_labels_target_training__/    # crossmoda_106_hrT2_Label.nii.gz ... crossmoda_203_hrT2_Label.nii.gz
    # |- __omitted_labels_target_validation__/  # crossmoda_211_hrT2_Label.nii.gz ... crossmoda_242_hrT2_Label.nii.gz
    # |- __additional_data_source_domain__/     # crossmoda_106_ceT1(_Label).nii.gz ... crossmoda_242_ceT1(_Label).nii.gz and others
    # |- __additional_data_target_domain__/     # crossmoda_10_hrT2(_Label).nii.gz ... crossmoda_236_hrT2(_Label).nii.gz and others

    # L2_resampled_05mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/

    # L3_coarse_fixed_crop 128x128x192vox @ 0.5x0.5x0.5mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/

    # L4_fine_localized_crop 128x128x128vox @ 0.5x0.5x0.5mm
    # |- source_training_labeled/
    # |- target_training_unlabeled/
    # |- target_validation_unlabeled/
    # |- __omitted_labels_target_training__/
    # |- __omitted_labels_target_validation__/


    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", required=False)
    args = parser.parse_args(argv)

    base_dir = Path(args.output_folder, SUBDIR)
    assert not base_dir.is_dir(), f"Output directory '{base_dir}' exists. Please remove it to continue."
