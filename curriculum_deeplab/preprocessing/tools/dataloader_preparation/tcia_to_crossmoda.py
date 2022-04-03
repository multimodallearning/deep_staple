import meidic_vtach_utils.datasets.data_management as dm
import meidic_vtach_utils.datasets.nifti_sets as nsets
from pathlib import Path
import shutil
from csv import DictReader

VOL_TERMS = [
    'Vol2016CISS', 'Vol1y',
    'Vol2.5', 'Vol2.5y',
    'Vol2y', 'Vol3y',
    'Vol2015', 'Vol2018',
    'VolT22y','Vol1',
    'Vol3m', 'Vol16',
    'Vol20mo', 'Vol2016',
    'Vol2017', 'Vol2014'
]

ADDITIONAL_WORDS = [
    'cochlea_ref',  'Cochlea_ref',
    'TV_ref','tv_ref', 'AN_ref',
    'an_ref', 'men_ref',
    'Men_ref', 'Modiolus_ref',
    'modiolus_ref', 'Cochlea_c_ref',
    'Cochlea_d_ref', 'tumour'] + vol

def get_cp_ren_closure(target_dir, new_name):
    def copy_closure(file_path):
        shutil.copy(file_path, Path(target_dir, new_name))

    return copy_closure

def main(argv):

    parser.add_argument("-i", "--input-folder", required=True)
    parser.add_argument("-o", "--output-folder", required=True)
    args = parser.parse_args(argv)

    nifti_paths = nsets.get_nifti_filepaths(
        args.input_folder,
        with_subdirs=True
    )
    img_dct = dm.get_image_id_dict(
        nifti_paths,
        use_basename=False,
        only_first_numeric_id=False,
        custom_id_list=['T1', 'T2'] + ADDITIONAL_WORDS)

    mangling_dcts = []
    mapping_file_path = Path(args.i + "mapping.csv")

    with open(mapping_file_path) as mapping_file:
        rdr = DictReader(mapping_file)
        for line in rdr:
            mangling_dcts.append(line)

    mangling_dcts = {int(entry['tcia']): int(entry['crossmoda']) for entry in mangling_dcts}

    for keys, _path in img_dct.items():
        if 'T1' in keys:
            modality = 'ceT1'
        elif 'T2' in keys:
            modality = 'hrT2'
        tcia_num = keys[2]

        lbl_types = [lbl_type.lower().rstrip('_ref').replace('.', '_') \
            for lbl_type in ADDITIONAL_WORDS if lbl_type in _path]
        lbl_string="".join(lbl_types)
        if lbl_string in ['tv', 'an', 'tumour']:
            lbl_string='_Label'
        elif lbl_string ==  '':
            pass
        else:
            lbl_string = '_'+lbl_string + '_additionalLabel'
        if tcia_num in mangling_dcts:
            new_fname = f"crossmoda_{mangling_dcts[tcia_num]}_{modality}{lbl_string}.nii.gz"
        else:
            new_fname = f"tcia_id_{tcia_num}_{modality}{lbl_string}.nii.gz"

        dm.process_data(
            dm.ProcessMode.FILES,
            get_cp_ren_closure(args.output_folder, new_fname),
            _path, args.output_folder
        )

if __name__ == "__main__":
    main(sys.argv[1:])
