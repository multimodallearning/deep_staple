
import os
import re
import collections.abc
import nibabel
import ants
import numpy

from curriculum_deeplab.utils.data_management import unfold_directories

_NIFTI_REGEX_ = r".*\.nii(\.gz)?$"

def get_nifti_info_from_root(root_dir, sort_by_value, store_hash):
    """Reads nifti file headers returns a json object containg the info.
       Every .nii or .nii.gz file in the subtree of the root_dir will be read.
    """
    nifti_dirs = unfold_directories(root_dir, constants._NIFTI_REGEX_)

    info = {}
    for _dir in nifti_dirs:
        dir_info = get_nifti_dir_info_dict(_dir, sort_by_value=sort_by_value, store_hash=store_hash)
        if sort_by_value:
            def update_dict_leaves(dct, dct_w_upd):
                if not dct:
                    return dct_w_upd
                for key, val in dct_w_upd.items():
                    if isinstance(val, collections.abc.Mapping):
                        dct[key] = update_dict_leaves(dct.get(key, {}), val)
                    else:
                        path_list = dct.get(key,[])
                        path_list.extend(val)
                        dct[key] = path_list
                return dct
            info = update_dict_leaves(info, dir_info)
        else:
            info.update(dir_info)
    return info



def get_nifti_dir_info_dict(_dir, sort_by_value=False, store_hash=False):
    """Retrieves nifti info (header + other) per file in directory"""

    UN_VAL_THRESHH = 60

    all_nifti_paths = get_nifti_filepaths(_dir)
    if store_hash:
        raise NotImplementedError()
    else:
        hash_dict = None

    dir_info_dict = {}

    for nifti_path in all_nifti_paths:
        # current_fname = os.path.basename(nifti_path)
        nifti = nibabel.load(nifti_path)
        header = nifti.header
        nibabel_orientation = ''.join(nibabel.aff2axcodes(nifti.affine))

        # Do not read ants image directly. May result in SegFault when img is nifti2 image
        nifti1_img = nibabel.Nifti1Image(nifti.get_fdata(), nifti.affine, nifti.header)

        try:
            # Read ITK snap canonical orientation output
            ants_img = ants.utils.convert_nibabel.from_nibabel(nifti1_img)
            ants_orientation = ants_img.get_orientation()

        except RuntimeError:
            ants_orientation = 'N/A'
        # Add the RAS, AIL orientation as custom filed to header (merge dicts)
        orient_dict = {
            'nibabel_orientation': nibabel_orientation,
            'itk_orientation': ants_orientation
        }
        file_dir_info_dict = {**dict(header),**orient_dict}

        unique_vals, unique_counts = numpy.unique(numpy.array(nifti.dataobj), return_counts=True)

        if len(unique_vals < UN_VAL_THRESHH):
            un_vals_dict = {
                'unique_values': unique_vals,
                'unique_values_counts': unique_counts
            }
            file_dir_info_dict = {**file_dir_info_dict,**un_vals_dict}

        if store_hash:
            file_dir_info_dict['md5'] = hash_dict[nifti_path]

        if os.path.islink(nifti_path):
            file_dir_info_dict['is_symlink_to'] = os.readlink(nifti_path)

        if sort_by_value:
            for info_key, value in file_dir_info_dict.items():
                # Unify whitespaces to ' '
                value_as_string = ' '.join(str(value).split())
                field_dict = dir_info_dict.get(info_key, {})
                path_list = field_dict.get(value_as_string, [])
                path_list.append(nifti_path)
                field_dict[value_as_string] = path_list
                dir_info_dict[info_key] = field_dict
        else:
            dir_info_dict[nifti_path] = {
                # Unify whitespaces to ' '
                'dumped_info': ' '.join(str(file_dir_info_dict).split()),
            }

    return dir_info_dict



def get_nifti_filepaths(_dir, with_subdirs=False):
    all_nifti_paths = []

    if with_subdirs:
        for _unfolded in unfold_directories(_dir):
            all_nifti_paths.extend(get_nifti_filepaths(_unfolded, with_subdirs=False))
    else:
        all_nifti_paths = [os.path.join(_dir, filename) for filename in os.listdir(_dir)
        if re.match(_NIFTI_REGEX_, filename)]

    all_nifti_paths.sort()
    return all_nifti_paths