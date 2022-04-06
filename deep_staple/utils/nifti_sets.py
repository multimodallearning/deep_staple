
import os
import re
import collections.abc
import nibabel
import ants
import numpy

_NIFTI_REGEX_ = r".*\.nii(\.gz)?$"



def unfold_directories(root_dir, file_regex=None):
    """Returns a list of directories if dir contains files matching the given regex.
    """
    root_dir = Path(root_dir)
    elem_list = root_dir.glob('**/*')
    elem_list = list(elem_list)
    elem_list.append(root_dir)

    # Filter for matching files
    if file_regex:
        elem_list = filter(
            lambda _path: _path.is_file() and re.match(file_regex, str(_path)),
            elem_list)
        elem_list = map(lambda _file: _file.parent, elem_list)
    else:
        elem_list = filter(
            lambda _path: _path.is_dir(), elem_list)

    # Return unique dir elements
    return set(elem_list)



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