import os
import shutil
import re
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable

import paramiko
from scp import SCPClient

from enum import IntEnum, auto

import curriculum_deeplab.utils.constants as constants

class FileAction(IntEnum):
    COPY = auto()
    COPY_REMOTE = auto()
    MOVE = auto()
    LINK = auto()

class ProcessMode(IntEnum):
    FILES = auto()
    DIRS = auto()
    SUBDIRS = auto()

__LBL_TAGS = ['lbl', 'seg', 'label', 'gt']

def zip_with_scalar(iterable, scalar):
    return [(item, scalar) for item in iterable]

def process_data(mode: ProcessMode, act:FileAction, _input, target=None,
    file_regex=constants._FILE_REGEX_, **kwargs):
    """ Process directories or files with different file actions.
        ARGS:
            mode: ProcessMode (FILES, DIRS, SUBDIRS)
            act: FileAction (COPY, COPY_REMOTE, MOVE, LINK) or a callable function
                with signature 'action(_path)' with _path eiter accepting a filepath or directory path
                (depending on selected mode arg)
            _input: Path or list of paths to a dir or file (depending on which mode was selected)
            target: Path of directory where processed files will be stored or target paths with
                length of _input if ProcessMode is 'Files'. Can be None for custom file actions
                but needs to be set for standard file actions.
        KEYWORDS:
            file_regex: Regex filter for file names
        CONDITIONAL KEYWORDS:
            relative_link: bool, mandatory when MOVE action is specified
            host_name: str, mandatory when COPY_REMOTE action is specified
            user_name: str, mandatory when COPY_REMOTE action is specified
    """

    if isinstance(target, Iterable):
        if mode is not ProcessMode.FILES:
            raise(ValueError, "Multiple targets can only be passed for ProcessMode.FILES.")

        if not isinstance(_input, Iterable) or len(target) != len(_input):
            raise(ValueError, "Input and target iterables need to have same lenghts.")

    if act is FileAction.COPY_REMOTE and mode is not ProcessMode.DIRS:
        raise(ValueError, "Copying from remote machines requires source directories as input.")

    if act is FileAction.COPY_REMOTE and ('host_name' not in kwargs or 'user_name' not in kwargs):
        raise(ValueError, "Please provide host_name and user_name if copying from remote.")

    if isinstance(_input, os.PathLike) or isinstance(_input, str):
        _input = [_input]

    if act is FileAction.COPY_REMOTE:
        copy_remote = get_file_action(act, source_dirs=_input, **kwargs)
        copy_remote()
        return

    elif act in [FileAction.COPY, FileAction.LINK, FileAction.MOVE]:
        action = get_file_action(act, **kwargs)

    elif callable(act):
        action = act

    if mode == ProcessMode.DIRS:
        for f_dir in _input:
            if Path(f_dir).is_dir():
                src_files = os.listdir(f_dir)

                for file_name in src_files:
                    full_file_path = os.path.join(f_dir, file_name)
                    if os.path.isfile(full_file_path) and re.search(file_regex,full_file_path):
                        action(full_file_path)
            else:
                raise Warning(f"Directory '{f_dir}' does not exist. Skipping directory processing.")

    elif mode == ProcessMode.FILES:
        if isinstance(target, Iterable):
            zipped = zip(_input, target)
        else:
            zipped = zip_with_scalar(_input, target)

        for input_path, target_path in zipped:
            if Path(input_path).is_file():
                if re.search(file_regex, str(input_path)):
                    action(input_path, target_path)
            else:
                raise Warning(f"'{input_path}' is not present or not a file. Skipping file processing.")

    elif mode == ProcessMode.SUBDIRS:
        # Recall this function with unfolded directories
        process_data(ProcessMode.DIRS, act, unfold_directories(_input), target, file_regex, **kwargs)
    else:
        raise(ValueError)



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


def get_file_action(act: FileAction, **kwargs):

    if act == FileAction.COPY:
        return shutil.copy

    elif act == FileAction.COPY_REMOTE:

        if 'source_dirs' not in kwargs:
            raise(ValueError)

        def remote_copy_action():
            with paramiko.SSHClient() as ssh_client:
                ssh_client.load_system_host_keys()
                ssh_client.connect(kwargs['host_name'], username=kwargs['user_name'])
                scp_client = SCPClient(ssh_client.get_transport())
                for _dir in kwargs['source_dirs']:
                    scp_client.get(_dir, target_dir, recursive=True, preserve_times=True)

        return remote_copy_action

    elif act == FileAction.MOVE:
        # Move file to target dir
        def move_action(file_path, target):
            target_path = os.path.join(target, os.path.basename(file_path))
            if Path(target_path).exists():
                os.remove(target_path)
            shutil.move(file_path, target_path)

        return move_action

    elif act == FileAction.LINK:
        # Link file to target dir
        def link_action(file_path, target_path):
            if os.path.isdir(target_path):
                target_path = os.path.join(target, os.path.basename(file_path))
            # if Path(target_path).is_symlink():
            #     os.remove(target_path)
            if  kwargs.get('relative_link', False):
                # see https://stackoverflow.com/questions/54825010/why-does-os-symlink-uses-path-relative-to-destination
                rel_path_src = os.path.relpath(Path(file_path).resolve(), os.path.dirname(target_path))
                os.symlink(rel_path_src, target_path)
            else:
                os.symlink(
                    os.path.abspath(Path(file_path).resolve()),
                    os.path.abspath(target_path))

        return link_action

    else:
        raise(ValueError)

@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd
    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)



def get_file_id_dict(all_files, use_basename=True, only_first_numeric_id=True, join_ids=False, force_id_identifier=False,
    include_tags:list=None, exclude_tags=None):

    """Get file ids and paths based on regex
    """
    id_dict = {}
    exclude_tags = [] if not exclude_tags else exclude_tags
    include_tags = [] if not include_tags else include_tags

    for file_path in all_files:
        file_name = os.path.basename(file_path) if use_basename else file_path
        if not any([tag in file_name.lower() for tag in exclude_tags]):
            id = get_path_case_id(file_path, use_basename, only_first_numeric_id, join_ids, force_id_identifier,
                    include_tags)

            if not id: continue

            if id in id_dict:
                raise(ValueError(f"Same id '{id}' is already in dict. Ambigous ids detected. Please change id detection args."))
            id_dict[id] = file_path

    # If only single elements are in tuples reduce tuples
    only_singles = id_tuple_are_singles(id_dict.keys())

    id_dict = {postprocess_id_tuple(id, join_ids=join_ids, reduce_single_tuple=only_singles): _path
        for id, _path in id_dict.items()}

    return id_dict



# Img/label file specific operations
def get_image_id_dict(all_files, use_basename=True, only_first_numeric_id=True, join_ids=False, force_id_identifier=False,
    include_tags:list=None, exclude_tags=None):

    """Get image ids and paths based on regex
    """
    exclude_tags = exclude_tags + __LBL_TAGS if exclude_tags else __LBL_TAGS
    id_dict = get_file_id_dict(all_files, use_basename, only_first_numeric_id, join_ids, force_id_identifier,
        include_tags, exclude_tags)

    return id_dict



def get_label_id_dict(all_files, use_basename=True, only_first_numeric_id=True, join_ids=False, force_id_identifier=False,
    include_tags:list=None, exclude_tags=None):
    """Get label ids and paths based on regex
    """

    def remove_tags_from_tuple(tpl, tags):
        return tuple([tg for tg in tpl if tg not in tags])

    def remove_tags_from_string(_str, tags):
        for tg in tags:
            _str = _str.replace(tg, "")
        return _str

    include_tags = include_tags + __LBL_TAGS if include_tags else __LBL_TAGS
    id_dict = get_file_id_dict(all_files, use_basename, only_first_numeric_id, join_ids, force_id_identifier,
        include_tags, exclude_tags)

    # Make sure label tags are included
    id_dict = {ids: _path for ids, _path in id_dict.items() if set(ids).intersection(__LBL_TAGS)}

    id_dict_wo_lbl_tags = {}
    for ids, _path in id_dict.items():
        if isinstance(ids, tuple):
            id_dict_wo_lbl_tags[remove_tags_from_tuple(ids, __LBL_TAGS)] = _path
        elif isinstance(ids, str):
             id_dict_wo_lbl_tags[remove_tags_from_string(ids, __LBL_TAGS)] = _path
        else:
            id_dict_wo_lbl_tags[ids] = _path

    only_singles = id_tuple_are_singles(id_dict_wo_lbl_tags.keys())
    id_dict_wo_lbl_tags = {
        postprocess_id_tuple(id, join_ids=join_ids, reduce_single_tuple=only_singles): _path
            for id, _path in id_dict_wo_lbl_tags.items()
    }

    return id_dict_wo_lbl_tags



def postprocess_id_tuple(id_tuple, join_ids:bool=False, reduce_single_tuple=False):
    id_tuple = tuple(["".join([str(prt) for prt in id_tuple])]) if join_ids else id_tuple
    if reduce_single_tuple and len(id_tuple) == 1:
        id_tuple = id_tuple[0]
        id_tuple = int(id_tuple) if isinstance(id_tuple, str) and id_tuple.isnumeric() else id_tuple
    return id_tuple



def id_tuple_are_singles(id_tuples):
    return all([isinstance(tpl, (str, int)) or len(tpl) == 1 for tpl in id_tuples])



def get_path_case_id(file_path, use_basename=True, only_first_numeric_id= True, join_ids=False, force_id_identifier=False,
    include_tags:list=None):

    file_name = os.path.basename(file_path) if use_basename else file_path


    parts = ()
    all_parts = ()

    if force_id_identifier:
        explicit_id_match = re.match(constants._FORCED_ID_REGEX_, file_name)
        parts = [explicit_id_match.group('id')]
    else:
        fm_match = re.match(constants._FIXED_MOVING_REGEX_, file_name, re.VERBOSE)
        mul_groups = tuple(re.findall(constants._MULTIPLE_ID_REGEX_, file_name))
        if fm_match:
            parts = (int(fm_match.group('fixed_id')), int(fm_match.group('moving_id')))
        elif mul_groups:
            parts = mul_groups

    int_parts = tuple(int(id_string) for id_string in parts)
    if int_parts:
        int_parts = int_parts[0:1] if only_first_numeric_id else int_parts
        all_parts = all_parts + int_parts
    if include_tags:
        tag_parts = tuple(tg for tg in include_tags if tg in file_name.lower())
        if len(tag_parts) == 0:
            return tuple()
        all_parts = all_parts + tag_parts

    return postprocess_id_tuple(all_parts, join_ids=join_ids, reduce_single_tuple=False)



def lazy_dir_select(root: Path, lazy_keys):
    """Selects a dir path based on root directory and a keychain which
    needs to match the exact subpath only loosely.
    e.g.: /dir/to/root/stage_0000_idea/0001_test/leaf can be matched with:
        root = /dir/to/root
        lazy_keys = idea.0001.leaf

    Args:
        root (Path): The root path to start searching from
        lazy_keys (str): Lazy keys chained by dots

    Raises:
        ValueError if more than one match can be found.

    Returns:
        (Path): Exactly one matching path is returned.
    """
    lazy_keys = lazy_keys.split('.')
    depth = len(lazy_keys)
    key_regex = r".*?\/.*?".join(lazy_keys)
    key_regex = rf".*{key_regex}.*"
    unfolded = unfold_directories(root)
    unfolded = (Path(_path).relative_to(root) for _path in unfolded)
    unfolded = (_path for _path in unfolded if len(_path.parents) == depth)
    matched = [_path for _path in unfolded if re.match(key_regex, str(_path), flags=re.IGNORECASE)]

    if len(matched) != 1:
        raise(ValueError(f"Lazy dir selection must exactly fit one path. Got {len(matched)} matches."))

    return root.joinpath(matched[0]).resolve()

def get_experiment_root(_path):
    """Find the experiment root folder (indicated by .EXERIMENT_ROOT file)

    Args:
        _path (os.PathLike): The path from which the experiment root is searched (upwards)

    Returns:
        (Path): Dir path of experiment root
    """
    _path = Path(_path)
    if not _path.is_dir():
        _path = _path.parent
    while(len(_path).parts > 1):
        if '.EXPERIMENT_ROOT' in os.listdir(_path):
            return _path
        _path = _path.parent

    return None



def zip_dicts(*dicts, fillvalue=None):
    """Zip dicts and add fillvalue if key is not in both dicts

    Args:
        fillvalue (Any, optional): The fillvalue for non-existent values. Defaults to None.

    Returns:
        (dict): Zipped dict containg all keys of input dicts and a list of values with
            size n=num_of_passed_dicts
    """
    all_keys = {key for dc in dicts for key in dc.keys()}
    return {key: [dc.get(key, fillvalue) for dc in dicts] for key in all_keys}