import os
import re
import hashlib
from pathlib import Path
import json
from contextlib import contextmanager

from meidic_vtach_utils.datasets.data_management import set_directory

JSON_POSTFIX = ".hashed.json"

def hash_datadir(datadir):
    assert os.path.isdir(datadir), "Passed datadir is not a directory."

    dir_subelems = os.listdir(datadir)
    filepaths = [os.path.join(datadir, elem) for elem in dir_subelems \
                 if os.path.isfile(os.path.join(datadir, elem))]

    return hash_files(filepaths, key_type='basename')



def hash_files(filepaths, key_type='basename'):
    hash_dict = {}
    for path in filepaths:
        if key_type == 'abspath':
            file_key = os.path.abspath(path)
        elif key_type == 'basename':
            file_key = os.path.basename(path)
        elif key_type == 'passed_paths':
            file_key = path
        else:
            raise(NotImplementedError)

        with open(path, 'rb') as content:
            hash = hashlib.md5(content.read()).hexdigest()
        hash_dict[file_key] = hash

    return hash_dict



def get_dumpfiles(dir):
    assert os.path.isdir(dir), "Passed dir is not a valid directory."
    dir_subelems = os.listdir(dir)
    regex = re.compile(fr".*{JSON_POSTFIX}$")
    return list(filter(regex.match, dir_subelems))



@contextmanager
def hashed_source(source: Path):
    """Hash a file or directory to identify the source. Writes a source_hashed.json file

    Args:
        source (Path): The source to be hashed

    Yields:
        (Path): Path to the exact same source, so that it can be reused, like:
                with hashed_source(u_source) as h_source:
                    ...
    """
    try:
        with set_directory(source.parent):
            # Set active directory as parent directory so that hash.json
            # will describe dir or file in same location
            if os.path.isdir(source):
                hash_dict = hash_datadir(source)

            elif os.path.isfile(source):
                with open(source, 'rb') as content:
                    hash = hashlib.md5(content.read()).hexdigest()
                hash_dict = {source: hash}
            else:
                raise ValueError("Passed source is not a file or folder.")

            with open(f"{source.name}_dir{JSON_POSTFIX}", 'w') as dump_file:
                dump_file.write(json.dumps(hash_dict, indent=4, sort_keys=True))

        yield source

    finally:
        pass