import os, pathlib
import time
import numpy as np


def list_file_recursively(base_dir, depth=None):
    # traverse all if depth is None
    if depth == -1:
        return []

    relative_files_path = []

    for name in os.listdir(base_dir):
        abs_path = os.path.join(base_dir, name)
        if os.path.isfile(abs_path):
            relative_files_path.append(name)
            continue

        assert os.path.isdir(abs_path), abs_path

        sub_names = list_file_recursively(
            os.path.join(base_dir, name), depth=None if depth is None else depth - 1
        )

        relative_files_path += [os.path.join(name, n) for n in sub_names]

    return relative_files_path


def load_files_list(base_dir, files_list_path, with_extension=None, mkdir=False):
    # with_extension: change extension of files path, None to skip this part,
    # empty string to create a folder (without file extension)

    relative_files_path = np.loadtxt(files_list_path, dtype="str").tolist()

    if isinstance(relative_files_path, str):
        relative_files_path = [relative_files_path]

    if with_extension is not None:
        # change files' extension
        relative_files_path = [
            pathlib.Path(p).with_suffix(with_extension) for p in relative_files_path
        ]

    file_path = [os.path.join(base_dir, p) for p in relative_files_path]

    if mkdir:
        for p in file_path:

            if with_extension == "":
                # create folder
                pathlib.Path(p).mkdir(parents=True, exist_ok=True)
            else:
                # create parents folder
                pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

    return file_path


def normalized_np(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
