import os, pathlib
import time
import numpy as np
import torch


def list_file_recursively(base_dir, depth=None):
    # traverse all if depth is None
    if depth == -1:
        return []

    relative_files_path = []

    for name in sorted(os.listdir(base_dir)):
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
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def get_similarity_func_id(metric_type):
    # comparing string in the search function is slow
    # supported_funcs = [
    #     "dot",
    #     "exp_dot",
    #     "taylor_exp_dot",
    #     "cosine",
    #     "exp_cosine",
    #     "taylor_exp_cosine",
    #     "inverse_L2",
    # ]
    
    # assert metric_type in supported_funcs

    # len of base_sims should be smaller than 8
    base_sims = ["dot", "cosine", "inverse_L2"]

    base_id = [metric_type.endswith(suf) for suf in base_sims].index(True)

    ext_prefixes =  ["exp_", "taylor_exp_", ""]
    ext_id = [metric_type.startswith(suf) for suf in ext_prefixes].index(True)

    return base_id + ext_id * 8


@torch.jit.script
def compute_similarity(A: torch.Tensor, B: torch.Tensor, metric_id: int = 16):
    # metric_id of 16 is 'dot'

    base_id = metric_id % 8
    ext_id = metric_id // 8

    if base_id == 0:
        # dot product
        sim = A @ B.T
    elif base_id == 1:
        # cosine
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.nn.functional.normalize(B, dim=-1)
        sim = A @ B.T
    else:
        # L2
        d = torch.cdist(A, B, p=2.0)
        sim = 1 / (1 + d)
    

    if ext_id == 0:
        # exp
        sim = torch.exp(sim)
    elif ext_id == 1:
        # taylor_exp
        sim = 1 + sim + sim**2 / 2
    
    return sim