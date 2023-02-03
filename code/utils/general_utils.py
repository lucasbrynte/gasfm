import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from scipy.io import savemat
import shutil
from pyhocon import HOCONConverter,ConfigTree
import sys
import json
import os
import numpy as np
import random
import pandas as pd
from utils.path_utils import path_to_exp, path_to_predictions, path_to_tb_events, path_to_code_logs


global get_tb_writer
_tb_writer = None
def get_tb_writer(conf):
    global _tb_writer
    if _tb_writer is None:
        tb_path = path_to_tb_events(conf)
        _tb_writer = SummaryWriter(log_dir=tb_path)
    return _tb_writer


def log_code(conf):
    code_path = path_to_code_logs(conf)

    files_to_log = [
        'train.py',
        'single_scene_optimization.py',
        'multiple_scenes_learning.py',
        'evaluation.py',
        'loss_functions.py',
        'main.py',
    ]
    for file_name in files_to_log:
        shutil.copyfile('{}'.format(file_name), os.path.join(code_path, file_name))

    dirs_to_log = [
        'datasets',
        'models',
        'utils',
    ]
    for dir_name in dirs_to_log:
        shutil.copytree('{}'.format(dir_name), os.path.join(code_path, dir_name))

    # Print conf
    with open(os.path.join(code_path, 'exp.conf'), 'w') as conf_log_file:
        conf_log_file.write(HOCONConverter.convert(conf, 'hocon'))


def dump_predictions(conf, pred_dict, scene, phase, epoch=None, dump_npz=True, dump_mat=False, additional_identifiers=[]):
    path_cameras = path_to_predictions(conf, phase, epoch=epoch, scene=scene, additional_identifiers=additional_identifiers)
    if dump_npz:
        np.savez(path_cameras+'.npz', **pred_dict)
    if dump_mat:
        savemat(path_cameras+'.mat', pred_dict)


def write_results(conf, df, file_name="Results", additional_identifiers=[], append=False):
    exp_path = path_to_exp(conf)
    file_name = '_'.join([file_name] + additional_identifiers)
    results_file_path = os.path.join(exp_path, '{}.xlsx'.format(file_name))

    if append:
        assert df.index.name is not None

        if os.path.exists(results_file_path):
            prev_df = pd.read_excel(results_file_path).set_index(df.index.name)
            merged_err_df = pd.concat([prev_df, df], axis=0)
        else:
            merged_err_df = df

        merged_err_df.to_excel(results_file_path, na_rep='NULL')
    else:
        df.to_excel(results_file_path, na_rep='NULL')


def gen_dflt_exp_dir():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_error(err_string):
    print(err_string, file=sys.stderr)


def config_tree_to_string(config):
    config_dict={}
    for it in config.keys():
        if isinstance(config[it],ConfigTree):
            it_dict = {key:val for key,val in config[it].items()}
            config_dict[it]=it_dict
        else:
            config_dict[it] = config[it]
    return json.dumps(config_dict)


def get_additional_identifiers_for_outlier_injection(outlier_injection_rate):
    if outlier_injection_rate is None:
        return []
    else:
        return ['outlier_rate{:.2f}'.format(outlier_injection_rate)]


def any_model_parameters_on_gpu(model):
    return any(p.is_cuda for p in model.parameters())


def bmvm(bmats, bvecs):
    return torch.bmm(bmats, bvecs.unsqueeze(-1)).squeeze()


def mark_occurences_of_tensor_in_other(A, B):
    """
    Given 1D-tensors A & B, return a boolean tensor of the same shape as A, where each element indicates whether the corresponding element in A also exists anywhere in B.
    """
    assert A.ndim == 1
    assert B.ndim == 1

    A_df = pd.DataFrame({'idx': A.cpu()})
    A_df['val'] = False
    assert A_df.shape[0] == A.shape[0]
    B_df = pd.DataFrame({'idx': B.cpu()})
    B_df['val'] = True
    tmp = pd.concat([A_df, B_df])
    tmp = tmp.drop_duplicates(subset=['idx'], keep='last').set_index('idx')
    A_notin_B = tmp.loc[A_df['idx'], 'val'].values
    assert A_notin_B.shape[0] == A.shape[0]
    A_notin_B = torch.from_numpy(A_notin_B).to(A.device)
    assert A_notin_B.shape == A.shape

    # # Simple (memory-inefficient) implementation:
    # result_trivial = torch.any(A[:, None] == B[None, :], dim=1)
    # assert result_trivial.shape == A.shape
    # assert torch.all(A_notin_B == result_trivial)
    # # A_notin_B = result_trivial

    return A_notin_B

def nonzero_safe(A):
    """
    Given a pytorch tensor / numpy array A (typically a boolean mask), return a tuple of indices for each of the dimensions, pointing to the non-zero elements of the input tensor / array.
    That is, the functionality is the same as np.nonzero(A) or torch.nonzero(A, as_tuple=True).
    The only difference is that if A is a torch CPU tensor, the torch.nonzero() function is avoided, as the CPU-implementation appears to contain bugs related to memory management.
    """
    if isinstance(A, np.ndarray):
        return np.nonzero(A)
    else:
        if A.is_cuda:
            return torch.nonzero(A, as_tuple=True)
        else:
            return tuple(torch.from_numpy(x) for x in np.nonzero(A.detach().numpy()))


def shuffle_coo_matrix_along_axis_while_preserving_sparsity_pattern(values, indices, shuffle_axis=0):
    assert len(indices.shape) == 2
    assert indices.shape[0] == 2 # We expect a matrix
    assert shuffle_axis < 2
    nse = indices.shape[1] # Number of specified elements
    assert values.shape[0] == nse

    # One axis is shuffled, and the other one is sorted
    sort_axis = {0: 1, 1: 0}[shuffle_axis]

    # Verify > 1 index in every partition (corresponds to every point being visible in at least 2 cameras):
    _, unique_counts = np.unique(indices[sort_axis, :], return_counts=True)
    assert np.all(unique_counts > 1)

    # Make sure that we start out from an ordering which is sorted w.r.t. "sort_axis", i.e. that we have grouped / partitioned the elements.
    # Here we also reorder the values correspondingly, so there is no effective reordering.
    init_sort_idx = np.argsort(indices[sort_axis, :])
    indices = indices[:, init_sort_idx]
    values = values[init_sort_idx, ...]

    # Shuffle all indices completely randomly
    shuffle_idx = np.random.choice(nse, size=(nse,), replace=False)
    indices = indices[:, shuffle_idx]
    values = values[shuffle_idx, ...]

    # Again, sort these shuffled indices w.r.t. "sort_axis", such that we recover the same partitioning.
    final_sort_idx = np.argsort(indices[sort_axis, :])
    indices = indices[:, final_sort_idx]
    values = values[final_sort_idx, ...]

    # NOTE: So far, no effective reordering has been made, since indices and values have been jointly reordered.

    # Finally, apply a cyclic shift of each partition, this time without reordering both indices and values.
    # This leads to an effective random shuffling of each partition, conditioned on no indices being unchanged (with the exception of partitions of size 1).
    indices_shifted = np.roll(indices, 1, axis=1)
    start_idx_mask = indices[sort_axis, :] != indices_shifted[sort_axis, :]
    start_idx = np.nonzero(start_idx_mask)
    assert isinstance(start_idx, tuple) and len(start_idx) == 1
    start_idx = start_idx[0]
    prev_end_idx = np.mod(start_idx - 1, start_idx_mask.shape[0]) # Obtain end indices of the previous partition by subtracting 1 from the start indices (and apply np.mod() to remain within bounds).
    end_idx = np.roll(prev_end_idx, -1, axis=0) # Shift with "-1" to acquire end indices that match the start indices (corresponding to the same partition).

    # Verify that no indices are out of bounds:
    assert np.all(end_idx >= 0)
    assert np.all(end_idx < start_idx_mask.shape[0])

    # Verify no duplicates:
    _, unique_counts = np.unique(indices[sort_axis, start_idx], return_counts=True)
    assert np.all(unique_counts == 1)
    # assert np.all(np.unique(indices[sort_axis, start_idx], return_counts=True)[1] == 1)

    # Verify that all start & end indices are matching, i.e. that each such pair corresponds to the same partition:
    assert np.all(indices[sort_axis, start_idx] == indices[sort_axis, end_idx])

    # # Verify existence of start as well as end indices for each partition:
    # all_partition_ids_sorted = np.sort(np.unique(indices[sort_axis, :]))
    # assert np.all(np.sort(indices[sort_axis, start_idx]) == all_partition_ids_sorted)
    # assert np.all(np.sort(indices[sort_axis, end_idx]) == all_partition_ids_sorted)
    # # assert np.all(np.sort(indices[sort_axis, start_idx]) == np.sort(indices[sort_axis, end_idx]))

    # We are now prepared to perform the actual cyclic 1-shift:
    indices_shifted_per_partition = np.empty_like(indices)
    indices_shifted_per_partition[:, ~start_idx_mask] = indices_shifted[:, ~start_idx_mask]
    indices_shifted_per_partition[:, start_idx] = indices[:, end_idx]

    # Verify that the reordering does not result in any permutation across partitions:
    assert np.all(indices_shifted_per_partition[sort_axis, :] == indices[sort_axis, :])

    # Verify that the reordering results in every index being reordered within the partitions:
    assert np.all(indices_shifted_per_partition[shuffle_axis, :] != indices[shuffle_axis, :])

    indices = indices_shifted_per_partition

    # NOTE: It would be possible to consider multiple cyclic shifts, i.e. 1-shift, 2-shift, 3-shift, and so on,
    # in case we want to project each point into more than one camera.
    # Do however note that the only interesting shifts for a partition of size K would be k-shifts where k=1,...,K-1.

    return values, indices


def get_full_conf_vals(conf):
    # return a conf file as a dictionary as follow:
    # "key.key.key...key": value
    # Useful for the conf.put() command
    full_vals = {}
    for key, val in conf.items():
        if isinstance(val, dict):
            part_vals = get_full_conf_vals(val)
            for part_key, part_val in part_vals.items():
                full_vals[key + "." +part_key] = part_val
        else:
            full_vals[key] = val

    return full_vals

def detect_discrepancies_with_reference_configtree(ref_configtree, partial_configtree, ancestor_keys=[]):
    """
    Detect potential mismatches of the ConfigTree "partial_configtree", i.e. specified elements that are not present in "ref_configtree" (the keys should be consistent, but the values not necessarily the same).
    """
    assert isinstance(ref_configtree, ConfigTree)
    assert isinstance(partial_configtree, ConfigTree)
    mismatches = []
    for key, value in partial_configtree.items():
        assert isinstance(key, str)
        curr_keys = ancestor_keys + [key]
        if key not in ref_configtree:
            mismatches.append('.'.join(curr_keys))
            continue
        elif isinstance(ref_configtree[key], ConfigTree):
            if not isinstance(partial_configtree[key], ConfigTree):
                # Expected dict-like structure, but encountered something else
                mismatches.append('.'.join(curr_keys))
                continue
            # Recursive function call
            recursive_mismatches = detect_discrepancies_with_reference_configtree(
                ref_configtree[key],
                partial_configtree[key],
                ancestor_keys = curr_keys,
            )
            if len(recursive_mismatches) > 0:
                mismatches = mismatches + recursive_mismatches
                continue
        else:
            if isinstance(partial_configtree[key], ConfigTree):
                # Provided dict-like structure, but expected something else
                mismatches.append('.'.join(curr_keys))
                continue
    return mismatches

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

base_worker_rng = torch.Generator()
