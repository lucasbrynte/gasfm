import torch  # DO NOT REMOVE
import cv2  # DO NOT REMOVE

import utils.path_utils
from utils import geo_utils, general_utils, dataset_utils
import scipy.io as sio
import numpy as np
import os.path


def get_raw_data(scene, use_gt):
    """
    :return:
    M - Points Matrix (2mxn)
    Ns - Inversed Calibration matrix (Ks-1) (mx3x3)
    Ps_gt - GT projection matrices (mx3x4)
    NBs - Normzlize Bifocal Tensor (En) (3mx3m)
    triplets
    """

    # Init
    dataset_path_format = os.path.join(utils.path_utils.path_to_datasets(), 'Euclidean', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scene))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = np.linalg.inv(dataset['K_gt'])
    N33 = Ns[:, 2, 2][:, None, None]
    Ns /= N33 # Divide by N33 to ensure last row [0, 0, 1] (although generally the case, a small deviation in scale has been observed for e.g. the PantheonParis scene)
    Ps_gt /= np.linalg.det(Ns @ Ps_gt[:, :, :3])[:, None, None]**(1/3) # Likewise, ensure that P is scaled such that P=K*[R  t], where K=inv(N) has final row [0, 0, 1], and R is a rotation
    R_gt = Ns @ Ps_gt[:, :, :3]
    assert np.allclose(R_gt.swapaxes(1, 2) @ R_gt, np.eye(3)[None, :, :])

    if use_gt:
        M = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    return M, Ns, Ps_gt


def test_Ps_M(Ps, M, Ns):
    global_rep_err = geo_utils.calc_global_reprojection_error(Ps.numpy(), M.numpy(), Ns.numpy())
    print("Reprojection Error: Mean = {}, Max = {}".format(np.nanmean(global_rep_err), np.nanmax(global_rep_err)))


def test_euclidean_dataset(scene):
    dataset_path_format = os.path.join(utils.path_utils.path_to_datasets(), 'Euclidean', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scene))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    M_gt = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    print("Test Ps and M")
    test_Ps_M(Ps_gt, M, Ns)

    print("Test Ps and M_gt")
    test_Ps_M(Ps_gt, M_gt, Ns)


if __name__ == "__main__":
    scene = "Alcatraz Courtyard"
    test_euclidean_dataset(scene)