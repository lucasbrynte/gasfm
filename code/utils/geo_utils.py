import torch
import cv2
import numpy as np
import cvxpy as cp
from numpy.random._mt19937 import MT19937
from numpy.random import Generator
from utils import dataset_utils
from utils import sparse_utils
from utils import general_utils
from utils.general_utils import nonzero_safe
import dask.array as da


def compare_rotations(R1, R2):
    if isinstance(R1, np.ndarray):
        cos_err = (R1 @ np.transpose(R2, [0,2,1])) [:, np.arange(3), np.arange(3)]
        cos_err = (cos_err.sum(axis=-1) - 1) / 2
    else:
        cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return np.arccos(cos_err) * 180 / np.pi


def project_to_rot(m):
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    return torch.matmul(u, vt)


def generate_random_homography(n,seed=0,to_numpy=False,num_row_addition=5):
    rand_gen = Generator(MT19937(seed=seed))
    H=np.eye(n)
    for i in range(num_row_addition):
        constants = rand_gen.uniform(0.9, 1.1, (n, 1))*rand_gen.choice([-1,1],size=(n,1))
        H *= constants
        row1,row2=rand_gen.choice(a=np.arange(n),size=(2,),replace=False)
        H[row1]+=H[row2]
    if to_numpy:
        return H
    else:
        return torch.from_numpy(H)

def tranlsation_rotation_errors(R_fixed, t_fixed, gt_Rs, gt_ts):
    R_error = compare_rotations(R_fixed, gt_Rs)
    t_error = np.linalg.norm(t_fixed - gt_ts, axis=-1)
    return R_error, t_error



def align_cameras(pred_Rs, gt_Rs, pred_ts, gt_ts, return_alignment=False):
    # NOTE! All "t" vectors are in fact camera centers, C = -R^T * t.
    '''

    :param pred_Rs: torch double - n x 3 x 3 predicted camera rotation
    :param gt_Rs: torch double - n x 3 x 3 camera ground truth rotation
    :param pred_ts: torch double - n x 3 predicted translation
    :param gt_ts: torch double - n x 3 ground truth translation
    :return:
    '''
    # find rotation
    d = 3
    n = pred_Rs.shape[0]

    pred_Rs_orig = pred_Rs.copy()
    pred_ts_orig = pred_ts.copy()

    # Find "average" relative rotation between GT & pred (not necessarily itself a rotation):
    Q = np.sum(gt_Rs @ np.transpose(pred_Rs, [0,2,1]), axis=0) #sum over the n views of R_gt[i] @ R_pred[i].T
    try:
        Uq, _, Vqh = np.linalg.svd(Q)
    except np.linalg.LinAlgError as e:
        print('[WARNING] Camera alignment failed at SVD. Simply return predicted poses as-is.')
        print(repr(e))
        if return_alignment:
            similarity_mat = np.eye(4)
            return pred_Rs_orig, pred_ts_orig, similarity_mat
        else:
            return pred_Rs_orig, pred_ts_orig

    # Normalize singular values to acquire an orthogonal matrix:
    sv = np.ones(3)
    # Finally, apply sign-flip on final singular value, such that we acquire a rotation (with determinant 1):
    sv[-1] = np.linalg.det(Uq @ Vqh)
    R_opt = Uq @ np.diag(sv) @ Vqh

    R_fixed = R_opt.reshape([1,3,3]) @ pred_Rs

    # find translation
    pred_ts = pred_ts @ R_opt.T  # Apply the optimal rotation on all the translations
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    # Find optimal relative translation (t_opt), as well as a scale correction (c_opt) by solving an optimization problem numerically.
    # The cost function appears to be the sum of normed residuals of the "t" vectors (which are in fact (somewhat misleading) the camera centers, and thus all in the same coordinate system).
    # The norm is defined by cvxpy.norm, and is probably Euclidean.
    # If it were squared, we would simply have an unconstrained QP, which should alternatively be possible to solve analytically.
    constraints = []
    obj = cp.Minimize(
        cp.sum(cp.norm(gt_ts - (c_opt * pred_ts + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    # obj = cp.Minimize(cp.sum(cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + t_opt_rep), axis=1)))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
        if not prob.status == "optimal":
            raise cp.error.SolverError("Status: \"{}\" encountered during alignment between predicted & GT cameras".format(prob.status))
    except cp.error.SolverError as e:
        print('[WARNING] Camera alignment failed at optimization. Simply return predicted poses as-is.')
        print(repr(e))
        if return_alignment:
            similarity_mat = np.eye(4)
            return pred_Rs_orig, pred_ts_orig, similarity_mat
        else:
            return pred_Rs_orig, pred_ts_orig
    t_fixed = c_opt.value * pred_ts + t_opt.value.reshape([1,3])

    if return_alignment:
        similarity_mat = np.eye(4)
        similarity_mat[0:3, 0:3] = c_opt.value * R_opt
        similarity_mat[0:3, 3] = t_opt.value
        return R_fixed, t_fixed, similarity_mat
    else:
        return R_fixed, t_fixed


def invert_euclidean_trafo(Rs, ts):
    """
    Given a batch of Euclidean transformations, Rs and ts, such that X -> Rs[i]*X + ts[i],
    return the corresponding Rs & ts for the inverse transformation, i.e. such that the above
    formula with the new Rs and ts now maps in the opposite direction.
    """
    assert len(Rs.shape) == 3
    assert Rs.shape[1] == 3
    assert Rs.shape[2] == 3
    assert len(ts.shape) == 2
    assert ts.shape[1] == 3
    if isinstance(Rs, np.ndarray):
        Rs_inv = np.transpose(Rs, [0,2,1]) # Rs_inv = Rs.T
        ts_inv = (-Rs_inv @ ts.reshape([-1, 3, 1])).squeeze() # ts_inv = -Rs.T @ ts = -Rs_inv @ ts
    else:
        Rs_inv = Rs.transpose(1, 2) # Rs_inv = Rs.T
        ts_inv = torch.bmm(-Rs_inv, ts.unsqueeze(-1)).squeeze() # ts_inv = -Rs.T @ ts = -Rs_inv @ ts
    return Rs_inv, ts_inv


def decompose_camera_matrix(Ps, Ks=None, inverse_direction_camera2global=True):
    """
    Given camera matrices Ps, first normalize with the inverse of a given calibration matrix (if not provided, assumes camera calibrated already).
    Next extract R & t components from the P_norm = [R t]
    Finally returns Rs=R.T as well as camera centers, determined by ts = -R.T @ t.
    """
    if isinstance(Ps, np.ndarray):
        Rt = np.linalg.inv(Ks) @ Ps if Ks is not None else Ps
        Rs = Rt[:, 0:3, 0:3]
        ts = Rt[:, 0:3, 3]
    else:
        n_cams = Ps.shape[0]
        if Ks is None:
            Ks = torch.eye(3, device=Ps.device).expand((n_cams, 3, 3))

        Rt = torch.bmm(Ks.inverse(), Ps)
        Rs = Rt[:, 0:3, 0:3]
        ts = Rt[:, 0:3, 3]

    if inverse_direction_camera2global:
        Rs, ts = invert_euclidean_trafo(Rs, ts)

    return Rs, ts


def decompose_projection_matrix(Ps):
    Vs = Ps[:, 0:3, 0:3].inverse().transpose(1, 2)
    ts = torch.bmm(-Vs.transpose(1, 2), Ps[:, 0:3, 3].unsqueeze(-1)).squeeze()
    return Vs, ts


def decompose_essential_matrix(E, x1, x2):
    # [R1,t], [R1,−t], [R2,t], [R2,−t]
    if x1.shape[0] == 3:
        x1 = x1 / x1[2]
        x1 = x1[:2]
        x2 = x2 / x2[2]
        x2 = x2[:2]
    R1, R2, t = cv2.decomposeEssentialMat(E=E)
    pose_options = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
    P2_options = [np.concatenate(pose_options[i], axis=1) for i in range(4)]
    P1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
    number_of_points_in_front = np.zeros([4, 2])
    for i, P2 in enumerate(P2_options):
        X = pflat(cv2.triangulatePoints(P1, P2, x1, x2))
        proj1 = np.dot(P1, X)
        proj2 = np.dot(P2, X)
        number_of_points_in_front[i, 0] = np.sum(proj1[-1] > 0)
        number_of_points_in_front[i, 1] = np.sum(proj2[-1] > 0)
    best_option = int(np.argmax(np.sum(number_of_points_in_front, axis=1)))
    return pose_options[best_option]


def M_to_xs(M):
    """
    reshapes the 2d points
    :param M: [2*m, n]
    :return: xs [m,n,2]
    """
    m,n = M.shape
    m = m//2
    xs = M.reshape([m,2,n])
    if isinstance(M, np.ndarray):
        xs = np.transpose(xs, [0,2,1])
    else:
        xs = xs.transpose(1,2)
    return xs


def xs_to_M(xs):
    n,m,_ = xs.shape
    if isinstance(xs, np.ndarray):
        xs_tran = np.transpose(xs, [0,2,1])
    else:
        xs_tran = xs.transpose(1,2)
    M = xs_tran.reshape([2*n, m])
    return M


def get_V_from_RK(Rs,Ks):
    return torch.inverse(Ks).permute(0,2,1) @ Rs.permute(0,2,1)

def get_cross_product_matrix(t):
    T = torch.zeros((3, 3), device=t.device)
    T[0, 1] = -t[2]
    T[0, 2] = t[1]
    T[1, 0] = t[2]
    T[1, 2] = -t[0]
    T[2, 0] = -t[1]
    T[2, 1] = t[0]
    return T


def batch_get_cross_product_matrix(t):
    batch_size = t.shape[0]
    T = torch.zeros((batch_size, 3, 3), device=t.device)
    T[:, 0, 1] = -t[:, 2]
    T[:, 0, 2] = t[:, 1]
    T[:, 1, 0] = t[:, 2]
    T[:, 1, 2] = -t[:, 0]
    T[:, 2, 0] = -t[:, 1]
    T[:, 2, 1] = t[:, 0]
    return T


def get_essential_matrix(Ri, Rj, ti, tj):
    Ti = get_cross_product_matrix(ti)
    Tj = get_cross_product_matrix(tj)

    return Ri.T @ (Ti - Tj) @ Rj

def batch_get_bifocal_tensors(Rs, ts):
    n = len(Rs)
    E = torch.zeros([n, n, 3, 3])
    for i in range(n):
        for j in range(n):
            E[i, j] = Rs[i].T @ get_cross_product_matrix(ts[i] - ts[j]) @ Rs[j]
    return E

def batch_fundamental_from_essential(E, Ks):
    n = len(Ks)
    F = torch.zeros_like(E)
    for i in range(n):
        for j in range(n):
            F[i,j] = torch.inverse(Ks[i]).T @ E[i,j] @ torch.inverse(Ks[j])
    return F

def get_fundamental_matrix(Ri, Rj, ti, tj, Ki, Kj):
    Eij = get_essential_matrix(Ri, Rj, ti, tj)
    Fij = torch.inverse(Ki).T @ Eij @ torch.inverse(Kj)
    return Fij


def get_fundamental_from_V_t(Vi, Vj, ti, tj):
    Ti = get_cross_product_matrix(ti)
    Tj = get_cross_product_matrix(tj)
    return Vi @ (Ti - Tj) @ Vj.T


def batch_get_fundamental_from_V_t(Vi, Vj, ti, tj):
    Ti = batch_get_cross_product_matrix(ti)
    Tj = batch_get_cross_product_matrix(tj)
    return torch.bmm(Vi, torch.bmm((Ti - Tj),  Vj.transpose(1, 2)))


def get_camera_matrix(R, t, K):
    """
    Get the camera matrix as described in paper
    :param R: Orientation Matrix
    :param t: Camera Position   
    :param K: Intrinsic parameters
    :return: Camera matrix
    """
    if isinstance(R, np.ndarray):
        return K @ R.T @ np.concatenate((np.eye(3), -t.reshape(3, 1)), axis=1)
    else:
        return K @ R.T @ torch.cat((torch.eye(3), -t.view(3, 1)), dim=1)

def batch_get_camera_matrix_from_rtk(Rs, ts, Ks):
    n = len(Rs)
    if isinstance(Rs, np.ndarray):
        Ps = np.zeros([n,3,4])
    else:
        Ps = torch.zeros([n, 3, 4])
    for i,r,t,k in zip(np.arange(n),Rs,ts,Ks):
        Ps[i] = get_camera_matrix(r,t,k)
    return Ps

def get_camera_matrix_from_Vt(V, t):
    """
    Get the camera matrix as described in paper
    :param V: inv(K).T @ R.T Orientation Matrix
    :param t: Camera Position
    :return: Camera matrix
    """
    return torch.inverse(V).T @ torch.cat((torch.eye(3), -t), dim=1)


def batch_get_camera_matrix_from_Vt(Vs, ts):
    vT_inv = torch.inverse(Vs).transpose(1, 2)
    return torch.cat((vT_inv, torch.bmm(vT_inv, -ts.unsqueeze(-1))), dim=2)


def pflat(x):
    return x / x[-1, :]

def batch_pflat(x):
    return x / x[:, 2:3, :]

def correct_matches(P1, P2, pts_img1, pts_img2):
    pts3D = cv2.triangulatePoints(P1, P2, pts_img1[0:2, :], pts_img2[0:2, :])
    pts_img1 = torch.from_numpy(pflat(P1 @ pts3D)).float()
    pts_img2 = torch.from_numpy(pflat(P2 @ pts3D)).float()
    return pts_img1, pts_img2


def calc_pFp(Fij, pi, pj):
    tmp = Fij @ pj
    return torch.abs(torch.sum(pi * tmp, dim=0))


def batch_calc_pFp(Fij, pi, pj):
    return (pi * torch.bmm(Fij, pj)).sum(dim=1).abs()


def calc_reprojection_error(P1, P2, pts_img1, pts_img2):
    coorected_pts_img1, coorected_pts_img2 = correct_matches(P1.numpy(), P2.numpy(), pts_img1.numpy(), pts_img2.numpy())
    reproj_err1 = torch.norm(pflat(coorected_pts_img1)[0:2, :] - pflat(pts_img1)[0:2, :], dim=0, p=2)
    reproj_err2 = torch.norm(pflat(coorected_pts_img2)[0:2, :] - pflat(pts_img2)[0:2, :], dim=0, p=2)
    return reproj_err1, reproj_err2


def calc_global_reprojection_error(Ps, M, Ns):
    n = len(Ps)
    valid_pts = dataset_utils.get_M_valid_points(M)
    X = n_view_triangulation(Ps, M, Ns)
    projected_pts = batch_pflat(Ps @ X)[:, 0:2, :]
    image_points = M.reshape(n, 2, M.shape[-1])
    reproj_err = np.linalg.norm(image_points - projected_pts, axis=1)
    return np.where(valid_pts, reproj_err, np.full(reproj_err.shape, np.nan))


def reprojection_error_with_points(Ps, Xs, xs, visible_points=None):
    """
    :param Ps: [m,3,4]
    :param Xs: [n,3] or [n,4]
    :param xs: [m,n,2]
    :return: errors [m,n]
    """
    m,n,d = xs.shape
    _, D = Xs.shape
    X4 = np.concatenate([Xs, np.ones([n,1])], axis=1) if D == 3 else Xs

    if visible_points is None:
        visible_points = xs_valid_points(xs)

    projected_points = Ps @ X4.T  # [m,3,4] @ [4,n] -> [m,3,n]
    projected_points = projected_points.swapaxes(1, 2)
    visible_points_idx = nonzero_safe(visible_points)
    projected_points[visible_points_idx[0], visible_points_idx[1], :] = projected_points[visible_points_idx[0], visible_points_idx[1], :] / projected_points[visible_points_idx[0], visible_points_idx[1], -1][:, None]
    errors = np.linalg.norm(xs[:,:,:2] - projected_points[:,:,:2], axis=2)
    errors[~visible_points] = np.nan
    return errors

def reprojection_error_backproj_random_view_pairs(Ks, Ps, depths, xs, visible_points=None, calc_reproj_depths=False):
    """
    Given predicted depths and backprojected points in a number of given views, calculate an average "2-view reprojection error".
    For each point in each view, the backprojected point is reprojected in another (random) view in which it is also visible, yielding one residual.
    The given views may be either estimated or true cameras.

    :param Ks: [m,3,3]
    :param Ps: [m,3,4]
    :param depths: [m,n]
    :param xs: [m,n,2]
    :return: errors [m,n]
    """
    m, n, d = xs.shape
    assert d == 2
    assert Ks.shape == (m, 3, 3)
    assert Ps.shape == (m, 3, 4)
    assert depths.shape == (m, n)

    if visible_points is None:
        visible_points = xs_valid_points(xs)
    assert visible_points.shape == (m, n)

    # Decompose the camera matrices P=K*[R  t] and retrieve R_inv=R.T and t_inv = -R.T*t
    # Rs, ts = decompose_camera_matrix(Ps, Ks, inverse_direction_camera2global=False)
    # assert np.allclose(Ps, Ks @ np.concatenate([Rs, ts[:, :, None]], axis=2)) # Verify the identity P=K*[R  t]
    Rs_inv, ts_inv = decompose_camera_matrix(Ps, Ks, inverse_direction_camera2global=True)
    # assert np.allclose(np.linalg.norm(Rs, axis=2), np.ones((1, 1))) # Verify unit rows of R
    # assert np.allclose(np.linalg.norm(Rs_inv, axis=2), np.ones((1, 1))) # Verify unit rows of R_inv
    # assert np.allclose(Rs_inv, Rs.swapaxes(1, 2)) # Verify R_inv=R^T
    # assert np.allclose(ts_inv[:, :, None], -Rs.swapaxes(1, 2) @ ts[:, :, None]) # Verify t_inv=-R^T*t
    assert Rs_inv.shape == (m, 3, 3)
    assert ts_inv.shape == (m, 3)

    # Before backprojection, normalize the image points
    xs_hom = np.concatenate([xs, np.ones((m, n, 1))], axis=2)
    x_norm_hom = (np.linalg.inv(Ks) @ xs_hom.swapaxes(1, 2)).swapaxes(1, 2)
    x_norm = x_norm_hom[:, :, :-1] / x_norm_hom[:, :, [-1]]
    assert x_norm.shape == (m, n, d)

    # Build an array X4_local, containing the backprojected points in all views
    X4_local = np.ones([m, n, 3])
    X4_local[:, :, :2] = x_norm # First 2 coordinates determined by xs, after normalization
    X4_local *= depths[:, :, None]

    # Now transform the backprojected points to the global coordinate system:
    X4_global = ((Rs_inv @ X4_local.swapaxes(1, 2)) + ts_inv[:, :, None]).swapaxes(1, 2)

    # Next, before we do a batch matric multiplication where the initial dimension (the m-dimension) constitutes the batch,
    # we determine which backprojected point should be projected into which view. If skipping this step, we would simply project
    # all the points back into the cameras they were backprojected from, and there would be no error.

    # For simplicity and computational efficiency, we project each backprojected point into exactly one other camera (just not the same camera from which we backprojected).
    # Determine current indices and and values at the specified elements:
    X4_global_indices = np.array(np.nonzero(visible_points))
    X4_global_values = X4_global[X4_global_indices[0, :], X4_global_indices[1, :], :]
    # Carry out the permutation:
    X4_global_values, X4_global_indices = general_utils.shuffle_coo_matrix_along_axis_while_preserving_sparsity_pattern(X4_global_values, X4_global_indices)
    # Write back the values at the new (same but reshuffled) positions:
    X4_global[X4_global_indices[0, :], X4_global_indices[1, :], :] = X4_global_values

    X4_global_hom = np.concatenate([X4_global, np.ones((m, n, 1))], axis=2)
    projected_points = Ps @ X4_global_hom.swapaxes(1, 2) # [m,3,4] @ [m,4,n] -> [m,3,n]
    if calc_reproj_depths:
        reproj_depths = (np.linalg.inv(Ks) @ projected_points)[:, 2, :] # [m,3,n] -> [m,3,n] -> [m,n]
    projected_points = projected_points.swapaxes(1, 2)
    visible_points_idx = nonzero_safe(visible_points)
    projected_points[visible_points_idx[0], visible_points_idx[1], :] = projected_points[visible_points_idx[0], visible_points_idx[1], :] / projected_points[visible_points_idx[0], visible_points_idx[1], -1][:, None]
    errors = np.linalg.norm(xs[:,:,:2] - projected_points[:,:,:2], axis=2)
    errors[~visible_points] = np.nan
    if calc_reproj_depths:
        return errors, reproj_depths
    return errors

def get_points_in_view(M, img_idx, X=None):
    points_indices = np.logical_or(M[2 * img_idx, :] != 0, M[2 * img_idx + 1, :] != 0)
    if X is not None:
        points_indices = np.logical_and(points_indices, ~np.isnan(X[0, :]))

    return np.where(points_indices)[0]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def get_normalization_matrix(pts):
    if isinstance(pts, np.ndarray):
        norm_mat = np.eye(3)
        m = np.mean(pts[:2, :], axis=1)
        s = 1. / np.std(pts[:2, :], axis=1)
        norm_mat[0, 0] = s[0]
        norm_mat[1, 1] = s[1]
        norm_mat[:2, 2] = -s * m
    else:
        pts = pts.unique(dim=1)
        norm_mat = torch.eye(3)
        m = torch.mean(pts[0:2, :], dim=1)
        s = 1. / torch.std(pts[0:2, :], dim=1)
        norm_mat[0, 0] = s[0]
        norm_mat[1, 1] = s[1]
        norm_mat[0:2, 2] = -s * m
    return norm_mat


def batch_get_normalization_matrices(xs):
    """
    Given all the observed points return normalization matrices such thet N[i] takes the
    points at x[i] and normalizes them to have a zero mean and 1 std across both
    he x axis and the y axis.
    :param xs: np.ndarray [m,n,2] or [m,n,3]
    :return:  np.ndarray Ns [m,3,3]
    """
    m, n, d = xs.shape
    Ns = np.zeros([m, 3, 3])
    for i in range(m):
        Ns[i] = get_normalization_matrix(xs[i].T)  # xs[i].T is [3,n]
    return Ns


def normalize_points(points, Ns):
    # points is nXnX2
    n=len(points)
    norm_points=np.zeros(points.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            if np.sum(points[i,j,0]):
                norm_points[i,j,0]=np.dot(Ns[i],points[i,j,0])
                norm_points[i, j, 1] = np.dot(Ns[j], points[i, j, 1])
    return norm_points


def normalize_points_cams(Ps, xs, Ns):
    """
    Normalize the points and the cameras using the matrices in N.
    if :
    xs[i,j] ~ P[i] @ X[j]
    than so is:
     N[i] @ xs[i,j] ~ N[i] @ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs:  [m,n,2] or [m,n,3]
    :param Ns:  [m,3,3]
    :return:  norm_P, norm_x
    """
    assert isinstance(xs, np.ndarray)
    m, n, d = xs.shape
    xs_3 = np.concatenate([xs, np.ones([m, n, 1])], axis=2) if d == 2 else xs
    norm_P = np.zeros_like(Ps)
    norm_x = np.zeros_like(xs)
    for i in range(m):
        norm_P[i] = Ns[i] @ Ps[i]  # [3,3] @ [3,4]
        norm_points = (Ns[i] @ xs_3[i].T).T  # ([3,3] @ [3,n]) -> [n,3]
        norm_points[norm_points[:,-1]==0,-1] = 1
        norm_points = norm_points / norm_points[:, -1].reshape([-1,1])
        if d == 2:
            norm_x[i] = norm_points[:,:2]
    return norm_P, norm_x


def normalize_bifocal_mat(bifocalMat_ij, Ni, Nj):
    # pj_norm = N @ pj
    # F_norm = Ni^(-T) @ Fij @ Nj^(-1)
    return torch.inverse(Ni).T @ bifocalMat_ij @ torch.inverse(Nj)


def get_fundamental_from_P(P1,P2):
    V1,t1=decompose_projection_matrix(torch.unsqueeze(P1,0))
    V2, t2 = decompose_projection_matrix(torch.unsqueeze(P2,0))
    F12=get_fundamental_from_V_t(V1.squeeze(),V2.squeeze(),t1,t2)
    return F12

def sed_from_P_x(P1,P2,x1,x2):
    F12=get_fundamental_from_P(P1,P2)
    return symmetric_epipolar_distance(F12, x1, x2)


def symmetric_epipolar_distance(Fij, pts_i, pts_j):
    # sym_epi_dist = geo_utils.calc_pFp(Fij, pts_imgi, pts_imgj) * (1 / torch.norm(Fij @ pts_imgj, p=2, dim=0) + 1 / torch.norm(Fij.T @ pts_imgi, p=2, dim=0))
    return calc_pFp(Fij, pts_i, pts_j) * \
           (1 / torch.norm((Fij @ pts_j)[0:2, :], p=2, dim=0) + 1 / torch.norm((Fij.T @ pts_i)[0:2, :], p=2, dim=0))


def batch_symmetric_epipolar_distance(Fij, pts_i, pts_j):
    return batch_calc_pFp(Fij, pts_i, pts_j) * (1 / (torch.bmm(Fij, pts_j)[:, 0:2, :].norm(p=2, dim=1)) +
                                                1 / (torch.bmm(Fij.transpose(1, 2), pts_i)[:, 0:2, :].norm(p=2, dim=1)))


def get_inliers(Fij, pts_i, pts_j, threshold, method='SED'):
    if method == 'sampson':
        sam_dist = sampson_distance(Fij, pts_i, pts_j)
        inliers_idx = sam_dist < threshold
    else:
        epip_dist_i = calc_pFp(Fij, pts_i, pts_j) * (1 / torch.norm((Fij @ pts_j)[0:2, :], p=2, dim=0))
        epip_dist_j = calc_pFp(Fij, pts_i, pts_j) * (1 / torch.norm((Fij.T @ pts_i)[0:2, :], p=2, dim=0))
        inliers_idx = torch.logical_and(epip_dist_i < threshold, epip_dist_j < threshold)

    return inliers_idx


def sampson_distance(Fij, pts_i, pts_j):
    return calc_pFp(Fij, pts_i, pts_j) / torch.cat(((Fij @ pts_j)[0:2, :], (Fij.T @ pts_i)[0:2, :]), dim=0).norm(p=2, dim=0)


def batch_sampson_distance(Fij, pts_i, pts_j):
    return batch_calc_pFp(Fij, pts_i, pts_j) / \
           torch.cat((torch.bmm(Fij, pts_j)[:, 0:2, :], torch.bmm(Fij.transpose(1, 2), pts_i)[:, 0:2, :]), dim=1).norm(p=2, dim=1)

def dlt_triangulation(Ps, xs, visible_points, simplified_dlt=False, return_V_H=False):
    """
    Use  linear triangulation to find the points X[j] such that  xs[i,j] ~ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs: [m,n,2] or [m,n,3]
    :param visible_points: [m,n] a boolean matrix of which cameras see which points
    :return: Xs [n,4] normalized such the X[j,-1] == 1
    """
    m, n, _ = xs.shape
    X = np.zeros([n,4])
    if return_V_H:
        V_H_ret = []
    for i in range(n):
        cameras_showing_ind = np.where(visible_points[:, i])[0]  # The cameras that show this point
        num_cam_show = len(cameras_showing_ind)
        if num_cam_show < 2:
            X[i] = np.nan
            continue
        if simplified_dlt:
            A = np.zeros([2 * num_cam_show, 4])
        else:
            A = np.zeros([3 * num_cam_show, num_cam_show + 4])
        for j, cam_index in enumerate(cameras_showing_ind):
            xij = xs[cam_index, i, :2]
            Pj = Ps[cam_index]
            if simplified_dlt:
                A[2*j     : 2*j + 1, :] = xij[0]*Pj[2, :] - Pj[0, :]
                A[2*j + 1 : 2*j + 2, :] = xij[1]*Pj[2, :] - Pj[1, :]
            else:
                A[3 * j:3 * (j + 1), :4] = Pj
                A[3 * j:3 * j + 2, 4 + j] = -xij
                A[3 * j + 2, 4 + j] = -1

        if num_cam_show > 40:
            [U, S, V_H] = da.linalg.svd(da.from_array(A))  # in python svd returns V conjugate! so we need the last row and not column
            X[i] = pflat(V_H[-1, :4].compute().reshape([-1, 1])).squeeze()
            if return_V_H:
                V_H_ret.append((np.diag(S.compute()), V_H.compute()))
        else:
            [U, S, V_H] = np.linalg.svd(A)  # in python svd returns V conjugate! so we need the last row and not column
            X[i] = pflat(V_H[-1, :4].reshape([-1, 1])).squeeze()
            if return_V_H:
                V_H_ret.append((np.diag(S), V_H))

    if return_V_H:
        return X, V_H_ret
    return X

def n_view_triangulation(Ps, M, Ns=None, return_V_H=False):
    # normalizing matrix can be K inverse or a matrix that normalizes the points in M
    assert isinstance(Ps, np.ndarray) and isinstance(M, np.ndarray)
    xs = M_to_xs(M)
    visible_points = xs_valid_points(xs)
    if Ns is not None:
        Ps = Ps.copy()
        Ps, xs = normalize_points_cams(Ps, xs, Ns)
    if return_V_H:
        X, V_H = dlt_triangulation(Ps, xs, visible_points, return_V_H=return_V_H)
        return X.T, V_H
    X = dlt_triangulation(Ps, xs, visible_points)
    return X.T


def xs_valid_points(xs):
    """

    :param xs: [m,n,2]
    :return: A boolean matrix of the visible 2d points
    """
    return dataset_utils.get_M_valid_points(xs)
    # # NOTE / TODO: This function merely determines which points are visible in which cameras.
    # # Somewhat confusingly, dataset_utils.get_M_valid_points() is very similar, but returns a mask which is additionally False for entire columns, in case the corresponding point is not deemed to be visible in enough cameras.
    # if isinstance(xs, np.ndarray):
    #     return np.abs(xs).sum(axis=2) > 0
    # else:
    #     return torch.abs(xs).sum(dim=2) > 0


def normalize_M(M, Ns, valid_points=None):
    if valid_points is None:
        valid_points = dataset_utils.get_M_valid_points(M)
    norm_M = M.clone()
    n_images = norm_M.shape[0]//2
    norm_M = norm_M.reshape([n_images, 2, -1]) # [m,2,n]
    norm_M = torch.cat((norm_M, torch.ones(n_images, 1, norm_M.shape[-1], device=M.device)), dim=1)  # [m,3,n]

    norm_M = (Ns @ norm_M).permute(0, 2, 1)[:,:,:2]  # [m,3,3]@[m,3,n] -> [m,3,n]->[m,n,3]
    if norm_M.is_cuda:
        norm_M[~valid_points, :] = 0
    else:
        invalid_points_idx = np.nonzero((~valid_points).detach().numpy())
        norm_M[invalid_points_idx[0], invalid_points_idx[1], :] = 0
    return norm_M


def normalize_ts(ts, return_parameters=False):
    ts_norm = ts.clone()

    trans_vec = ts_norm.mean(dim=0)
    ts_norm = ts_norm - trans_vec

    scale_factor = ts_norm.norm(p=2, dim=1).mean()
    ts_norm = ts_norm / scale_factor

    if return_parameters:
        return ts_norm, trans_vec, scale_factor
    else:
        return ts_norm


def get_positive_projected_pts_mask(pts2D, infinity_pts_margin):
    return pts2D[:, 2, :] >= infinity_pts_margin


def get_projected_pts_mask(pts2D, infinity_pts_margin):
    return pts2D[:, 2, :].abs() >= infinity_pts_margin


def ones_padding(pts):
    return torch.cat((pts, torch.ones(1, pts.shape[1], device=pts.device)))


def sample_sub_matrix(m,n, part=0.5):
    i_idx = np.sort(np.random.choice(m, size=int(m*part), replace=False))
    j_idx = np.sort(np.random.choice(n, size=int(n*part), replace=False))
    together_idx = np.ix_(i_idx, j_idx)
    return together_idx


def bound_function(x,alpha=1,beta=1,gamma=1):
    return alpha*(x/beta)/((x/beta)+gamma)


def cross_product_2d_points(pts1, pts2, dim, epsilon=1e-4):
    cross = torch.cross(pts1 / (pts1.norm(dim=dim) + epsilon).unsqueeze(dim=dim),
                        pts2 / (pts2.norm(dim=dim) + epsilon).unsqueeze(dim=dim), dim=dim)
    return cross


def extract_specified_depths(depths_sparsemat=None, depths_dense=None, indices=None):
    """
    Given depths for all projections (either a SparseMat or a pair of a dense matrix along with
    indices of the specified elements), extract the depth values in the order of the provided indices
    (or in case of a SparseMat, the SparseMat.indices).
    """
    if depths_sparsemat is not None:
        # "depths_sparsemat" provided, in which case we do not expect "depths_dense" and "indices".
        assert depths_dense is None
        assert indices is None
        assert isinstance(depths_sparsemat, sparse_utils.SparseMat)

        assert len(depths_sparsemat.shape) == 3
        assert depths_sparsemat.shape[2] == 1

        depth_values = depths_sparsemat.values.squeeze(1)
    else:
        # "depths_dense" and "indices" provided, in which case we do not expect "depths_sparsemat".
        assert depths_sparsemat is None
        assert depths_dense is not None
        assert indices is not None

        assert indices.ndim == 2
        assert indices.shape[0] == 2

        depth_values = depths_dense[indices[0, :], indices[1, :]]

    assert depth_values.ndim == 1

    return depth_values


def determine_depth_scale(depth_values=None, depths_sparsemat=None, depths_dense=None, indices=None):
    """
    Given either true or estimated depths of the scenepoints when projecting in the cameras for which it is visible, determine a scale of these depths in a consistent manner, by which we may normalize the scene.
    """
    if depth_values is None:
        depth_values = extract_specified_depths(depths_sparsemat=depths_sparsemat, depths_dense=depths_dense, indices=indices)
    else:
        # "depth_values" provided, in which case we expect neither "depths_sparsemat" nor "depths_dense" and "indices".
        assert depths_sparsemat is None
        assert depths_dense is None
        assert indices is None

    # NOTE:
    # Currently, the total depth scale is determined by considering the depths of all projections as a collection of independent samples.
    # This is a quite simple approach. As an alternative, we could consider averaging all intra-view average depths, with equal weight on each view.
    # We would then need access to the original sparse matrix of depths, rather than the extracted "depth_values".

    # if depth_values is not None:
    #     # "depth_values" provided, in which case we expect neither "depths_sparsemat" nor "depths_dense" and "indices".
    #     assert depths_sparsemat is None
    #     assert depths_dense is None
    #     assert indices is None
    #     pass
    # elif depths_sparsemat is not None:
    #     # "depths_sparsemat" provided, in which case we expect neither "depth_values" nor "depths_dense" and "indices".
    #     assert depth_values is None
    #     assert depths_dense is None
    #     assert indices is None
    #     assert isinstance(depths_sparsemat, sparse_utils.SparseMat)

    #     assert len(depths_sparsemat.shape) == 3
    #     assert depths_sparsemat.shape[2] == 1

    #     depth_values = depths_sparsemat.values.squeeze(1)
    # else:
    #     # "depths_dense" and "indices" provided, in which case we do not expect "depth_values" or "depths_sparsemat".
    #     assert depth_values is None
    #     assert depths_sparsemat is None
    #     assert depths_dense is not None
    #     assert indices is not None

    #     assert indices.ndim == 2
    #     assert indices.shape[0] == 2

    #     depth_values = depths_dense[indices[0, :], indices[1, :]]

    assert depth_values.ndim == 1

    scale = torch.mean(depth_values)
    # scale = torch.median(depth_values)

    return scale
