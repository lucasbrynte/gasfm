import copy
import math
import torch
import torch_geometric
from datasets import SceneData
from utils import geo_utils, general_utils, sparse_utils, plot_utils
from utils.general_utils import nonzero_safe
from utils.constants import *
import numpy as np


def is_valid_sample(data):
    return data.x.pts_per_cam.min().item() >= MIN_N_POINTS_PER_VIEW and data.x.cam_per_pts.min().item() >= MIN_N_VIEWS_PER_POINT
    # return data.x.pts_per_cam.min().item() >= MIN_N_POINTS_PER_VIEW


def divide_indices_to_train_test(N, n_val, n_test=0):
    perm = np.random.permutation(N)
    test_indices = perm[:n_test] if n_test>0 else []
    val_indices = perm[n_test:n_test+n_val]
    train_indices = perm[n_test+n_val:]
    return train_indices, val_indices, test_indices


def sample_indices(N, num_samples, adjacent):
    if num_samples == 1:  # Return all the data
        indices = np.arange(N)
    else:
        if num_samples < 1:
            num_samples = int(np.ceil(num_samples * N))
        num_samples = max(2, num_samples)
        if num_samples>=N:
            return np.arange(N)
        if adjacent:
            start_ind = np.random.randint(0,N-num_samples+1)
            end_ind = start_ind+num_samples
            indices = np.arange(start_ind, end_ind)
        else:
            indices = np.random.choice(N,num_samples,replace=False)
    return indices


def dump_predictions(outputs, conf, curr_epoch, phase, include_img_pts=False, additional_identifiers=[]):
    if not include_img_pts and 'xs' in outputs:
        del outputs['xs']
    general_utils.dump_predictions(conf, outputs, outputs['scene_name'], phase, curr_epoch, additional_identifiers=additional_identifiers)


def get_data_statistics(all_data):
    valid_pts = all_data.valid_pts
    valid_pts_stat = valid_pts.sum(dim=0).float()
    stats = {"Max_2d_pt": all_data.M.max().item(), "Num_2d_pts": valid_pts.sum().item(), "n_pts":all_data.M.shape[-1],
             "Cameras_per_pts_mean": valid_pts_stat.mean().item(), "Cameras_per_pts_std": valid_pts_stat.std().item(),
             "Num of cameras": all_data.y.shape[0]}
    return stats


def correct_matches_global(M, Ps, Ns):
    M_invalid_pts = np.logical_not(get_M_valid_points(M))

    Xs = geo_utils.n_view_triangulation(Ps, M, Ns)
    xs = geo_utils.batch_pflat((Ps @ Xs))[:, 0:2, :]

    # Remove invalid points
    xs[np.isnan(xs)] = 0
    xs[np.stack((M_invalid_pts, M_invalid_pts), axis=1)] = 0

    return xs.reshape(M.shape)


def reshape_M_to_3D_array(M):
    """
    Reshape a (2*m, n) measurement matrix to a (m, n, 2) 3D array.
    """
    assert len(M.shape) == 2
    assert M.shape[0] % 2 == 0
    n_pts = M.shape[-1]
    M = M.reshape((-1, 2, n_pts)) # (m, 2, n)
    if type(M) is torch.Tensor:
        M = M.transpose(1, 2) # (m, n, 2)
    else:
        M = M.swapaxes(1, 2) # (m, n, 2)
    return M


def get_M_valid_points(M):
    n_pts = M.shape[-1]

    if len(M.shape) == 2:
        # (2*m, n) measurement matrix -> (m, n, 2) 3D array:
        M = reshape_M_to_3D_array(M)
    assert len(M.shape) == 3
    assert M.shape[2] == 2

    if type(M) is torch.Tensor:
        # (m, n, 2) measurement array -> (m, n) validity mask, determined by which points are == (0, 0)
        M_valid_pts = torch.abs(M).sum(dim=2) != 0
        # Due to that we sometimes subsample the set of views, update the (m, n) validity mask, with the further requirement that points are only valid if they are visible in >= 2 views.
        # As a consequence, there may be views without enough visible points to estimate camera motion.
        # While we do not annotate projections of invalid views (as we are projections of invalid points), at the moment, we seem to simple discard subsets of scenes with this issue during training, so it is not an issue in practice.
        # If we were to subsample the set of points as well, we should probably pay even more attention to this, as it may happen more frequently.
        if M_valid_pts.is_cuda:
            M_valid_pts[:, M_valid_pts.sum(dim=0) < MIN_N_VIEWS_PER_POINT] = False
        else:
            # NOTE: Workaround for bug in nonzero_out_cpu(), internally called by pytorch during advanced indexing operation.
            idx = np.nonzero((M_valid_pts.sum(dim=0) < MIN_N_VIEWS_PER_POINT).numpy())[0]
            assert len(idx.shape) == 1, 'Expected 1D-array, but encountered idx.shape == {}'.format(idx.shape)
            M_valid_pts[:, torch.from_numpy(idx)] = False
    else:
        M_valid_pts = np.abs(M).sum(axis=2) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < MIN_N_VIEWS_PER_POINT] = False

    return M_valid_pts


def M2sparse(M, normalize=False, Ns=None):
    """
    Given dense measurement matrix M with shape (2*m, n), return a reshaped, sparse, (m, n, 2) array.
    Optionally, normalize the 2D points with the provided view-specific matrices "Ns" as well, shape (m, 3, 3).
    """
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)
    if valid_pts.is_cuda:
        mat_indices = torch.nonzero(valid_pts).T
    else:
        ############################################
        # Workaround for buggy behavior of torch.nonzero() in Pytorch 1.12 when applied on CPU-tensors.
        #-------------------------------------------
        # The error appears intermittently.
        # If we are "lucky", the INTERNAL ASSERT at the following line in the nonzero_out_cpu() function will fail.
        # In that case, python prints a stack trace indicating that the error indeed arose during a call to torch.nonzero().
        # https://github.com/pytorch/pytorch/blob/67ece03c8cd632cce9523cd96efde6f2d1cc8121/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L2038
        # If we are less lucky, we observe an immediate crash, without any stack trace or any explicit information pointing at the torch.nonzero() call,
        # along with some variation of a "Segmentation Fault" / "dumped core" messages.
        # Experiments have, however, pointed at the torch.nonzero() call causing the crash, by printing a debug mesage before and after each call, and only seeing the former message printed before the crash.

        # Skip .T, as the numpy result is already transposed:
        mat_indices = torch.from_numpy(np.array(np.nonzero(valid_pts.numpy())))

    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        M = norm_M
    else:
        M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)

    mat_vals = M[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    return sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape)


class OutlierInjector():
    def __init__(self, all_proj, outlier_injection_rate):
        assert 0 < outlier_injection_rate < 1

        self.all_proj = all_proj.coalesce()
        self.outlier_injection_rate = outlier_injection_rate

        self.n_views, self.n_points = self.all_proj.shape[:2]
        self.all_proj_mask = sparse_utils.sparse_dim_mask(self.all_proj).coalesce()

        self.fixed_inliers_idx_mask, self.fixed_outliers_idx_mask = self.init_fixed_inliers_and_outliers()
        self.free_inliers_idx_mask, self.free_outliers_idx_mask = self.init_free_inliers_and_outliers()

        self.verify_partitions()

    @property
    def remaining_inliers_mask(self):
        return self.filter_by_idx_mask(self.all_proj_mask, self.fixed_inliers_idx_mask|self.free_inliers_idx_mask)

    @property
    def remaining_inliers_proj(self):
        return self.filter_by_idx_mask(self.all_proj, self.fixed_inliers_idx_mask|self.free_inliers_idx_mask)

    @property
    def n_proj_total(self):
        return self.all_proj_mask.indices().shape[1]

    @property
    def n_inliers(self):
        return torch.sum(self.fixed_inliers_idx_mask | self.free_inliers_idx_mask).item()

    @property
    def n_outliers(self):
        return torch.sum(self.fixed_outliers_idx_mask | self.free_outliers_idx_mask).item()

    @property
    def n_fixed_inliers(self):
        return torch.sum(self.fixed_inliers_idx_mask).item()

    @property
    def n_fixed_outliers(self):
        return torch.sum(self.fixed_outliers_idx_mask).item()

    @property
    def n_free_inliers(self):
        return torch.sum(self.free_inliers_idx_mask).item()

    @property
    def n_free_outliers(self):
        return torch.sum(self.free_outliers_idx_mask).item()

    @property
    def n_inlier_or_outlier_candidates(self):
        return torch.sum(~(self.fixed_inliers_idx_mask | self.fixed_outliers_idx_mask)).item()

    @property
    def target_n_outliers(self):
        return round(self.outlier_injection_rate * self.n_proj_total)

    def verify_partitions(self):
        # Verify that the subsets of projections are mutually exclusive, and that they are covering all projections.
        assert torch.all(torch.sum(torch.vstack((
            self.fixed_inliers_idx_mask,
            self.fixed_outliers_idx_mask,
            self.free_inliers_idx_mask,
            self.free_outliers_idx_mask,
        )), dim=0, dtype=torch.int64) == 1)

    def filter_by_idx_mask(self, x, keep_idx_mask):
        assert x.is_coalesced() # If not already coalesced, the ordering cannot be matched with the given index mask.
        preserved_idx = nonzero_safe(keep_idx_mask)[0]
        x_filtered = torch.sparse_coo_tensor(
            x.indices()[:, preserved_idx],
            x.values()[preserved_idx, ...],
            size = x.shape,
        ).coalesce()
        assert x_filtered.shape == x.shape
        return x_filtered

    def verify_enough_points_per_view(self, proj_mask=None, pts_per_view=None):
        if pts_per_view is None:
            if proj_mask is None:
                proj_mask = self.all_proj_mask
            pts_per_view = sparse_utils.get_n_nonempty(proj_mask, 1).to_dense()
        assert torch.all(pts_per_view >= MIN_N_POINTS_PER_VIEW)

    def verify_enough_views_per_point(self, proj_mask=None, views_per_pt=None):
        if views_per_pt is None:
            if proj_mask is None:
                proj_mask = self.all_proj_mask
            views_per_pt = sparse_utils.get_n_nonempty(proj_mask, 0).to_dense()
        assert torch.all(views_per_pt >= MIN_N_VIEWS_PER_POINT)

    def init_fixed_inliers_and_outliers(self):
        pts_per_view = sparse_utils.get_n_nonempty(self.all_proj_mask, 1).to_dense()
        views_per_pt = sparse_utils.get_n_nonempty(self.all_proj_mask, 0).to_dense()
        self.verify_enough_points_per_view(pts_per_view=pts_per_view)
        self.verify_enough_views_per_point(views_per_pt=views_per_pt)

        # Determine all projections that are candidates to be outliers (if removed from inliers, we would still have enough views per point and points per view for the inliers).
        # First, determine the projections that are not outlier candidates (rather, fixed inliers).
        disregarded_points_idx = nonzero_safe(views_per_pt < MIN_N_VIEWS_PER_POINT+1)[0]
        disregarded_views_idx = nonzero_safe(pts_per_view < MIN_N_POINTS_PER_VIEW+1)[0]
        disregarded_points_idx_mask = general_utils.mark_occurences_of_tensor_in_other(self.all_proj_mask.indices()[1, :], disregarded_points_idx)
        disregarded_views_idx_mask = general_utils.mark_occurences_of_tensor_in_other(self.all_proj_mask.indices()[0, :], disregarded_views_idx)

        fixed_inliers_idx_mask = disregarded_points_idx_mask | disregarded_views_idx_mask
        fixed_outliers_idx_mask = torch.zeros_like(fixed_inliers_idx_mask) # No fixed outliers so far
        # # And all other projections are candidates to be either inliers or outliers.
        # inlier_or_free_outliers_idx_mask = ~(fixed_inliers_idx_mask | fixed_outliers_idx_mask)

        return fixed_inliers_idx_mask, fixed_outliers_idx_mask

    def init_free_inliers_and_outliers(self):
        free_inliers_idx_mask = ~(self.fixed_inliers_idx_mask | self.fixed_outliers_idx_mask)
        # free_inliers_idx_mask = torch.zeros_like(self.fixed_inliers_idx_mask)
        free_outliers_idx_mask = torch.zeros_like(self.fixed_inliers_idx_mask)
        return free_inliers_idx_mask, free_outliers_idx_mask

    def add_margin_to_outlier_rate(self, outlier_injection_rate=None, w_desired=0.5):
        # Set a new outlier rate with some margin for unmarking some of the outliers, if they lead to too few points / view or views / point.
        if outlier_injection_rate is None:
            outlier_injection_rate = self.outlier_injection_rate
        # Use a weighted harmonic mean between the desired injection rate and 1.
        rate_with_margin = 1.0 / (w_desired*1.0/outlier_injection_rate + (1.0-w_desired)*1.0/1.0)
        assert 0 < outlier_injection_rate < rate_with_margin < 1
        return rate_with_margin

    def add_margin_to_n_new_outliers(self, target_n_new_outliers, w_desired=0.5):
        assert target_n_new_outliers <= self.n_free_inliers
        assert 0 < w_desired < 1
        rate_with_margin = self.add_margin_to_outlier_rate(
            # outlier_injection_rate = self.outlier_injection_rate,
            outlier_injection_rate = target_n_new_outliers / self.n_free_inliers,
            w_desired = w_desired,
        )
        target_n_new_outliers_with_margin = round(rate_with_margin * self.n_free_inliers)
        assert target_n_new_outliers_with_margin <= self.n_free_inliers

        return target_n_new_outliers_with_margin

    def sample_more_outliers(self, n_new_outliers):
        # Sample a subset of free inliers to be marked as (free) outliers, i.e. outlier candidates.
        self.free_inliers_idx_mask[np.random.choice(nonzero_safe(self.free_inliers_idx_mask)[0].cpu().numpy(), size=(n_new_outliers,), replace=False)] = False
        self.free_outliers_idx_mask = ~(self.fixed_inliers_idx_mask | self.fixed_outliers_idx_mask | self.free_inliers_idx_mask)
        self.verify_partitions()

    def blacklist_problematic_outliers(self):
        """
        Among all outlier candidates, find out which ones contribute to ending up with too few points per view or views per point, and force all of these to be inliers.
        """
        # Given the suggested inliers, determine the views & scenepoints with too few connections:
        too_few_views_per_point_idx = nonzero_safe(sparse_utils.get_n_nonempty(self.remaining_inliers_mask, 0).to_dense() < MIN_N_VIEWS_PER_POINT)[0]
        too_few_points_per_view_idx = nonzero_safe(sparse_utils.get_n_nonempty(self.remaining_inliers_mask, 1).to_dense() < MIN_N_POINTS_PER_VIEW)[0]
        too_few_views_per_point_idx_mask = general_utils.mark_occurences_of_tensor_in_other(self.all_proj_mask.indices()[1, :], too_few_views_per_point_idx)
        too_few_points_per_view_idx_mask = general_utils.mark_occurences_of_tensor_in_other(self.all_proj_mask.indices()[0, :], too_few_points_per_view_idx)

        # Then, mark all such projections as fixed inliers:
        self.fixed_inliers_idx_mask |= self.free_outliers_idx_mask & (too_few_views_per_point_idx_mask | too_few_points_per_view_idx_mask)
        self.free_outliers_idx_mask &= ~self.fixed_inliers_idx_mask
        self.verify_partitions()

    def remove_surplus_outlier_candidates(self):
        self.verify_enough_points_per_view(proj_mask=self.remaining_inliers_mask)
        self.verify_enough_views_per_point(proj_mask=self.remaining_inliers_mask)

        assert self.n_outliers >= self.target_n_outliers
        # There may be more than enough outlier candidates.
        # We simply sample a random subset of them to convert to inliers.
        assert self.n_free_outliers >= self.n_outliers - self.target_n_outliers
        self.free_outliers_idx_mask[np.random.choice(nonzero_safe(self.free_outliers_idx_mask)[0].cpu().numpy(), size=(self.n_outliers - self.target_n_outliers,), replace=False)] = False
        self.free_inliers_idx_mask = ~(self.fixed_inliers_idx_mask | self.fixed_outliers_idx_mask | self.free_outliers_idx_mask)
        assert self.n_outliers == self.target_n_outliers
        self.verify_partitions()

        self.verify_enough_points_per_view(proj_mask=self.remaining_inliers_mask)
        self.verify_enough_views_per_point(proj_mask=self.remaining_inliers_mask)

    def select_outliers(self, n_tries=5):
        if not n_tries > 0:
            # Give up
            return None
        while self.n_outliers < self.target_n_outliers:
            target_n_new_outliers = self.target_n_outliers - self.n_outliers
            if not target_n_new_outliers <= self.n_free_inliers:
                # There are not enough free points available to achieve the desired outlier rate
                # Try again from scratch, maybe we were unlucky.
                self.fixed_inliers_idx_mask, self.fixed_outliers_idx_mask = self.init_fixed_inliers_and_outliers()
                self.free_inliers_idx_mask, self.free_outliers_idx_mask = self.init_free_inliers_and_outliers()
                print('Retry outlier sampling, {} attempts remaining.'.format(n_tries-1))
                return self.select_outliers(n_tries=n_tries-1)
            assert target_n_new_outliers <= self.n_free_inliers
            target_n_new_outliers_with_margin = self.add_margin_to_n_new_outliers(target_n_new_outliers, w_desired=0.5)
            # Sample a subset of free inliers to be marked as (free) outliers, i.e. outlier candidates.
            self.sample_more_outliers(target_n_new_outliers_with_margin)
            # Among all outlier candidates, find out which ones contribute to ending up with too few points per view or views per point, and force all of these to be inliers.
            self.blacklist_problematic_outliers()
            # print('[WARNING] Aimed at {}/{} outliers, and added some margin to select {} outlier candidates, but only {} of them could be considered, or some point-to-view connections would be too few. Making recursive call to this function, to try to find more outliers to add.'.format(self.target_n_outliers, self.n_proj_total, target_n_new_outliers_with_margin, self.n_outliers))

        # Now, there should be enough outlier candidates to acquire the desired outlier rate.
        assert self.n_outliers >= self.target_n_outliers
        self.remove_surplus_outlier_candidates()

        return self.fixed_outliers_idx_mask | self.free_outliers_idx_mask

    def inject_outliers(self, all_proj=None, outliers_idx_mask=None):
        if all_proj is None:
            all_proj = self.all_proj
        if outliers_idx_mask is None:
            outliers_idx_mask = self.fixed_outliers_idx_mask | self.free_outliers_idx_mask

        # Extract the inlier / outlier projections
        outliers_proj = self.filter_by_idx_mask(all_proj, outliers_idx_mask).coalesce()
        inliers_proj = self.filter_by_idx_mask(all_proj, ~outliers_idx_mask).coalesce()

        # Make sure that there are enough points per view. We need at least 2 points for being able to estimate a covariance matrix (at least 3 for a non-degenerate one).
        assert torch.all(sparse_utils.get_n_nonempty(inliers_proj, 1).coalesce().values() >= MIN_N_POINTS_PER_VIEW)

        # Fit a bivariate Gaussian distribution to all inlier points in each view
        mu, sigma = sparse_utils.sparse_moment_estimation(inliers_proj, 1, keepdim=False)
        mu = mu.to_dense()
        sigma = sigma.to_dense()
        assert mu.shape == (self.n_views, 2)
        assert sigma.shape == (self.n_views, 2, 2)
        assert torch.all(sigma == sigma.swapaxes(1, 2)) # Should be symmetric
        # Factor the covariance matrix into two factors (any factorization Sigma=A*A^T will do, where A is real)
        LD, pivots = torch.linalg.ldl_factor(sigma)
        assert torch.all(pivots > 0) # If negative, there are 2x2 blocks present. Hopefully this never happens for PSD matrices. https://en.wikipedia.org/wiki/Cholesky_decomposition#Block_variant   https://www.netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
        # LD holds a compact joint representation of L and D, from which we need to extract the factors:
        L = LD.clone()
        L[:, [0, 1], [0, 1]] = 1
        D = torch.zeros_like(LD)
        D[:, [0, 1], [0, 1]] = LD[:, [0, 1], [0, 1]]
        scale_tril = L @ torch.sqrt(D)
        # NOTE: Typically, pivots[i,:] == [1, 2], and no permutation is carried out.
        # Sometimes there is a permutation of rows & columns, and while one would then expect pivots[i,:] == [2, 1], pivots[i,:] == [2, 2] has been observed instead.
        # Determine all matrices for which pivots[i,:] == [1, 2], and assume all others should be permuted (as we have 2x2 matrices, there is only one non-trivial joint permutation of rows & columns).
        permuted_mask = ~((pivots[:, 0] == 1) & (pivots[:, 1] == 2))
        # Switch the order of the rows:
        scale_tril[nonzero_safe(permuted_mask)[0], :, :] = torch.flip(scale_tril[nonzero_safe(permuted_mask)[0], :, :], [1]) # Flip (reverse order) along dim=1
        assert torch.all(torch.isclose(sigma, scale_tril @ scale_tril.swapaxes(1, 2)))

        # Sample artificial outliers from the distributions
        assert self.n_outliers == outliers_proj.values().shape[0]
        outlier_values = mu[outliers_proj.indices()[0, :], :] + (scale_tril[outliers_proj.indices()[0, :], :, :] @ torch.randn((self.n_outliers, 2, 1), device=scale_tril.device)).squeeze(2)
        # Inject the outliers (replacing the existing real points)
        outliers_proj = torch.sparse_coo_tensor(
            outliers_proj.indices(),
            # outliers_proj.values(),
            outlier_values,
            size = outliers_proj.shape,
        ).coalesce()

        # Merge the preserved inliers with the artificially injected outliers
        shape = outliers_proj.shape
        assert shape == inliers_proj.shape
        all_proj_incl_outliers = torch.sparse_coo_tensor(
            torch.cat((
                inliers_proj.indices(),
                outliers_proj.indices(),
            ), dim=1),
            torch.cat((
                inliers_proj.values(),
                outliers_proj.values(),
            ), dim=0),
            size = shape,
        )

        return all_proj_incl_outliers


def inject_outliers(scene_data, outlier_injection_rate):
    assert 0 < outlier_injection_rate < 1
    # # Normalized coordinates:
    # all_proj = scene_data.x.to_torch_hybrid_sparse_coo().coalesce()
    # Pixel coordinates:
    all_proj = M2sparse(scene_data.M, normalize=False).to_torch_hybrid_sparse_coo().coalesce()
    n_views, n_points = all_proj.shape[:2]

    outlier_injector = OutlierInjector(all_proj, outlier_injection_rate)

    outliers_idx_mask = outlier_injector.select_outliers(n_tries=5)
    if outliers_idx_mask is None:
        # There are not enough free points available to achieve the desired outlier rate
        return None

    all_proj_incl_outliers = outlier_injector.inject_outliers(
        all_proj = outlier_injector.all_proj,
        outliers_idx_mask = outliers_idx_mask,
    )

    M = all_proj_incl_outliers.to_dense().swapaxes(1, 2).reshape(2*n_views, n_points)
    scene_data = SceneData.SceneData(
        M,
        scene_data.Ns,
        scene_data.y,
        scene_data.scene_name,
        calibrated = scene_data.calibrated,
        store_depth_targets = scene_data.store_depth_targets,
        depths = scene_data.depths,
    )

    return scene_data


class AxialAggregationGraphWrapper():
    """
    Wrapper class for defining a graph corresponding to row-wise / column-wise aggregation of sparse matrix elements.
    """
    def __init__(
        self,
        m,
        n,
        agg_dim, # Determines along which dimension aggregation is carried out.
        valid_indices = None, # (2, n_valid_mat_elements) index matrix annotating the positions of valid matrix elements.
        device = None,
    ):
        if device is not None and valid_indices is not None:
            assert valid_indices.device == device
        if valid_indices is not None:
            self.device = valid_indices.device
        elif device is not None:
            self.device = device
        else:
            assert False
        self.m, self.n = m, n
        assert agg_dim in [0, 1]
        self.agg_dim = agg_dim
        self.non_agg_dim = {0: 1, 1: 0}[self.agg_dim] # The remaining dimension after aggregation
        self.n_agg_nodes = {0: m, 1: n}[self.non_agg_dim] # Determine the number of aggregation nodes from the size of the remaining dimension

        self.valid_indices = valid_indices
        if self.valid_indices is None:
            self.dense = True
            # If indices are not given, we assume dense connectivity.
            row_indices = torch.arange(m, dtype=torch.int64, device=self.device)
            col_indices = torch.arange(n, dtype=torch.int64, device=self.device)
            self.valid_indices = torch.cartesian_prod(row_indices, col_indices).T
            # Verify that the order is consistent with a coalesced sparse representation of a dense matrix, since this is what will determine the order during run-time:
            assert torch.all(self.valid_indices == torch.ones((m, n)).to_sparse_coo().indices())
        else:
            self.dense = False
        assert self.valid_indices.device == self.device
        assert self.valid_indices.dtype == torch.int64
        assert len(self.valid_indices.shape) == 2
        n_valid_mat_elements = self.valid_indices.shape[1]
        assert self.valid_indices.shape == (2, n_valid_mat_elements)

        # Define graph for the axial aggregation.
        self.edge_index = self.create_sparse_axial_aggregation_edges()

    # def create_sparse_axial_aggregation_graph(
    def create_sparse_axial_aggregation_edges(
        self,
        # m,
        # n,
    ):
        """
        Given a sparse feature matrix, create a graph aggregating every row or column into one node.
        The nodes consist of one node for every valid matrix element (referred to as a "matrix element nodes"), and one node for every row / column, referred to as "aggregation nodes".
        Every matrix element node is a source node, connected to the aggregation node of its corresponding row / column, consequently making up the target nodes.
        """
        # m, n = self.m, self.n
        n_valid_mat_elements = self.valid_indices.shape[1]

        # Initialize indices for aggregation nodes and matrix element nodes
        agg_node_indices = torch.arange(0, self.n_agg_nodes, dtype=torch.int64, device=self.device) # While the ordering of these indices is arbitrary, it determines the order of the output aggregation node features.
        mat_element_node_indices = torch.arange(0, n_valid_mat_elements, dtype=torch.int64, device=self.device) # While the choice of indices is arbitrary, the ordering is assumed to correspond to the elements in x_mat_elements = M[self.valid_indices[0, :], self.valid_indices[1, :], :] in self.generate_node_features().

        # Adjust aggregation node indices for the new shifted positions, resulting from concatenating source node features with (dummy) target node features in self.generate_node_features().
        agg_node_indices = agg_node_indices + n_valid_mat_elements

        # Define source and target node indices
        source_node_indices = mat_element_node_indices # Every matrix element is a source node.
        target_node_indices = agg_node_indices[self.valid_indices[self.non_agg_dim, :]] # The aggregation nodes to connect to are determined by the corresponding row / column index.

        edge_index = torch.cat([source_node_indices[None, :], target_node_indices[None, :]], dim = 0) # Graph edges, dtype long, shape (2, n_edges), where n_edges = n_valid_mat_elements

        return edge_index

    def generate_node_features(
        self,
        M, # (m, n, n_feat) feature matrix. For a sparse graph, we expect a torch hybrid sparse COO feature matrix, where the last dimension is dense.
        x_agg = None, # Optional features for aggregation nodes (acting as query vectors). Shape (n_agg_nodes, n_feat), where n_agg_nodes is either m or n, depending on along which dimension we are aggregating.
    ):
        """
        Given a sparse feature matrix, generate source nodes from its data, as well as (dummy) target nodes, on which we will apply the graph propagation.
        """
        assert M.device == self.device
        m, n, n_feat = M.shape
        assert (m, n) == (self.m, self.n)
        n_valid_mat_elements = m * n if self.dense else self.valid_indices.shape[1]

        if not self.dense:
            assert M.is_sparse
            assert M.sparse_dim() == 2
            assert M.dense_dim() == 1
            assert M.indices().shape[1] == n_valid_mat_elements
            assert torch.all(self.valid_indices == M.indices())

        # Extract matrix element node features from M
        if self.dense:
            x_mat_elements = M.reshape((m * n, n_feat)) # (n_valid_mat_elements, n_feat)
        else:
            x_mat_elements = M.values() # (n_valid_mat_elements, n_feat)
        assert x_mat_elements.shape == (n_valid_mat_elements, n_feat)

        if x_agg is not None:
            assert not x_agg.is_sparse
            assert x_agg.shape == (self.n_agg_nodes, n_feat)
        else:
            # Dummy zero features for the aggregation nodes
            x_agg = torch.zeros((self.n_agg_nodes, n_feat), dtype=torch.float32, device=self.device)

        # Concatenate source node features with (dummy) target node features
        x = torch.cat((x_mat_elements, x_agg), dim=0) # Projection features, concatenated with dummy aggregation features, shape (n_valid_mat_elements + n_agg_nodes, n_feat)

        return x

    def extract_target_node_features(self, x):
        """
        Given all node features, extract the target features only.
        Input: x, node features, shape (n_valid_mat_elements + n_agg_nodes, n_feat)
        Returns: M_agg_vec, target node features, shape (m, 1, n_feat) or (1, n, n_feat), depending on along which dimension we perform axial aggregation.
        """
        if self.agg_dim == 0:
            assert self.n_agg_nodes == self.n
            M_agg_vec = x[None, -self.n:, :]
        else:
            assert self.n_agg_nodes == self.m
            M_agg_vec = x[-self.m:, None, :]
        return M_agg_vec

    def to(self, device, **kwargs):
        ret = copy.copy(self)
        ret.device = device
        ret.valid_indices = ret.valid_indices.to(device, **kwargs)
        ret.edge_index = ret.edge_index.to(device, **kwargs)
        return ret
