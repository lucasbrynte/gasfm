import copy
import torch
from utils.general_utils import nonzero_safe


def sparse_dim_mask(x, dtype=torch.bool):
    """
    Given the sparse hybrid COO tensor x, return a sparse 0/1 mask tensor whose shape coincides with the sparse dimensions of x.
    The mask is 1 wherever x is nonempty and 0 elsewhere.
    """
    assert x.is_sparse
    x = x.coalesce()
    n_nonempty = x.values().shape[0]
    mask = torch.sparse_coo_tensor(
        x.indices(),
        torch.ones((n_nonempty,), dtype=dtype, device=x.device),
        size = x.shape[:x.sparse_dim()],
    )
    return mask

def dense_and_fill(x, fill_val):
    """
    Convert a sparse tensor to dense, and fill the non-specified elements with the given value.
    """
    assert x.is_sparse
    x = x.coalesce()
    mask = sparse_dim_mask(x)
    x_dense = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(fill_val)
    se_idx = tuple( idx.tolist() for idx in x.indices() )
    x_dense[se_idx, ...] = x.values()
    return x_dense

def get_n_nonempty(x, dim, keepdim=False):
    """
    Calculate the number of nonempty elements along the given sparse dimension.
    """
    assert 0 <= dim < x.sparse_dim()
    mask = sparse_dim_mask(x, dtype=torch.int64)
    # NOTE: Important to use integer rather than bool data type before summing.
    ret = torch.sparse.sum(mask, dim=dim)
    if keepdim:
        ret = ret.unsqueeze(dim)
    return ret

def filter_by_1dmask(x, dim, mask=None, keep_idx=None):
    """
    Filter a sparse tensor along a dimension, keeping only the indices that are True according to the provided 1D mask.
    Equivalent to masked indexing of dense tensors with a 1D mask.
    """
    assert x.is_sparse
    assert 0 <= dim < x.sparse_dim()
    assert x.sparse_dim() == 2

    if mask is None:
        assert keep_idx is not None
        # Construct a corresponding (1D) mask from the indices
        mask = torch.zeros((x.shape[dim],), dtype=torch.bool, device=x.device)
        mask[keep_idx] = True
    if keep_idx is None:
        assert mask is not None
        assert not mask.is_sparse
        assert mask.shape == (x.shape[dim],)
        # keep_idx = nonzero_safe(mask)[0]

    x = x.coalesce()

    cum_n_dropped = torch.cumsum(~mask, dim=0)
    old_indices = x.indices()
    old_values = x.values()

    # Reindex according to the elements to be dropped:
    indices = old_indices.clone()
    indices[dim, :] -= cum_n_dropped[old_indices[dim, :]]

    # Drop masked elements:
    idx_of_idx_to_keep = nonzero_safe(mask[old_indices[dim, :]])[0]
    # _, cnts = torch.unique(idx_of_idx_to_keep, return_counts=True)
    # assert torch.all(cnts == 1)
    indices = indices[:, idx_of_idx_to_keep]
    values = old_values[idx_of_idx_to_keep, ...]

    shape = list(x.shape)
    n_pruned = torch.sum(~mask).item()
    if n_pruned > 0:
        assert cum_n_dropped[-1] == n_pruned
    shape[dim] -= n_pruned

    x = torch.sparse_coo_tensor(indices, values, size=shape)
    return x

def sparse_mean(x, dim, keepdim=False, bessels_correction=False):
    """
    Calculate the mean of the sparse tensor x along the given dimension.
    Empty elements do not contribute to the sum of elements in the denominator of the average, when normalizing the sum.
    If bessels_correction=True, then division by N-1 rather than N is carried out (useful for the averaging in an unbiased variance / covariance estimator).
    """
    if not isinstance(dim, (tuple, list)):
        dim = [dim]
    for curr_dim in dim:
        assert 0 <= curr_dim < x.sparse_dim()

    assert x.is_sparse
    x = x.coalesce()

    mask = torch.sparse_coo_tensor(x.indices(), torch.ones_like(x.values()), size=x.shape)
    x_sum = torch.sparse.sum(x, dim=dim)
    x_N = torch.sparse.sum(mask, dim=dim)

    assert x_sum.is_sparse == x_N.is_sparse

    if len(dim) == x.sparse_dim():
        # In case we are summing over all sparse dimensions, the resulting sum will be dense.
        assert not x_sum.is_sparse
        x_mean = x_sum / (x_N - 1) if bessels_correction else x_sum / x_N
    else:
        # Otherwise, the result is sparse.
        assert x_sum.is_sparse
        # We want to perform element-wise division, but this is not implemented for sparse COO tensors.
        # Instead, exploit the fact that the specified elements and the corresponding indices are equal for the numerator and denominator, and simply perform the division on the values.
        assert torch.all(x_sum.indices() == x_N.indices())
        x_mean = torch.sparse_coo_tensor(
            x_sum.indices(),
            x_sum.values() / (x_N.values() - 1) if bessels_correction else x_sum.values() / x_N.values(),
            size = x_sum.shape,
        )

    if keepdim:
        for curr_dim in sorted(dim):
            x_mean = x_mean.unsqueeze(curr_dim)

    return x_mean


def sparse_elementwise_outer_product(x):
    assert x.is_sparse
    assert x.dense_dim() == 1

    x = x.coalesce()

    shape = list(x.shape)
    shape.append(shape[-1])

    outer_prod = torch.sparse_coo_tensor(
        x.indices(),
        x.values()[:, :, None] @ x.values()[:, None, :],
        size = shape,
    )

    return outer_prod

def sparse_moment_estimation(x, dim, keepdim=False):
    """
    """
    if not isinstance(dim, (tuple, list)):
        dim = [dim]
    for curr_dim in dim:
        assert 0 <= curr_dim < x.sparse_dim()

    assert x.is_sparse
    x = x.coalesce()

    x_mu = sparse_mean(x, dim, keepdim=keepdim)
    x_sigma = sparse_mean(sparse_elementwise_outer_product(x), dim, keepdim=keepdim, bessels_correction=True)

    return x_mu, x_sigma


def squeeze_dense_dim(x, dim):
    assert x.is_sparse
    if dim < 0:
        dim += x.ndim
    assert dim >= x.sparse_dim()
    assert dim < x.ndim
    x = x.coalesce()
    shape = list(x.shape)
    new_shape = shape[:dim] + shape[dim+1:]
    x = torch.sparse_coo_tensor(
        x.indices(),
        x.values().squeeze(dim - (x.sparse_dim() - 1)),
        size = new_shape,
    )
    return x

def unsqueeze_dense_dim(x, dim):
    assert x.is_sparse
    if dim < 0:
        dim += x.ndim + 1
    assert dim >= x.sparse_dim()
    assert dim < x.ndim + 1
    x = x.coalesce()
    shape = list(x.shape)
    new_shape = shape[:dim] + [1] + shape[dim:]
    x = torch.sparse_coo_tensor(
        x.indices(),
        x.values().unsqueeze(dim - (x.sparse_dim() - 1)),
        size = new_shape,
    )
    return x

def sparse_scatter_reduce(x, dim, reduce, keepdim=False, return_dense=False):
    """
    Perform a scatter-reduce operation of the sparse tensor x along the given dimension.
    Assumes a (m, n, d) tensor, with 2 sparse and 1 dense dimension, or alternatively a (m, n) tensor, in which case it is internally reshaped to a (m, n, 1) tensor.
    """
    assert 0 <= dim < x.sparse_dim()
    assert x.sparse_dim() == 2
    if x.dense_dim() == 0:
        unsqueezed = True
        x = unsqueeze_dense_dim(x, -1)
    else:
        unsqueezed = False
    assert x.dense_dim() == 1
    m, n, d = x.shape

    assert x.is_sparse
    x = x.coalesce()

    if reduce == 'sum':
        init_val = 0
    elif reduce == 'prod':
        init_val = 1
    elif reduce == 'mean':
        init_val = 0
    elif reduce == 'amin':
        init_val = float('Inf')
    elif reduce == 'amax':
        init_val = -float('Inf')
    else:
        raise NotImplementedError('Support for reduce operation {} not implemented.'.format(reduce))

    non_agg_dim = {0: 1, 1: 0}[dim]
    nonreduced_dim_size = {0: m, 1: n}[non_agg_dim]
    xagg_dense = torch.empty((nonreduced_dim_size, d), dtype=x.values().dtype, device=x.values().device).fill_(init_val)
    xagg_dense.scatter_reduce_(0, x.indices()[non_agg_dim, :, None].repeat(1, d), x.values(), reduce)

    if keepdim:
        xagg_dense = xagg_dense.unsqueeze(dim)

    # Convert to sparse representation.
    if return_dense:
        xagg = xagg_dense
        if unsqueezed:
            xagg = xagg.squeeze(-1)
    else:
        mask = sparse_dim_mask(x, dtype=torch.int64)
        x_N = torch.sparse.sum(mask, dim=dim)
        x_N = x_N.coalesce()
        if keepdim:
            assert x_N.indices().shape[0] == 2 # Expect 2D sparse tensor
            assert xagg_dense.ndim == 3
            xagg = torch.sparse_coo_tensor(
                x_N.indices(),
                xagg_dense[x_N.indices()[0, :], x_N.indices()[1, :], :],
                size = xagg_dense.shape,
            )
        else:
            assert x_N.indices().shape[0] == 1 # Expect 1D sparse tensor
            assert xagg_dense.ndim == 2
            xagg = torch.sparse_coo_tensor(
                x_N.indices(),
                xagg_dense[x_N.indices()[0, :], :],
                size = xagg_dense.shape,
            )
        if unsqueezed:
            xagg = squeeze_dense_dim(xagg, -1)

    return xagg


def sparse_min(x, dim, keepdim=False, return_dense=False):
    """
    Calculate "min" of the sparse tensor x along the given dimension.
    """
    return sparse_scatter_reduce(x, dim, 'amin', keepdim=keepdim, return_dense=return_dense)


def sparse_max(x, dim, keepdim=False, return_dense=False):
    """
    Calculate "max" of the sparse tensor x along the given dimension.
    """
    return sparse_scatter_reduce(x, dim, 'amax', keepdim=keepdim, return_dense=return_dense)


def sparsemat_feature_cat(sparsemats_list):
    """
    Given a list of SparseMat instances, create a new SparseMat which is the concatentaiton of them all along the feature dimension.
    Assumes a fixed sparsity pattern.
    """
    assert len(sparsemats_list) > 0
    indices = sparsemats_list[0].indices
    cam_per_pts = sparsemats_list[0].cam_per_pts
    pts_per_cam = sparsemats_list[0].pts_per_cam
    shape = sparsemats_list[0].shape
    device = sparsemats_list[0].device

    new_nfeat = 0
    for x in sparsemats_list:
        assert torch.all(x.indices == indices)
        assert torch.all(x.cam_per_pts == cam_per_pts)
        assert torch.all(x.pts_per_cam == pts_per_cam)
        assert len(x.shape) == len(shape) and x.shape[:2] == shape[:2]
        assert x.device == device
        new_nfeat += x.shape[2]

    cat_vals = torch.cat([x.values for x in sparsemats_list], dim=1)

    return SparseMat(cat_vals, indices, cam_per_pts, pts_per_cam, list(shape[:2])+[new_nfeat])


def sparse_pextend(x):
    """
    Homogenize a sparse collection of vectors.
    """
    assert x.is_sparse
    assert x.sparse_dim() == 2
    assert x.dense_dim() == 1
    m, n, d = x.shape

    x = x.coalesce()

    indices = x.indices()
    values = x.values()
    nse = indices.shape[1]
    assert indices.shape == (2, nse)
    assert values.shape == (nse, d)
    values = torch.cat(
        (values, torch.ones((nse, 1), dtype=x.dtype, device=x.device)),
        dim = 1,
    )
    assert values.shape == (nse, d+1)

    x = torch.sparse_coo_tensor(indices, values, size=(m, n, d+1))
    return x


def sparse_pflat(x):
    """
    Dehomogenize a sparse collection of vectors.
    """
    assert x.is_sparse
    assert x.sparse_dim() == 2
    assert x.dense_dim() == 1
    m, n, d = x.shape

    x = x.coalesce()

    indices = x.indices()
    values = x.values()
    nse = indices.shape[1]
    assert indices.shape == (2, nse)
    assert values.shape == (nse, d)
    values = values[:, :-1] / values[:, [-1]]
    assert values.shape == (nse, d-1)

    x = torch.sparse_coo_tensor(indices, values, size=(m, n, d-1))
    return x


def sparse_rowwise_matvecmul(A, x):
    """
    Calculate the rowwise matrix-vector multiplication A*x.
    A holds M (m, n) matrices, bundled in a single dense tensor A of shape (M, m, n)
    x is a sparse matrix of shape (M, N, n), where M & N are sparse dimensions, holding a sparse collection of n-vectors.
    The result has shape (M, N, m).
    """
    assert not A.is_sparse
    assert A.ndim == 3
    assert x.is_sparse
    assert x.sparse_dim() == 2
    assert x.dense_dim() == 1
    M, N = x.shape[:2]
    m, n = A.shape[1:]
    assert A.shape[0] == M
    assert x.shape[2] == n

    x = x.coalesce()

    indices = x.indices()
    values = x.values()
    nse = indices.shape[1]
    assert indices.shape == (2, nse)
    assert values.shape == (nse, n)

    # Associate row indices with dimension 0 of A:
    values = (A[indices[0], :, :] @ values[:, :, None]).squeeze(2)
    assert values.shape == (nse, m)

    x = torch.sparse_coo_tensor(indices, values, size=(M, N, m))
    return x


class SparseMat:
    def __init__(self, values, indices, cam_per_pts, pts_per_cam, shape):
        assert len(shape) == 3
        self.values = values
        self.indices = indices
        self.shape = shape
        self.cam_per_pts = cam_per_pts
        self.pts_per_cam = pts_per_cam
        self.device = self.values.device

    @property
    def size(self):
        return self.shape

    def sum(self, dim):
        assert dim == 1 or dim == 0
        n_features = self.shape[2]
        out_size = self.shape[0] if dim == 1 else self.shape[1]
        indices_index = 0 if dim == 1 else 1
        mat_sum = torch.zeros(out_size, n_features, device=self.device)
        return mat_sum.index_add(0, self.indices[indices_index], self.values)

    def mean(self, dim):
        assert dim == 1 or dim == 0
        if dim == 0:
            return self.sum(dim=0) / self.cam_per_pts
        else:
            return self.sum(dim=1) / self.pts_per_cam

    def to(self, device, **kwargs):
        ret = copy.copy(self)
        ret.device = device
        ret.values = ret.values.to(device, **kwargs)
        ret.indices = ret.indices.to(device, **kwargs)
        ret.pts_per_cam = ret.pts_per_cam.to(device, **kwargs)
        ret.cam_per_pts = ret.cam_per_pts.to(device, **kwargs)
        return ret

    def __add__(self, other):
        assert self.shape == other.shape
        # assert (self.indices == other.indices).all()  # removed due to runtime
        new_values = self.values + other.values
        return SparseMat(new_values, self.indices, self.cam_per_pts, self.pts_per_cam, self.shape)

    def to_torch_hybrid_sparse_coo(self):
        """
        Convert the sparse matrix representation to a conventional torch sparse (hybrid) COO tensor.
        This representaiton is practically the same.
        The shape is and remains (m, n, n_feat), where there are two sparse dimensions (m, n), followed by one dense dimension (n_feat,).
        """
        ret = torch.sparse_coo_tensor(
            self.indices,
            self.values,
            size = self.shape,
        ).coalesce()
        assert ret.sparse_dim() == 2
        assert ret.dense_dim() == 1
        return ret
