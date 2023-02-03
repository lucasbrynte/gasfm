import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F
import loss_functions


def get_loss_func(conf):
    loss_func_spec = conf.get_string('loss.func')
    if loss_func_spec in ['ESFMLoss', 'ExpDepthRegularizedOSELoss', 'GTLoss']:
        assert conf.get_bool('model.view_head.enabled')
        assert conf.get_bool('model.scenepoint_head.enabled')
        assert not conf.get_bool('model.depth_head.enabled'), "Make sure that model.depth_head.enabled=False, if there is no loss applied on such output."
    elif loss_func_spec == 'DirectDepthLoss':
        assert conf.get_bool('model.depth_head.enabled')
        assert not conf.get_bool('model.view_head.enabled'), "Make sure that model.view_head.enabled=False, if there is no loss applied on such output."
        assert not conf.get_bool('model.scenepoint_head.enabled'), "Make sure that model.scenepoint_head.enabled=False, if there is no loss applied on such output."
    else:
        assert False, "Unknown loss function: {}.".format(loss_func_spec)
    loss_func = getattr(loss_functions, loss_func_spec)(conf)
    return loss_func


class DirectDepthLoss(nn.Module):
    """
    """
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.depth_head.enabled')
        self.cost_fcn = conf.get_string('loss.cost_fcn')
        assert self.cost_fcn in ['L1', 'L2']
        if not conf.get_bool('dataset.calibrated'):
            # NOTE: Even in the uncalibrated case, the depth should be possible to infer by normalizing the camera matrix such that the third row has unit length...
            raise NotImplementedError

    def forward(self, pred_dict, data, epoch=None):
        depths_pred = geo_utils.extract_specified_depths(
            depths_sparsemat = pred_dict['depths'],
        )
        depths_gt = geo_utils.extract_specified_depths(
            depths_dense = data.depths,
            indices = pred_dict['depths'].indices, # Use the same indices as above, to make sure the predicted & GT depth values are in a aingle consistent order.
        )

        # Determine depth scale
        s_pred = geo_utils.determine_depth_scale(depth_values=depths_pred)
        s_gt = geo_utils.determine_depth_scale(depth_values=depths_gt)
        # NOTE: Currently, the total depth scale is determined by considering the depths of all projections as a collection of independent samples.
        # This is a quite simple approach. In case we e.g. want to average the depths first per each view and then across all views, we would need to pass the original sparse matrix of depths to the above function instead of the extracted specified elements.
        # s_pred = geo_utils.determine_depth_scale(depths_sparsemat=pred_dict['depths'])
        # s_gt = geo_utils.determine_depth_scale(depths_dense=data.depths, indices=pred_dict['depths'].indices)

        # Normalize depths
        depths_pred = depths_pred / s_pred
        depths_gt = depths_gt / s_gt

        # TODO: Add a (small) depth scale regularization, e.g. reg_weight * (s_pred - 1.0)**2

        if self.cost_fcn == 'L1':
            loss = torch.mean(torch.abs(depths_pred - depths_gt))
        elif self.cost_fcn == 'L2':
            loss = torch.mean((depths_pred - depths_gt)**2)
        else:
            assert False

        return loss


class ESFMLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.view_head.enabled', default=False)
        assert conf.get_bool('model.scenepoint_head.enabled', default=False)
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.pts_grad_equalization_pre_perspective_divide = conf.get_bool("loss.pts_grad_equalization_pre_perspective_divide")
        if self.pts_grad_equalization_pre_perspective_divide:
            self.normalize_grad_wrt_valid_projections_only = conf.get_bool("loss.normalize_grad_wrt_valid_projections_only")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_dict, data, epoch=None):
        # The predicted cameras "Ps_norm" and the GT 2D points "data.norm_M" are normalized with the N matrices given in the dataset.
        # Consequently, the reprojection error itself is calculated in normalized image space.
        # In the Euclidean setting, N=inv(K).
        Ps = pred_dict["Ps_norm"]
        pts_2d = Ps @ pred_dict["pts3D"]  # [m, 3, n]

        # Get point for reprojection loss
        if self.hinge_loss:
            # Return mask of points whose perspective depths are positive, and larger than the threshold:
            positive_projected_pts_mask = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        else:
            # Return mask of points whose perspective depths are larger than the threshold (in absolute value): (we allow negative depths)
            positive_projected_pts_mask = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Normalize gradient
        if self.pts_grad_equalization_pre_perspective_divide:
            if self.normalize_grad_wrt_valid_projections_only:
                pts_2d.register_hook(lambda grad: torch.where(
                    positive_projected_pts_mask[:, None, :].repeat(1, 3, 1),
                    F.normalize(grad, dim=1) / max(1, torch.sum(data.valid_pts & positive_projected_pts_mask).item()), # Divide with the number of valid projections, or 1 if there are none.
                    grad, # Keep gradient unchanged for the invalid projections.
                ))
            else:
                # Original behavior (hinge loss weight effectively disregarded)
                pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / torch.sum(data.valid_pts))

        # Calculate hinge Loss
        # NOTE: While at this point the "hinge loss" is just a linear loss applied on all depths irrespective of sign, in the end it will be used to replace the reprojection error for only the points with invalid depth, and hence effectively be a hinge loss after all.
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # NOTE: While at this point reprojection errors are computed for all points, only the points with valid depths will be used in the end.
        pts_2d = (pts_2d / torch.where(positive_projected_pts_mask, pts_2d[:, 2, :], torch.ones_like(positive_projected_pts_mask).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)

        # NOTE: Use either the reprojection error or the negative depth loss, depending on whether the depth is valid or not.
        assert data.valid_pts.is_cuda # If not, we would have to modify the masking below, to avoid an implicit call to pytorch's buggy CPU-implementation of nonzero.
        return torch.where(positive_projected_pts_mask, reproj_err, hinge_loss)[data.valid_pts].mean()


class ExpDepthRegularizedOSELoss(nn.Module):
    """
    Implements the combination of Object Space Error (OSE) and an exponential depth regularization term, for pushing scene points in front of the camera.
    The result is a smooth loss function without a barrier at the principal plane.
    For each projected point, the OSE can be seen as a reprojection error computed at a z-shifted image plane, such that its depth equals the predicted depth.
    """
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.view_head.enabled', default=False)
        assert conf.get_bool('model.scenepoint_head.enabled', default=False)
        self.depth_regul_weight = conf.get_float("loss.depth_regul_weight")

    def forward(self, pred_dict, data, epoch=None):
        Ps = pred_dict["Ps_norm"]
        pts_2d = Ps @ pred_dict["pts3D"]  # [m, 3, n]

        # Calculate exponential depth regularizaiton term
        depth_reg_term = self.depth_regul_weight * torch.exp(-pts_2d[:, 2, :])

        # Calculate OSE
        pts_2d_gt = data.norm_M.reshape(Ps.shape[0], 2, -1) # (m, 2, n)
        ose_err = (pts_2d[:, :2, :] - pts_2d[:, [2], :]*pts_2d_gt).norm(dim=1)

        assert data.valid_pts.is_cuda # If not, we would have to modify the masking below, to avoid an implicit call to pytorch's buggy CPU-implementation of nonzero.
        return (ose_err + depth_reg_term)[data.valid_pts].mean()


class GTLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.view_head.enabled', default=False)
        assert conf.get_bool('model.scenepoint_head.enabled', default=False)
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_dict, data, epoch=None):
        # Get orientation
        Vs_gt = data.y[:, 0:3, 0:3].inverse().transpose(1, 2)
        if self.calibrated:
            Rs_gt = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs_gt).transpose(1, 2))

        # Get Location
        t_gt = -torch.bmm(data.y[:, 0:3, 0:3].inverse(), data.y[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # Normalize scene by points
        # trans = pts3D_gt.mean(dim=1)
        # scale = (pts3D_gt - trans.unsqueeze(1)).norm(p=2, dim=0).mean()

        # Normalize scene by cameras
        trans = t_gt.mean(dim=0)
        scale = (t_gt - trans).norm(p=2, dim=1).mean()

        t_gt = (t_gt - trans)/scale
        new_Ps = geo_utils.batch_get_camera_matrix_from_Vt(Vs_gt, t_gt)

        Vs_invT = pred_dict["Ps_norm"][:, 0:3, 0:3]
        Vs = torch.inverse(Vs_invT).transpose(1, 2)
        ts = torch.bmm(-Vs.transpose(1, 2), pred_dict["Ps"][:, 0:3, 3].unsqueeze(dim=-1)).squeeze()

        # Translation error
        translation_err = (t_gt - ts).norm(p=2, dim=1)

        # Calculate error
        if self.calibrated:
            Rs = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs).transpose(1, 2))
            orient_err = (Rs - Rs_gt).norm(p=2, dim=1)
        else:
            Vs_gt = Vs_gt / Vs_gt.norm(p='fro', dim=(1, 2), keepdim=True)
            Vs = Vs / Vs.norm(p='fro', dim=(1, 2), keepdim=True)
            orient_err = torch.min((Vs - Vs_gt).norm(p='fro', dim=(1, 2)), (Vs + Vs_gt).norm(p='fro', dim=(1, 2)))

        orient_loss = orient_err.mean()
        tran_loss = translation_err.mean()
        loss = orient_loss + tran_loss

        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("loss = {}, orient err = {}, trans err = {}".format(loss, orient_loss, tran_loss))

        return loss

