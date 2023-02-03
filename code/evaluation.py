import torch
import os
from time import time
from utils import geo_utils, ba_functions
import numpy as np


def compute_core_errors(data, pred_dict, conf):
    """
    Compute a smaller set of "core" errors, in particular reprojection error, which may be logged at every iteration without being too heavy.
    """
    core_errors = {}

    calibrated = conf.get_bool('dataset.calibrated')
    depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
    view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    # Ns = data.Ns.cpu().numpy()
    Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()  # Ks for calibrated, a normalization matrix for uncalibrated
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(M)

    if explicit_est_avail:
        Ps_norm = pred_dict['Ps_norm'].detach().cpu().numpy()  # Normalized camera!!
        Ps = Ns_inv @ Ps_norm  # unnormalized cameras
        pts3D_pred = geo_utils.pflat(pred_dict['pts3D']).detach().cpu().numpy()

        core_errors['our_repro'] = np.nanmean(geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs))

    if calc_reprojerr_with_gtposes_for_depth_pred:
        assert calibrated
        Ks = Ns_inv  # data.Ns.inverse().cpu().numpy()
        Ps_gt = data.y.detach().cpu().numpy()

        if depth_head_enabled:
            depths_pred = pred_dict['depths']
        # elif explicit_est_avail:
        elif view_head_enabled and scenepoint_head_enabled:
            raise NotImplementedError()
            # TODO:
            # - Calculate depths for all projections by projecting the estimated scenepoints in the estimated cameras.
            assert 'depths' not in pred_dict
            depths_pred = None # TODO
            pass
        else:
            assert False

        s_pred = geo_utils.determine_depth_scale(
            depths_sparsemat = depths_pred,
        ).detach().cpu().numpy()
        s_gt = geo_utils.determine_depth_scale(
            depths_dense = data.depths, # GT depths
            indices = depths_pred.indices, # Use the same indices as above, to make sure the predicted & GT depth values are in a aingle consistent order.
        ).detach().cpu().numpy()

        # GPU SparseMat -> Dense numpy
        depths_pred_dense = depths_pred.to_torch_hybrid_sparse_coo().detach().to_dense().cpu().numpy().squeeze(2)

        # Rescale predicted depths to match GT depths / poses / scene points
        depths_pred_dense_normalized = depths_pred_dense / s_pred

        core_errors['repro_backproj_rnd_gt_2view'] = np.nanmean(geo_utils.reprojection_error_backproj_random_view_pairs(
            Ks, # Calibration matrices
            Ps_gt, # GT poses (unnormalized)
            depths_pred_dense_normalized * s_gt, # predicted depths, rescaled to match GT
            # data.depths.cpu().numpy(), # GT depths (for debugging!)
            xs,
            visible_points = None, # Infer specified elements from xs
        ))

    return core_errors

def prepare_predictions(data, pred_dict, conf, bundle_adjustment):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scene_name'] = data.scene_name

    calibrated = conf.get_bool('dataset.calibrated')
    depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
    view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    Ns = data.Ns.cpu().numpy()
    Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()  # Ks for calibrated, a normalization matrix for uncalibrated
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(M)

    outputs['xs'] = xs  # to compute reprojection error later

    if calibrated:
        Ks = Ns_inv  # data.Ns.inverse().cpu().numpy()
        outputs['Ks'] = Ks

    if calc_reprojerr_with_gtposes_for_depth_pred:
        assert calibrated
        outputs['Ps_gt'] = data.y.detach().cpu().numpy()

        if depth_head_enabled:
            depths_pred = pred_dict['depths']
        # elif explicit_est_avail:
        elif view_head_enabled and scenepoint_head_enabled:
            raise NotImplementedError()
            # TODO:
            # - Calculate depths for all projections by projecting the estimated scenepoints in the estimated cameras.
            assert 'depths' not in pred_dict
            depths_pred = None # TODO
            pass
        else:
            assert False

        outputs['s_pred'] = geo_utils.determine_depth_scale(
            depths_sparsemat = depths_pred,
        ).detach().cpu().numpy()
        outputs['s_gt'] = geo_utils.determine_depth_scale(
            depths_dense = data.depths, # GT depths
            indices = depths_pred.indices, # Use the same indices as above, to make sure the predicted & GT depth values are in a aingle consistent order.
        ).detach().cpu().numpy()

        # GPU SparseMat -> Dense numpy
        outputs['depths_pred_dense'] = depths_pred.to_torch_hybrid_sparse_coo().detach().to_dense().cpu().numpy().squeeze(2)
        outputs['depths_gt_dense'] = data.depths.cpu().numpy()

    if not explicit_est_avail:
        return outputs

    Ps_norm = pred_dict['Ps_norm'].detach().cpu().numpy()  # Normalized camera!!
    Ps = Ns_inv @ Ps_norm  # unnormalized cameras
    pts3D_pred = geo_utils.pflat(pred_dict['pts3D']).detach().cpu().numpy()

    # NOTE: Here we carry out a DLT triangulation (algebraic error minimization...).
    # This triangulation may be different from the initial one optionally performed by Ceres before BA below.
    try:
        pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)
    except np.linalg.LinAlgError as e:
        # Triangulation may fail due to poor conditioning of the M matrix if the predicted poses are totally off.
        # In this case, we should handle this gracefully rather than crashing.
        pts3D_triangulated = None

    outputs['Ps'] = Ps
    outputs['Ps_norm'] = Ps_norm
    outputs['pts3D_pred'] = pts3D_pred  # 4,m
    outputs['pts3D_triangulated'] = pts3D_triangulated  # 4,n

    if calibrated:
        # NOTE!!! R & t are in fact R & C...

        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks, inverse_direction_camera2global=True)  # For alignment and R,t errors
        outputs['Rs_gt'] = Rs_gt
        outputs['ts_gt'] = ts_gt

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm, inverse_direction_camera2global=True)
        outputs['Rs'] = Rs_pred
        outputs['ts'] = ts_pred
        assert (len(Rs_pred.shape) == 3 and Rs_pred.shape[1:] == (3, 3)), 'Unexpected shape: {}'.format(Rs_pred.shape)
        assert (len(Rs_gt.shape) == 3 and Rs_gt.shape[1:] == (3, 3)), 'Unexpected shape: {}'.format(Rs_gt.shape)
        assert (len(ts_pred.shape) == 2 and ts_pred.shape[1] == 3), 'Unexpected shape: {}'.format(ts_pred.shape)
        assert (len(ts_gt.shape) == 2 and ts_gt.shape[1] == 3), 'Unexpected shape: {}'.format(ts_gt.shape)
        outputs['cam_centers'] = ts_pred
        outputs['cam_centers_gt'] = ts_gt

        # Correct for ambiguity by finding an optimal similarity transformation:
        Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True) # Align  Rs_fixed, tx_fixed
        outputs['Rs_fixed'] = Rs_fixed
        outputs['ts_fixed'] = ts_fixed
        outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_pred)  # 4,n
        outputs['pts3D_triangulated_fixed'] = None if pts3D_triangulated is None else (similarity_mat @ pts3D_triangulated)

        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            print_out = conf.get_bool('ba.print_out', default=True)
            # The Ceres wrapper receives the initial result of our inference, including 3D points "Xs_ours".
            # NOTE however, that if triangulation = True, this is effectively replaced by a DLT triangulation.
            # Optionally, if repeat = True, a second BA is performed based on a new (!!) DLT triangulation given the resulting cameras from the first one.
            begin_time = time()
            ba_res = ba_functions.euc_ba(xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns),
                                         Xs_our=pts3D_pred.T, Ps=None,
                                         Ns=Ns, repeat=repeat, triangulation=triangulation, return_repro=True, print_out=print_out) #    Rs, ts, Ps, Xs
            ba_time = time() - begin_time
            outputs['ba_time'] = ba_time
            outputs['Rs_ba'] = ba_res['Rs']
            outputs['ts_ba'] = ba_res['ts']
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']
            outputs['ba_converged1'] = ba_res['converged1']
            if repeat:
                outputs['repro_ba_before'] = ba_res['repro_before']
                outputs['repro_ba_middle'] = ba_res['repro_middle']
                outputs['repro_ba_middle_triangulated'] = ba_res['repro_middle_triangulated']
                outputs['repro_ba_after'] = ba_res['repro_after']
                outputs['ba_converged2'] = ba_res['converged2']

            R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                       return_alignment=True)  # Align  Rs_fixed, tx_fixed
            outputs['Rs_ba_fixed'] = R_ba_fixed
            outputs['ts_ba_fixed'] = t_ba_fixed
            outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])

    else:
        # TODO: Is it feasible to align perspective cameras similar to the calibrated case above? (Searching for a projective rather than similarity transformation)
        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            print_out = conf.get_bool('ba.print_out', default=True)
            begin_time = time()
            ba_res = ba_functions.proj_ba(Ps=Ps, xs=xs, Xs_our=pts3D_pred.T, Ns=Ns, repeat=repeat,
                                          triangulation=triangulation, return_repro=True, normalize_in_tri=True, print_out=print_out)   # Ps, Xs
            ba_time = time() - begin_time
            outputs['ba_time'] = ba_time
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']
            outputs['ba_converged1'] = ba_res['converged1']
            if repeat:
                outputs['repro_ba_before'] = ba_res['repro_before']
                outputs['repro_ba_middle'] = ba_res['repro_middle']
                outputs['repro_ba_middle_triangulated'] = ba_res['repro_middle_triangulated']
                outputs['repro_ba_after'] = ba_res['repro_after']
                outputs['ba_converged2'] = ba_res['converged2']

    return outputs


def compute_errors(outputs, conf, bundle_adjustment):
    model_errors = {}

    calibrated = conf.get_bool('dataset.calibrated')
    depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
    view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    xs = outputs['xs']
    visible_points_mask = geo_utils.xs_valid_points(xs)

    if depth_head_enabled:
        # Rescale predicted depths to match GT depths / poses / scene points
        depths_pred_dense_normalized = outputs['depths_pred_dense'] / outputs['s_pred']
        depths_gt_dense_normalized = outputs['depths_gt_dense'] / outputs['s_gt']

        # Log stats of predicted depths
        model_errors['depth_pred_norm_mean'] = depths_pred_dense_normalized[visible_points_mask].mean()
        quantiles = [10, 25, 50, 75, 90]
        for q, x in zip(
            quantiles,
            np.quantile(
                depths_pred_dense_normalized[visible_points_mask],
                0.01 * np.array(quantiles),
            ),
        ):
            model_errors['depth_pred_norm_q{:02d}'.format(q)] = x
        model_errors['depth_pred_norm_min'] = depths_pred_dense_normalized[visible_points_mask].min()
        model_errors['depth_pred_norm_max'] = depths_pred_dense_normalized[visible_points_mask].max()

        # Log stats of GT depths
        model_errors['depth_gt_norm_mean'] = depths_gt_dense_normalized[visible_points_mask].mean()
        quantiles = [10, 25, 50, 75, 90]
        for q, x in zip(
            quantiles,
            np.quantile(
                depths_gt_dense_normalized[visible_points_mask],
                0.01 * np.array(quantiles),
            ),
        ):
            model_errors['depth_gt_norm_q{:02d}'.format(q)] = x
        model_errors['depth_gt_norm_min'] = depths_gt_dense_normalized[visible_points_mask].min()
        model_errors['depth_gt_norm_max'] = depths_gt_dense_normalized[visible_points_mask].max()

        # Calculate mean error of normalized depth prediction (equivalent to DirectDepthLoss with L1 loss)
        model_errors['depth_pred_err_mean'] = np.mean(np.abs(depths_pred_dense_normalized[visible_points_mask] - depths_gt_dense_normalized[visible_points_mask]))

    if calc_reprojerr_with_gtposes_for_depth_pred:
        assert depth_head_enabled
        reproj_errors, reproj_depths = geo_utils.reprojection_error_backproj_random_view_pairs(
            outputs['Ks'], # Calibration matrices
            outputs['Ps_gt'], # GT poses (unnormalized)
            depths_pred_dense_normalized * outputs['s_gt'], # predicted depths, rescaled to match GT
            outputs['xs'],
            visible_points = None, # Infer specified elements from xs
            calc_reproj_depths = True,
        )
        # Rescale the depths to the normalized depth scale
        reproj_depths /= outputs['s_gt']
        model_errors['repro_backproj_rnd_gt_2view'] = np.nanmean(reproj_errors)
        model_errors['repro_backproj_depth_norm_mean_rnd_gt_2view'] = reproj_depths[visible_points_mask].mean()
        model_errors['repro_backproj_depth_norm_min_rnd_gt_2view'] = reproj_depths[visible_points_mask].min()
        model_errors['repro_backproj_depth_norm_max_rnd_gt_2view'] = reproj_depths[visible_points_mask].max()
        quantiles = [10, 25, 50, 75, 90]
        for q, x in zip(
            quantiles,
            np.quantile(
                reproj_depths[visible_points_mask],
                0.01 * np.array(quantiles),
            ),
        ):
            model_errors['repro_backproj_depth_norm_q{:02d}_rnd_gt_2view'.format(q)] = x

    if not explicit_est_avail:
        return model_errors

    Ps = outputs['Ps']
    pts3D_pred = outputs['pts3D_pred']
    pts3D_triangulated = outputs['pts3D_triangulated']

    model_errors["our_repro"] = np.nanmean(geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs))
    model_errors["triangulated_repro"] = np.nan if pts3D_triangulated is None else np.nanmean(geo_utils.reprojection_error_with_points(Ps, pts3D_triangulated.T, xs))
    if calibrated:
        Rs_fixed = outputs['Rs_fixed']
        ts_fixed = outputs['ts_fixed']
        Rs_gt = outputs['Rs_gt']
        ts_gt = outputs['ts_gt']
        Rs_error, ts_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
        model_errors["t_err_mean"] = np.mean(ts_error)
        model_errors["t_err_med"] = np.median(ts_error)
        model_errors["R_err_mean"] = np.mean(Rs_error)
        model_errors["R_err_med"] = np.median(Rs_error)
        assert (len(outputs["cam_centers"].shape) == 2 and outputs["cam_centers"].shape[1] == 3), 'Unexpected shape: {}'.format(outputs["cam_centers"].shape)
        assert (len(outputs["cam_centers_gt"].shape) == 2 and outputs["cam_centers_gt"].shape[1] == 3), 'Unexpected shape: {}'.format(outputs["cam_centers_gt"].shape)
        model_errors["cam_centers_std"] = np.mean(np.linalg.norm(outputs['cam_centers'] - np.mean(outputs['cam_centers'], keepdims=True), axis=1))
        model_errors["cam_centers_gt_std"] = np.mean(np.linalg.norm(outputs['cam_centers_gt'] - np.mean(outputs['cam_centers_gt'], keepdims=True), axis=1))

    if bundle_adjustment:
        Xs_ba = outputs['Xs_ba']
        Ps_ba = outputs['Ps_ba']
        model_errors['repro_ba'] = np.nanmean(geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, xs))
        model_errors['ba_time'] = outputs['ba_time']
        model_errors['ba_converged1'] = 1 if outputs['ba_converged1'] else 0
        if conf.get_bool('ba.repeat'):
            model_errors['repro_ba_before'] = outputs['repro_ba_before']
            model_errors['repro_ba_middle'] = outputs['repro_ba_middle']
            model_errors['repro_ba_middle_triangulated'] = outputs['repro_ba_middle_triangulated']
            model_errors['repro_ba_after'] = outputs['repro_ba_after']
            model_errors['ba_converged2'] = 1 if outputs['ba_converged2'] else 0
        if calibrated:
            Rs_fixed = outputs['Rs_ba_fixed']
            ts_fixed = outputs['ts_ba_fixed']
            Rs_gt = outputs['Rs_gt']
            ts_gt = outputs['ts_gt']
            Rs_ba_error, ts_ba_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
            model_errors["t_err_ba_mean"] = np.mean(ts_ba_error)
            model_errors["t_err_ba_med"] = np.median(ts_ba_error)
            model_errors["R_err_ba_mean"] = np.mean(Rs_ba_error)
            model_errors["R_err_ba_med"] = np.median(Rs_ba_error)
    # Rs errors mean, ts errors mean, ba repro, rs ba mean, ts ba mean

    pts2D_pred = Ps @ pts3D_pred # (m, 3, n)
    positive_projected_pts_mask = geo_utils.get_positive_projected_pts_mask(pts2D_pred, conf.get_float('loss.infinity_pts_margin'))
    visible_but_negative_projected_pts_mask = np.logical_and(~positive_projected_pts_mask, visible_points_mask)

    # n_views, n_pts = visible_points_mask.shape # Would also count views without visible points or points visible in no views
    n_views = np.any(visible_points_mask, axis=1).sum() # #non-zero rows
    n_pts = np.any(visible_points_mask, axis=1).sum() # #non-zero columns
    model_errors['fraction_views_neg_depth_for_any_point'] = np.any(visible_but_negative_projected_pts_mask, axis=1).sum() / n_views
    model_errors['fraction_points_neg_depth_in_any_view'] = np.any(visible_but_negative_projected_pts_mask, axis=0).sum() / n_pts
    model_errors['total_fraction_points_neg_depth'] = visible_but_negative_projected_pts_mask.sum() / visible_points_mask.sum()
    model_errors['point_depth_mean'] = pts2D_pred[:, 2, :][visible_points_mask].mean()
    model_errors['point_depth_min'] = pts2D_pred[:, 2, :][visible_points_mask].min()
    model_errors['point_depth_max'] = pts2D_pred[:, 2, :][visible_points_mask].max()

    return model_errors


def get_dummy_errors(conf, bundle_adjustment):
    model_errors = {}

    calibrated = conf.get_bool('dataset.calibrated')
    depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
    view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    if calc_reprojerr_with_gtposes_for_depth_pred:
        model_errors['repro_backproj_rnd_gt_2view'] = np.nan
        model_errors['repro_backproj_depth_norm_mean_rnd_gt_2view'] = np.nan
        model_errors['repro_backproj_depth_norm_min_rnd_gt_2view'] = np.nan
        model_errors['repro_backproj_depth_norm_max_rnd_gt_2view'] = np.nan
        for q in [10, 25, 50, 75, 90]:
            model_errors['repro_backproj_depth_norm_q{:02d}_rnd_gt_2view'.format(q)] = np.nan

    if depth_head_enabled:
        model_errors['depth_pred_norm_mean'] = np.nan
        model_errors['depth_pred_norm_min'] = np.nan
        model_errors['depth_pred_norm_max'] = np.nan
        for q in [10, 25, 50, 75, 90]:
            model_errors['depth_pred_norm_q{:02d}'.format(q)] = np.nan
        model_errors['depth_gt_norm_mean'] = np.nan
        model_errors['depth_gt_norm_min'] = np.nan
        model_errors['depth_gt_norm_max'] = np.nan
        for q in [10, 25, 50, 75, 90]:
            model_errors['depth_gt_norm_q{:02d}'.format(q)] = np.nan
        model_errors['depth_pred_err_mean'] = np.nan

    if not explicit_est_avail:
        return model_errors

    model_errors["our_repro"] = np.nan
    model_errors["triangulated_repro"] = np.nan
    if calibrated:
        model_errors["t_err_mean"] = np.nan
        model_errors["t_err_med"] = np.nan
        model_errors["R_err_mean"] = np.nan
        model_errors["R_err_med"] = np.nan

    if bundle_adjustment:
        model_errors['repro_ba'] = np.nan
        model_errors['ba_converged1'] = np.nan
        if conf.get_bool('ba.repeat'):
            model_errors['repro_ba_before'] = np.nan
            model_errors['repro_ba_middle'] = np.nan
            model_errors['repro_ba_middle_triangulated'] = np.nan
            model_errors['repro_ba_after'] = np.nan
            model_errors['ba_converged2'] = np.nan
        if calibrated:
            model_errors["t_err_ba_mean"] = np.nan
            model_errors["t_err_ba_med"] = np.nan
            model_errors["R_err_ba_mean"] = np.nan
            model_errors["R_err_ba_med"] = np.nan

    model_errors['fraction_views_neg_depth_for_any_point'] = np.nan
    model_errors['fraction_points_neg_depth_in_any_view'] = np.nan
    model_errors['total_fraction_points_neg_depth'] = np.nan
    model_errors['point_depth_mean'] = np.nan
    model_errors['point_depth_min'] = np.nan
    model_errors['point_depth_max'] = np.nan

    return model_errors
