import sys
import os
import traceback
import time
import torch
import numpy as np
import math

import loss_functions
import evaluation
import copy
from datasets import SceneData, ScenesDataSet
from datasets.ScenesDataSet import dataloader_collate_fn as collate_fn
from torch.utils.data import DataLoader
from utils import path_utils, dataset_utils, plot_utils, general_utils
from utils.general_utils import get_additional_identifiers_for_outlier_injection
from time import time
import pandas as pd
from utils.Phases import Phases


def tb_log_train_step(tb_writer, batch_idx, signal_name, signal_val, phase, additional_identifiers=[], scene=None):
    assert all(isinstance(x, str) for x in additional_identifiers)
    if phase == Phases.TRAINING:
        if scene is None:
            main_tag_id = '{}-all-scenes'.format(phase.name)
        else:
            main_tag_id = '{}-per-scene'.format(phase.name)
    else:
        assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
        assert scene is not None
        main_tag_id = '{}-train'.format(phase.name)

    tag = [main_tag_id]
    tag += additional_identifiers
    if scene is not None:
        tag.append(''.join(scene.split()))
    tag.append('batch')
    tag.append(signal_name)
    tag = '/'.join(tag) # Serialize

    tb_writer.add_scalar(
        tag,
        signal_val,
        global_step = batch_idx+1,
    )


def epoch_train(conf, device, train_loader, model, loss_func, optimizer, scheduler, epoch, phase, tb_writer, outlier_injection_rate=None, additional_identifiers=[], scene=None, prev_n_batches=0, tb_log_train_per_scene=True):
    grad_clip_mode = conf.get_float('loss.grad_clip_mode', default=None)
    grad_clip_th = conf.get_float('loss.grad_clip_th', default=None)
    tb_log_memory_consumption = conf.get_bool('memory.tb_log_training_memory_consumption', default=True)
    view_head_enabled = conf.get_bool('model.view_head.enabled')
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled')
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    model.train()
    train_losses = []
    for batch_idx, train_batch in enumerate(train_loader):  # Loop over all sets - 30
        batch_loss = torch.tensor([0.0], device=device)
        if explicit_est_avail:
            batch_mean_repro = 0.0
        if calc_reprojerr_with_gtposes_for_depth_pred:
            batch_mean_repro_backproj = 0.0
        optimizer.zero_grad()
        for curr_data in train_batch:
            curr_data = curr_data.to(device, dense_on_demand=True) # Request for the SceneData instance to transfer its data to device
            if not dataset_utils.is_valid_sample(curr_data):
                # print('{} {} has a camera with not enough points'.format(epoch, curr_data.scene_name))
                print('{} {} has a camera with not enough points or a point with not enough cameras'.format(epoch, curr_data.scene_name))
                continue
            if outlier_injection_rate is None:
                pred_dict = model(curr_data)
            else:
                curr_data_outlier_injected = dataset_utils.inject_outliers(curr_data, outlier_injection_rate)
                if curr_data_outlier_injected is None:
                    # There are not enough free points available to achieve the desired outlier rate
                    print('Failed outlier sampling for {} - skipping training sample.'.format(curr_data.scene_name))
                    continue
                pred_dict = model(curr_data_outlier_injected)
            if not curr_data.valid_pts.is_cuda:
                print(device)
                print(curr_data.device)
                print(curr_data.valid_pts.device)
                assert False, 'Unexpected CUDA device'
            loss = loss_func(pred_dict, curr_data)
            batch_loss += loss
            train_losses.append(loss.item())

            core_errors = evaluation.compute_core_errors(curr_data, pred_dict, conf)
            if explicit_est_avail:
                batch_mean_repro += core_errors['our_repro']
            if calc_reprojerr_with_gtposes_for_depth_pred:
                batch_mean_repro_backproj += core_errors['repro_backproj_rnd_gt_2view']
        if explicit_est_avail:
            batch_mean_repro /= len(train_batch)
        if calc_reprojerr_with_gtposes_for_depth_pred:
            batch_mean_repro_backproj /= len(train_batch)

        # Log batch loss & reprojection error
        if phase == Phases.TRAINING:
            tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'loss', batch_loss.item(), phase, additional_identifiers=additional_identifiers)
            if explicit_est_avail:
                tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'our_repro', batch_mean_repro, phase, additional_identifiers=additional_identifiers)
            if calc_reprojerr_with_gtposes_for_depth_pred:
                tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'repro_backproj_rnd_gt_2view', batch_mean_repro_backproj, phase, additional_identifiers=additional_identifiers)
            if tb_log_train_per_scene:
                tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'loss', batch_loss.item(), phase, additional_identifiers=additional_identifiers, scene=curr_data.scene_name)
                if explicit_est_avail:
                    tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'our_repro', batch_mean_repro, phase, additional_identifiers=additional_identifiers, scene=curr_data.scene_name)
                if calc_reprojerr_with_gtposes_for_depth_pred:
                    tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'repro_backproj_rnd_gt_2view', batch_mean_repro_backproj, phase, additional_identifiers=additional_identifiers, scene=curr_data.scene_name)
        else:
            assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
            tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'loss', batch_loss.item(), phase, additional_identifiers=additional_identifiers, scene=scene)
            if explicit_est_avail:
                tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'our_repro', batch_mean_repro, phase, additional_identifiers=additional_identifiers, scene=scene)
            if calc_reprojerr_with_gtposes_for_depth_pred:
                tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'repro_backproj_rnd_gt_2view', batch_mean_repro_backproj, phase, additional_identifiers=additional_identifiers, scene=scene)

        # Log memory consumption
        if tb_log_memory_consumption:
            tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'memory_allocated', torch.cuda.memory_allocated(), phase, additional_identifiers=additional_identifiers, scene=scene)
            tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'memory_reserved', torch.cuda.memory_reserved(), phase, additional_identifiers=additional_identifiers, scene=scene)

        # Log LR
        curr_lr = scheduler.get_last_lr()
        assert len(curr_lr) == 1
        curr_lr = curr_lr[0]
        tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'learning_rate', curr_lr, phase, additional_identifiers=additional_identifiers, scene=scene)

        if batch_loss.item()>0:
            batch_loss.backward()

            # Log gradient magnitude
            grad = torch.cat([ p.grad.flatten() for p in model.parameters() ], dim=0)
            grad_norm = grad.norm()
            tb_log_train_step(tb_writer, prev_n_batches+batch_idx, 'grad_norm', grad_norm, phase, additional_identifiers=additional_identifiers, scene=scene)

            # Gradient clipping
            if grad_clip_mode is not None:
                assert grad_clip_th is not None
                if grad_clip_mode == 'norm':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_th)
                elif grad_clip_mode == 'value':
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_th)
                else:
                    assert False, 'Could not interpret gradient clipping mode "{}".'.format(grad_clip_mode)

            optimizer.step()
        scheduler.step()

    n_batches = batch_idx + 1

    mean_loss = torch.tensor(train_losses).mean()
    return mean_loss, train_losses, n_batches


def eval_errors_list2df(errors_list):
    df_errors = pd.DataFrame(errors_list)
    df_errors.set_index("Scene", inplace=True)
    mean_errors = df_errors.mean(axis=0, numeric_only=True).to_frame(name="Mean").T
    mean_errors.index.name = 'Scene' # Index name lost when taking the mean above
    df_errors = pd.concat([df_errors, mean_errors], axis=0)
    print(df_errors.round(3).to_string(), flush=True)
    assert df_errors.index.name == 'Scene'
    return df_errors

def epoch_evaluation(data_loader, model, conf, device, epoch, phase, outlier_injection_rate=None, dump_and_plot_predictions=False, additional_identifiers=[], bundle_adjustment=True, log_memory_consumption=False, crash_on_scene_exhausting_memory=True):
    view_head_enabled = conf.get_bool('model.view_head.enabled')
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled')
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled

    errors_list = []
    model.eval()
    with torch.no_grad():
        for j, batch_data in enumerate(data_loader):
            if log_memory_consumption:
                print('Scene batch {}/{}.'.format(j+1, len(data_loader)))
            for curr_data in batch_data:
                if log_memory_consumption:
                    print("Memory before eval on {}: Allocated {} MiB, reserved {} MiB.".format(curr_data.scene_name, torch.cuda.memory_allocated()//2**20, torch.cuda.memory_reserved()//2**20))

                try:
                    curr_data = curr_data.to(device, dense_on_demand=True) # Request for the SceneData instance to transfer its data to device

                    # Get predictions
                    if outlier_injection_rate is None:
                        begin_time = time()
                        pred_dict = model(curr_data)
                    else:
                        # if not dataset_utils.is_valid_sample(curr_data):
                        #     # print('{} [EVAL] {} has a camera with not enough points'.format(epoch, curr_data.scene_name))
                        #     print('{} [EVAL] {} has a camera with not enough points or a point with not enough cameras'.format(epoch, curr_data.scene_name))
                        #     continue
                        curr_data_outlier_injected = dataset_utils.inject_outliers(curr_data, outlier_injection_rate)
                        assert curr_data_outlier_injected is not None
                        # if curr_data_outlier_injected is None:
                        #     # There are not enough free points available to achieve the desired outlier rate
                        #     print('[EVAL] Failed outlier sampling for {} - skipping evaluation.'.format(curr_data.scene_name))
                        #     continue
                        begin_time = time()
                        pred_dict = model(curr_data_outlier_injected)
                    pred_time = time() - begin_time

                    # Eval results
                    outputs = evaluation.prepare_predictions(curr_data, pred_dict, conf, bundle_adjustment)
                    errors = evaluation.compute_errors(outputs, conf, bundle_adjustment)

                    errors['Inference time'] = pred_time
                    errors['Scene'] = curr_data.scene_name

                    # Get scene statistics on best evaluation
                    if epoch is None:
                        stats = dataset_utils.get_data_statistics(curr_data)
                        errors.update(stats)

                    if dump_and_plot_predictions:
                        dataset_utils.dump_predictions(outputs, conf, curr_epoch=epoch, phase=phase, additional_identifiers=additional_identifiers)
                        if conf.get_bool('dataset.calibrated') and explicit_est_avail:
                            # We only plot scene + cameras if we are doing Euclidean reconstruction, as the result is otherwise deemed nonsensical due to projective ambiguity / distorsion.
                            # In theory, however, would it not be possible to do a projective registration between predicted & GT poses, similar to the similarity registration done in the calibrated case..?
                            path = plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scene=curr_data.scene_name, epoch=epoch, bundle_adjustment=bundle_adjustment, additional_identifiers=additional_identifiers)
                except (torch.cuda.OutOfMemoryError, ValueError) as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        # Out-of-memory error.
                        pass
                    else:
                        # Determine origin of error (where it was raised):
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        filename, line_num, func_name, text = traceback.extract_tb(exc_tb)[-1]
                        if os.path.basename(filename) == 'message_passing.py' and func_name == '__lift__':
                            # Not unlikely to be an out-of-memory error, but we can not be sure - it may also be some other error raised in PyTorch Geometric code.
                            # In older versions of PyTorch Geometric, CUDA errors were sometimes caught and not preserved and reraised, but rather another error was raised.
                            # This has now been fixed, but not yet released at the time of writing: (latest release 2.2.0)
                            # https://github.com/pyg-team/pytorch_geometric/pull/6417
                            pass
                        else:
                            # Unknown error - do not catch it.
                            raise e
                    if crash_on_scene_exhausting_memory:
                        raise e
                    else:
                        print('Ran out of memory when evaluating on {}.'.format(curr_data.scene_name))
                        errors = evaluation.get_dummy_errors(conf, bundle_adjustment)
                        errors['Inference time'] = np.nan
                        errors['Scene'] = curr_data.scene_name

                errors_list.append(errors)

                if log_memory_consumption:
                    print("Memory after eval on {}: Allocated {} MiB, reserved {} MiB.".format(curr_data.scene_name, torch.cuda.memory_allocated()//2**20, torch.cuda.memory_reserved()//2**20))

    df_errors = eval_errors_list2df(errors_list)

    model.train()

    return df_errors


def aggregate_val_metric(validation_errors, metric_column=None, scene=None):
    assert metric_column is not None
    assert isinstance(metric_column, str)
    if scene is None:
        # If a scene name is not provided, extract the mean result.
        scene = "Mean"
    val_metric = validation_errors.loc[[scene], [metric_column]].values.item()
    return val_metric


def tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=[], scene=None, include_post_ba_metrics=False):
    assert all(isinstance(x, str) for x in additional_identifiers)
    depth_head_enabled = conf.get_bool('model.depth_head.enabled', default=False)
    view_head_enabled = conf.get_bool('model.view_head.enabled', default=False)
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled', default=False)
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    metric_columns = []
    if calc_reprojerr_with_gtposes_for_depth_pred:
        metric_columns += [
            'repro_backproj_rnd_gt_2view',
            'repro_backproj_depth_norm_mean_rnd_gt_2view',
            'repro_backproj_depth_norm_min_rnd_gt_2view',
            'repro_backproj_depth_norm_max_rnd_gt_2view',
        ]
        for q in [10, 25, 50, 75, 90]:
            metric_columns.append('repro_backproj_depth_norm_q{:02d}_rnd_gt_2view'.format(q))
    if depth_head_enabled:
        metric_columns += [
            'depth_pred_norm_mean',
            'depth_pred_norm_min',
            'depth_pred_norm_max',
        ]
        for q in [10, 25, 50, 75, 90]:
            metric_columns.append('depth_pred_norm_q{:02d}'.format(q))
        metric_columns += [
            'depth_gt_norm_mean',
            'depth_gt_norm_min',
            'depth_gt_norm_max',
        ]
        for q in [10, 25, 50, 75, 90]:
            metric_columns.append('depth_gt_norm_q{:02d}'.format(q))
        metric_columns += [
            'depth_pred_err_mean',
        ]
    if explicit_est_avail:
        metric_columns += [
            'our_repro',
            'triangulated_repro',
        ]
        if conf.get_bool('dataset.calibrated'):
            metric_columns += [
                't_err_mean',
                't_err_med',
                'R_err_mean',
                'R_err_med',
                'cam_centers_std',
                'cam_centers_gt_std',
            ]
        if include_post_ba_metrics:
            metric_columns += [
                'repro_ba',
            ]
            if conf.get_bool('dataset.calibrated'):
                metric_columns += [
                    't_err_ba_mean',
                    't_err_ba_med',
                    'R_err_ba_mean',
                    'R_err_ba_med',
                ]
        metric_columns += [
            'fraction_views_neg_depth_for_any_point',
            'fraction_points_neg_depth_in_any_view',
            'total_fraction_points_neg_depth',
            'point_depth_mean',
            'point_depth_min',
            'point_depth_max',
        ]
    for metric_column in metric_columns:
        if phase == Phases.VALIDATION:
            if scene is None:
                main_tag_id = '{}-scene-avg'.format(phase.name)
            else:
                main_tag_id = '{}-per-scene'.format(phase.name)
        elif phase == Phases.TRAINING:
            if scene is None:
                main_tag_id = '{}-eval-scene-avg'.format(phase.name)
            else:
                main_tag_id = '{}-eval-per-scene'.format(phase.name)
        else:
            assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
            assert scene is not None
            main_tag_id = '{}-eval'.format(phase.name)

        tag = [main_tag_id]
        tag += additional_identifiers
        if scene is not None:
            tag.append(''.join(scene.split()))
        tag.append('epoch')
        tag.append(metric_column)
        tag = '/'.join(tag) # Serialize

        tb_writer.add_scalar(
            tag,
            aggregate_val_metric(validation_errors, metric_column=metric_column, scene=scene),
            global_step = epoch+1,
        )


def train(conf, device, train_loader, model, phase, train_loader_for_eval=None, val_loader=None, test_loader=None, additional_identifier=None):
    additional_identifiers = [] if additional_identifier is None else [additional_identifier]
    n_epochs = conf.get_int('train.n_epochs')
    sequentially_increment_views = False if phase == Phases.TRAINING else conf.get_bool("train.sequentially_increment_views", default=False)
    outlier_injection_rate = conf.get_float('train.outlier_injection_rate', default=None)
    print_interval = conf.get_int('train.print_interval', default=None)
    eval_interval = conf.get_int('eval.eval_interval', default=500)
    finetune_dump_model_interval = conf.get_int('train.finetune_dump_model_interval', default=None)
    finetune_dump_and_plot_pred_interval = conf.get_int('train.finetune_dump_and_plot_pred_interval', default=None)
    stdout_log_eval_memory_consumption = conf.get_bool('memory.stdout_log_eval_memory_consumption', default=True)
    depth_head_enabled = conf.get_bool('model.depth_head.enabled')
    view_head_enabled = conf.get_bool('model.view_head.enabled')
    scenepoint_head_enabled = conf.get_bool('model.scenepoint_head.enabled')
    explicit_est_avail = view_head_enabled and scenepoint_head_enabled
    calc_reprojerr_with_gtposes_for_depth_pred = conf.get_bool('eval.calc_reprojerr_with_gtposes_for_depth_pred')

    # Copy model, and move to GPU memory
    assert not general_utils.any_model_parameters_on_gpu(model) # If true, we would store duplicate parameters on GPU
    model = copy.deepcopy(model) # Make sure to train a copy of the model, preserving the original parameters as-is, in case they are to be used for other purposes outside of this function.
    model.to(device)

    if phase == Phases.TRAINING:
        if conf.get_bool('eval.eval_on_train_set', default=False):
            assert train_loader_for_eval is not None
        validation_metric = conf.get_string('train.validation_metric', default=None)
        if validation_metric is None:
            if explicit_est_avail:
                validation_metric = "our_repro"
            elif depth_head_enabled:
                validation_metric = "repro_backproj_rnd_gt_2view"
        tb_log_train_per_scene = conf.get_bool('train.tb_log_train_per_scene')
        tb_log_val_per_scene = conf.get_bool('train.tb_log_val_per_scene')
    else:
        assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
        if train_loader_for_eval is None:
            train_loader_for_eval = train_loader
        assert len(train_loader) == 1 # Only one batch expected in optimization mode.
        the_batch = next(iter(train_loader))
        dataloader_num_workers = train_loader.num_workers
        assert len(the_batch) == 1 # And only one sample in that only batch.
        if sequentially_increment_views:
            increment_views_interval = conf.get_int("train.increment_views_interval")
            # Initialize a variable for the full scene data, from which we will later extract new data samples with a certain number of views.
            fullscene_data = the_batch[0]
            total_n_views = fullscene_data.y.shape[0]
            # Initialize the view counts, which will be updated every "increment_views_interval" epochs:
            prev_n_views = None
            curr_n_views = None
            n_epochs_sequential = (total_n_views - 1) * increment_views_interval
            # Modify configured #epochs and LR scheduling milestones to refer to the post-sequential optimization phase.
            n_epochs += n_epochs_sequential

    tb_writer = general_utils.get_tb_writer(conf)

    # The only phase for which we expect to receive validation / test data is during TRAINING (=multi-scene learning).
    # Verify that these dataloaders are indeed received iff we are in the TRAINING phase.
    assert (phase == Phases.TRAINING) == (val_loader is not None)
    assert (phase == Phases.TRAINING) == (test_loader is not None)

    # Loss functions
    loss_func = loss_functions.get_loss_func(conf)

    # Optimizer params
    lr = conf.get_float('train.lr')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    main_scheduler = conf.get_string('train.lr_schedule.main_scheduler')
    lr_warmup_n_steps = conf.get_int('train.lr_schedule.lr_warmup_n_steps', default=0)

    if main_scheduler == 'constant':
        lr_scheduler_main = None
    elif main_scheduler == 'exponential':
        exp_gamma_after_n_steps = conf.get_float('train.lr_schedule.exp_gamma_after_n_steps')
        exp_n_steps = conf.get_float('train.lr_schedule.exp_n_steps')
        gamma = exp_gamma_after_n_steps ** (1.0 / exp_n_steps)
        lr_scheduler_main = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif main_scheduler == 'multistep':
        multistep_gamma = conf.get_float('train.lr_schedule.multistep_gamma', default=0.1)
        multistep_milestones = conf.get_list('train.lr_schedule.multistep_milestones')
        if sequentially_increment_views:
            multistep_milestones = [ epoch + n_epochs_sequential for epoch in multistep_milestones ]
        lr_scheduler_main = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=multistep_milestones, gamma=multistep_gamma)
    else:
        raise NotImplementedError('Unknown LR scheduler: {}'.format(main_scheduler))
    lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1.0 / (lr_warmup_n_steps+1),
        # NOTE! Important to not use any other end_factor than 1.0, without making sure we obtain expected behavior.
        # SequentialLR doesn't seem to guarantee "LR stitching", but instead, each scheduler resets at the same "base_lr".
        end_factor = 1.0,
        total_iters = lr_warmup_n_steps,
    )
    if lr_scheduler_main is not None:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = [lr_scheduler_warmup, lr_scheduler_main],
            milestones = [lr_warmup_n_steps],
        )
    else:
        scheduler = lr_scheduler_warmup

    if phase == Phases.TRAINING and validation_metric is not None:
        # Initialize early stopping variables: best_validation_metric, best_model, best_epoch, converge_time
        best_validation_metric = math.inf
        best_model = None
        best_epoch = -1
        converge_time = -1
        begin_time = time()

    run_ba = conf.get_bool('ba.run_ba', default=True)
    ba_during_training = run_ba and not conf.get_bool('ba.only_last_eval')

    # Always executed before fine-tuning, since the initial model is not random but a viable candidate.
    if conf.get_bool('eval.eval_init', default=False) or phase == Phases.FINE_TUNE:
        epoch = -1
        dump_and_plot_predictions = finetune_dump_and_plot_pred_interval is not None
        if phase == Phases.TRAINING:
            # Evaluate on validation set
            validation_errors = epoch_evaluation(val_loader, model, conf, device, epoch, Phases.VALIDATION, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
            tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=None, include_post_ba_metrics=ba_during_training)
            if tb_log_val_per_scene:
                for scene in conf.get_list('dataset.validation_set'):
                    tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
            if outlier_injection_rate is not None:
                # Extra outlier-free validation
                validation_errors = epoch_evaluation(val_loader, model, conf, device, epoch, Phases.VALIDATION, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers, scene=None, include_post_ba_metrics=ba_during_training)
                if tb_log_val_per_scene:
                    for scene in conf.get_list('dataset.validation_set'):
                        tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)

            if conf.get_bool('eval.eval_on_train_set', default=False):
                # Evaluate on training set
                train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, Phases.TRAINING, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=None, include_post_ba_metrics=ba_during_training)
                if tb_log_train_per_scene:
                    for scene in conf.get_list('dataset.train_set'):
                        tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
                if outlier_injection_rate is not None:
                    # Extra outlier-free validation
                    train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, Phases.TRAINING, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                    tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers, scene=None, include_post_ba_metrics=ba_during_training)
                    if tb_log_train_per_scene:
                        for scene in conf.get_list('dataset.train_set'):
                            tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)
        else:
            assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
            scene = conf.get_string('dataset.scene')
            # Evaluation on the training set (=a single scene in this case).
            train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, phase, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
            tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=phase, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
            validation_errors = train_errors
            if outlier_injection_rate is not None:
                # Extra outlier-free validation
                train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, phase, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=phase, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)

        if phase == Phases.TRAINING and validation_metric is not None:
            metric = aggregate_val_metric(validation_errors, metric_column=validation_metric)

            if metric < best_validation_metric:
                # This should almost surely happen, i.e. an improvement w.r.t. math.inf.
                best_validation_metric = metric
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                print('Updated best validation metric: {}'.format(best_validation_metric))

                # Save model
                path = os.path.join(path_utils.path_to_models_dir(conf, phase, additional_identifiers=additional_identifiers), 'best_model.pt')
                torch.save(model.state_dict(), path)

        # Save model
        if finetune_dump_model_interval is not None:
            path = os.path.join(path_utils.path_to_models_dir(conf, phase, additional_identifiers=additional_identifiers), 'model_epoch{:06d}.pt'.format(epoch+1))
            torch.save(model.state_dict(), path)


    total_n_batches = 0
    n_epochs_post_warmup = None if lr_warmup_n_steps > 0 else 0
    for epoch in range(n_epochs):
        if phase == Phases.TRAINING:
            # The scene is not unique during learning.
            scene = None
            # Keep dataloader unchanged.
            curr_train_loader = train_loader
        else:
            assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
            scene = conf.get_string('dataset.scene')

            if sequentially_increment_views:
                prev_n_views = curr_n_views
                curr_n_views = 2 + n_epochs_post_warmup // increment_views_interval if n_epochs_post_warmup is not None else 2
                if curr_n_views >= total_n_views:
                    curr_train_loader = train_loader
                elif curr_n_views != prev_n_views:
                    # The number of views has changed. Switch from one subset of views to another.
                    print('Updating #views: {} -> {}'.format(prev_n_views, curr_n_views))
                    subscene_data = SceneData.get_subset(fullscene_data, curr_n_views)
                    subscene_dataset = ScenesDataSet.ScenesDataSet([subscene_data], return_all=True)
                    curr_train_loader = DataLoader(subscene_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng)
            else:
                # Not in sequential mode - keep dataloader unchanged.
                curr_train_loader = train_loader

        mean_train_loss, train_losses, n_batches = epoch_train(conf, device, curr_train_loader, model, loss_func, optimizer, scheduler, epoch, phase, tb_writer, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, prev_n_batches=total_n_batches, tb_log_train_per_scene=tb_log_train_per_scene if phase==Phases.TRAINING else None)
        total_n_batches += n_batches

        if n_epochs_post_warmup is not None:
            n_epochs_post_warmup += 1
        elif total_n_batches >= lr_warmup_n_steps:
                n_epochs_post_warmup = 0

        if print_interval is not None and epoch % print_interval == 0:
            print('{} Train Loss: {}'.format(epoch, mean_train_loss))
        if (epoch+1) % eval_interval == 0 or epoch == 0 or epoch == n_epochs - 1:  # Eval current results
            dump_and_plot_predictions = finetune_dump_and_plot_pred_interval is not None and (epoch+1) % finetune_dump_and_plot_pred_interval == 0
            if phase == Phases.TRAINING:
                # Evaluate on validation set
                validation_errors = epoch_evaluation(val_loader, model, conf, device, epoch, Phases.VALIDATION, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=None, include_post_ba_metrics=ba_during_training)
                if tb_log_val_per_scene:
                    for scene in conf.get_list('dataset.validation_set'):
                        tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
                if outlier_injection_rate is not None:
                    # Extra outlier-free validation
                    validation_errors = epoch_evaluation(val_loader, model, conf, device, epoch, Phases.VALIDATION, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                    tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers, scene=None, include_post_ba_metrics=ba_during_training)
                    if tb_log_val_per_scene:
                        for scene in conf.get_list('dataset.validation_set'):
                            tb_log_eval_step(conf, tb_writer, epoch, validation_errors, phase=Phases.VALIDATION, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)

                if conf.get_bool('eval.eval_on_train_set', default=False):
                    # Evaluate on training set
                    train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, Phases.TRAINING, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                    tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=None, include_post_ba_metrics=ba_during_training)
                    if tb_log_train_per_scene:
                        for scene in conf.get_list('dataset.train_set'):
                            tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
                    if outlier_injection_rate is not None:
                        # Extra outlier-free validation
                        train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, Phases.TRAINING, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                        tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers, scene=None, include_post_ba_metrics=ba_during_training)
                        if tb_log_train_per_scene:
                            for scene in conf.get_list('dataset.train_set'):
                                tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=Phases.TRAINING, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)
            else:
                # NOTE: During optimization, we carry out evaluation every train.finetune_eval_interval = 250 epochs.
                # Furthermore, no matter the interval, it should always be done after the very first, as well as the very last epoch.
                # Also note that in this case, the training loop is called from single-scene optimization, and the phase is one of FINE_TUNE / SHORT_OPTIMIZATION / OPTIMIZATION.
                # In this case, we don't provide any validation / test data arguments.
                # Instead, we may run an evaluation on the training set (=a single scene in this case).
                assert phase in [Phases.FINE_TUNE, Phases.SHORT_OPTIMIZATION, Phases.OPTIMIZATION]
                train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, phase, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=phase, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), scene=scene, include_post_ba_metrics=ba_during_training)
                validation_errors = train_errors
                if outlier_injection_rate is not None:
                    # Extra outlier-free validation
                    train_errors = epoch_evaluation(train_loader_for_eval, model, conf, device, epoch, phase, outlier_injection_rate=None, dump_and_plot_predictions=dump_and_plot_predictions, additional_identifiers=additional_identifiers, bundle_adjustment=ba_during_training, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=True)
                    tb_log_eval_step(conf, tb_writer, epoch, train_errors, phase=phase, additional_identifiers=additional_identifiers, scene=scene, include_post_ba_metrics=ba_during_training)

            if epoch == n_epochs - 1:
                final_model = copy.deepcopy(model)
                final_model.cpu()

            if phase == Phases.TRAINING and validation_metric is not None:
                metric = aggregate_val_metric(validation_errors, metric_column=validation_metric)

                if epoch == n_epochs - 1:
                    final_validation_metric = metric

                if metric < best_validation_metric:
                    converge_time = time()-begin_time
                    best_validation_metric = metric
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    best_model.cpu() # No need to store these checkpoint parameters on GPU throughout the training.
                    print('Updated best validation metric: {} time so far: {}'.format(best_validation_metric, converge_time))

            if any([
                finetune_dump_model_interval is not None and (epoch+1) % finetune_dump_model_interval == 0,
                phase == Phases.TRAINING and validation_metric is not None and epoch == best_epoch,
            ]): # Save a model checkpoint only when we have seen an improvement in validation metric.
                path = os.path.join(path_utils.path_to_models_dir(conf, phase, additional_identifiers=additional_identifiers), 'model_epoch{:06d}.pt'.format(epoch+1))
                torch.save(model.state_dict(), path)


    model.cpu() # After training, no longer any need to store the continuously updated model parameters in GPU memory.

    # Verify that no instances of the model parameters are currently stored in GPU memory.
    # Instead, they are to be moved to GPU memory one at a time, when needed.
    assert not general_utils.any_model_parameters_on_gpu(model)
    if phase == Phases.TRAINING and validation_metric is not None:
        assert not general_utils.any_model_parameters_on_gpu(best_model)
    assert not general_utils.any_model_parameters_on_gpu(final_model)

    trained_models = {}

    # Saving the final model
    trained_models['final_model'] = final_model
    path = os.path.join(path_utils.path_to_models_dir(conf, phase, additional_identifiers=additional_identifiers), 'final_model.pt')
    torch.save(final_model.state_dict(), path)

    if phase == Phases.TRAINING and validation_metric is not None:
        # Saving the best model
        trained_models['best_model'] = best_model
        path = os.path.join(path_utils.path_to_models_dir(conf, phase, additional_identifiers=additional_identifiers), 'best_model.pt')
        torch.save(best_model.state_dict(), path)

    if phase == Phases.TRAINING and validation_metric is not None:
        train_stats = {}
        train_stats['Convergence time'] = converge_time
        train_stats['best_epoch'] = best_epoch+1
        train_stats['best_validation_metric'] = best_validation_metric
        train_stats['final_validation_metric'] = final_validation_metric
        train_stats = pd.DataFrame([train_stats])
    else:
        train_stats = get_dummy_train_stats()

    return trained_models, train_stats

def get_dummy_train_stats():
    train_stats = {}
    train_stats['Convergence time'] = np.nan
    train_stats['best_epoch'] = np.nan
    train_stats['best_validation_metric'] = np.nan
    train_stats['final_validation_metric'] = np.nan
    train_stats = pd.DataFrame([train_stats])
    return train_stats
