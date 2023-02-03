import sys
import os
import traceback
import numpy as np
from datasets import SceneData, ScenesDataSet
from datasets.ScenesDataSet import dataloader_collate_fn as collate_fn
import train
import evaluation
from utils import general_utils, path_utils
from utils.general_utils import get_additional_identifiers_for_outlier_injection
import torch
from torch.utils.data import DataLoader


def train_model_single_scene(conf, model, device, phase, additional_identifier=None, crash_on_scene_exhausting_memory=True):
    additional_identifiers = [] if additional_identifier is None else [additional_identifier]
    outlier_injection_rate = conf.get_float('train.outlier_injection_rate', default=None)
    dataloader_num_workers = conf.get_int('dataset.dataloader_num_workers')
    run_ba = conf.get_bool('ba.run_ba', default=True)
    stdout_log_eval_memory_consumption = conf.get_bool('memory.stdout_log_eval_memory_consumption', default=True)
    post_train_eval_no_crash_on_scene_exhausting_memory = conf.get_bool('memory.post_train_eval_no_crash_on_scene_exhausting_memory', default=True)

    assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.

    # Create data
    scene_data = SceneData.create_scene_data(conf)

    # Optimize Scene
    scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
    scene_loader = DataLoader(scene_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng)
    trained_models, train_stats = train.train(conf, device, scene_loader, model, phase, additional_identifier=additional_identifier)

    # Run evaluation
    try:
        trained_models['final_model'].to(device)
        final_train_errors = train.epoch_evaluation(scene_loader, trained_models['final_model'], conf, device, -1, phase, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=True, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
        if outlier_injection_rate is not None:
            # Extra outlier-free validation
            final_train_errors_outlierfree = train.epoch_evaluation(scene_loader, trained_models['final_model'], conf, device, -1, phase, dump_and_plot_predictions=True, additional_identifiers=additional_identifiers, bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
        trained_models['final_model'].cpu()

        if conf.get_string('train.validation_metric', default=None) is not None:
            assert 'best_model' in trained_models
            trained_models['best_model'].to(device)
            best_train_errors = train.epoch_evaluation(scene_loader, trained_models['best_model'], conf, device, None, phase, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=True, additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
            if outlier_injection_rate is not None:
                # Extra outlier-free validation
                best_train_errors_outlierfree = train.epoch_evaluation(scene_loader, trained_models['best_model'], conf, device, None, phase, dump_and_plot_predictions=True, additional_identifiers=additional_identifiers, bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
            trained_models['best_model'].cpu()
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
            print('Ran out of memory when fine-tuning on {}.'.format(scene_data.scene_name))
            errors = evaluation.get_dummy_errors(conf, conf.get_bool('ba.run_ba', default=True))
            errors['Inference time'] = np.nan
            errors['Scene'] = scene_data.scene_name
            errors_list = [errors]
            final_train_errors = train.eval_errors_list2df(errors_list)
            if conf.get_string('train.validation_metric', default=None) is not None:
                best_train_errors = train.eval_errors_list2df(errors_list)
            train_stats = train.get_dummy_train_stats()

    # Write results of final model
    final_train_errors.drop("Mean", inplace=True) # Drop mean column (there is only one scene)
    # Augment the dataframe columns with the train stats.
    # First, determine the (one and only) scene, and add a corresponding index for the train_stats DF, to aid the merge.
    train_stats["Scene"] = final_train_errors.index
    train_stats.set_index("Scene", inplace=True)
    train_res = final_train_errors.join(train_stats) # Add the new columns
    # Write this (single-row) dataframe to .xls file.
    # It may exist already, in which case we append a new row with the result for this scene.
    general_utils.write_results(conf, train_res.round(3), file_name="final_train_errors_{}".format(phase.name), additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), append=True)
    if outlier_injection_rate is not None:
        # Extra outlier-free validation
        final_train_errors_outlierfree.drop("Mean", inplace=True) # Drop mean column (there is only one scene)
        # Augment the dataframe columns with the train stats.
        # First, determine the (one and only) scene, and add a corresponding index for the train_stats DF, to aid the merge.
        train_stats["Scene"] = final_train_errors_outlierfree.index
        train_stats.set_index("Scene", inplace=True)
        train_res = final_train_errors_outlierfree.join(train_stats) # Add the new columns
        # Write this (single-row) dataframe to .xls file.
        # It may exist already, in which case we append a new row with the result for this scene.
        general_utils.write_results(conf, train_res.round(3), file_name="final_train_errors_{}".format(phase.name), additional_identifiers=additional_identifiers, append=True)

    # Write results of best model
    if conf.get_string('train.validation_metric', default=None) is not None:
        best_train_errors.drop("Mean", inplace=True) # Drop mean column (there is only one scene)
        # Augment the dataframe columns with the train stats.
        # First, determine the (one and only) scene, and add a corresponding index for the train_stats DF, to aid the merge.
        train_stats["Scene"] = best_train_errors.index
        train_stats.set_index("Scene", inplace=True)
        train_res = best_train_errors.join(train_stats) # Add the new columns
        # Write this (single-row) dataframe to .xls file.
        # It may exist already, in which case we append a new row with the result for this scene.
        general_utils.write_results(conf, train_res.round(3), file_name="best_train_errors_{}".format(phase.name), additional_identifiers=additional_identifiers+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), append=True)
        if outlier_injection_rate is not None:
            # Extra outlier-free validation
            best_train_errors_outlierfree.drop("Mean", inplace=True) # Drop mean column (there is only one scene)
            # Augment the dataframe columns with the train stats.
            # First, determine the (one and only) scene, and add a corresponding index for the train_stats DF, to aid the merge.
            train_stats["Scene"] = best_train_errors_outlierfree.index
            train_stats.set_index("Scene", inplace=True)
            train_res = best_train_errors_outlierfree.join(train_stats) # Add the new columns
            # Write this (single-row) dataframe to .xls file.
            # It may exist already, in which case we append a new row with the result for this scene.
            general_utils.write_results(conf, train_res.round(3), file_name="best_train_errors_{}".format(phase.name), additional_identifiers=additional_identifiers, append=True)
