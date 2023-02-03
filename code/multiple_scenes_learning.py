import torch
from torch.utils.data import DataLoader
from utils import general_utils, dataset_utils
from utils.general_utils import get_additional_identifiers_for_outlier_injection
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet
from datasets.ScenesDataSet import dataloader_collate_fn as collate_fn
from datasets import SceneData
from single_scene_optimization import train_model_single_scene
import train
import copy


def create_eval_dataloaders(conf):
    min_num_views_sampled = conf.get_int('dataset.min_num_views_sampled')
    max_num_views_sampled = conf.get_int('dataset.max_num_views_sampled')
    inplane_rot_aug_max_angle = conf.get_float('dataset.inplane_rot_aug_max_angle')
    tilt_rot_aug_max_angle = conf.get_float('dataset.tilt_rot_aug_max_angle')
    batch_size = conf.get_int('dataset.batch_size')
    dataloader_num_workers = conf.get_int('dataset.dataloader_num_workers')

    # Create train, test and validation sets
    test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
    validation_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.validation_set'), conf)
    train_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)

    train_set = ScenesDataSet(
        train_scenes,
        return_all = False,
        min_num_views_sampled = min_num_views_sampled,
        max_num_views_sampled = max_num_views_sampled,
        inplane_rot_aug_max_angle = inplane_rot_aug_max_angle,
        tilt_rot_aug_max_angle = tilt_rot_aug_max_angle,
    )
    train_set_for_eval = ScenesDataSet(train_scenes, return_all=True)
    validation_set = ScenesDataSet(validation_scenes, return_all=True)
    test_set = ScenesDataSet(test_scenes, return_all=True)

    datasets = {
        'train_set': train_set,
        'train_set_for_eval': train_set_for_eval,
        'validation_set': validation_set,
        'test_set': test_set,
    }

    # Create dataloaders
    eval_data_loaders = {
        'train_loader_for_eval': DataLoader(train_set_for_eval, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng),
        'validation_loader': DataLoader(validation_set, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng),
        'test_loader': DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng),
    }

    return datasets, eval_data_loaders

def train_model(conf, model, train_set, eval_data_loaders, device, phase):
    assert phase == Phases.TRAINING

    # Get configuration
    batch_size = conf.get_int('dataset.batch_size')
    dataloader_num_workers = conf.get_int('dataset.dataloader_num_workers')

    assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.

    # Create dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=dataloader_num_workers, worker_init_fn=general_utils.seed_worker, generator=general_utils.base_worker_rng)

    # Train model
    trained_models, train_stats = train.train(conf, device, train_loader, model, phase, train_loader_for_eval=eval_data_loaders['train_loader_for_eval'], val_loader=eval_data_loaders['validation_loader'], test_loader=eval_data_loaders['test_loader'], additional_identifier=None)
    assert not any(general_utils.any_model_parameters_on_gpu(m) for m in trained_models.values()) # The trained models should no longer be on GPU.
    general_utils.write_results(conf, train_stats.round(3), file_name='train_stats', additional_identifiers=[])

    return trained_models, train_stats


def eval_model(conf, model, data_loaders, store_as_epoch, filename_prefix, device):
    # Get configuration
    outlier_injection_rate = conf.get_float('train.outlier_injection_rate', default=None)
    run_ba = conf.get_bool('ba.run_ba', default=True)
    stdout_log_eval_memory_consumption = conf.get_bool('memory.stdout_log_eval_memory_consumption', default=True)
    post_train_eval_no_crash_on_scene_exhausting_memory = conf.get_bool('memory.post_train_eval_no_crash_on_scene_exhausting_memory', default=True)

    model.to(device)
    train_errors = train.epoch_evaluation(data_loaders['train_loader_for_eval'], model, conf, device, store_as_epoch, Phases.TRAINING, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=True, additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
    val_errors = train.epoch_evaluation(data_loaders['validation_loader'], model, conf, device, store_as_epoch, Phases.VALIDATION, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=True, additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
    test_errors = train.epoch_evaluation(data_loaders['test_loader'], model, conf, device, store_as_epoch, Phases.TEST, outlier_injection_rate=outlier_injection_rate, dump_and_plot_predictions=True, additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate), bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
    if outlier_injection_rate is not None:
        # Extra outlier-free validation
        train_errors_outlierfree = train.epoch_evaluation(data_loaders['train_loader_for_eval'], model, conf, device, store_as_epoch, Phases.TRAINING, dump_and_plot_predictions=True, additional_identifiers=[], bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
        val_errors_outlierfree = train.epoch_evaluation(data_loaders['validation_loader'], model, conf, device, store_as_epoch, Phases.VALIDATION, dump_and_plot_predictions=True, additional_identifiers=[], bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
        test_errors_outlierfree = train.epoch_evaluation(data_loaders['test_loader'], model, conf, device, store_as_epoch, Phases.TEST, dump_and_plot_predictions=True, additional_identifiers=[], bundle_adjustment=run_ba, log_memory_consumption=stdout_log_eval_memory_consumption, crash_on_scene_exhausting_memory=not post_train_eval_no_crash_on_scene_exhausting_memory)
    model.cpu()
    general_utils.write_results(conf, train_errors.round(3), file_name=filename_prefix+'train_errors', additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate))
    general_utils.write_results(conf, val_errors.round(3), file_name=filename_prefix+'val_errors', additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate))
    general_utils.write_results(conf, test_errors.round(3), file_name=filename_prefix+'test_errors', additional_identifiers=[]+get_additional_identifiers_for_outlier_injection(outlier_injection_rate))
    if outlier_injection_rate is not None:
        # Extra outlier-free validation
        general_utils.write_results(conf, train_errors_outlierfree.round(3), file_name=filename_prefix+'train_errors', additional_identifiers=[])
        general_utils.write_results(conf, val_errors_outlierfree.round(3), file_name=filename_prefix+'val_errors', additional_identifiers=[])
        general_utils.write_results(conf, test_errors_outlierfree.round(3), file_name=filename_prefix+'test_errors', additional_identifiers=[])


def optimization_all_test_scenes(conf, model, device, phase, additional_identifier=None):
    # Get configuration
    finetune_n_epochs = conf.get_int("train.finetune_n_epochs")
    finetune_eval_interval = conf.get_int('train.finetune_eval_interval')
    finetune_dump_model_interval = conf.get_int('train.finetune_dump_model_interval', default=None)
    finetune_dump_and_plot_pred_interval = conf.get_int('train.finetune_dump_and_plot_pred_interval', default=None)
    finetune_lr = conf.get_float('train.finetune_lr')
    finetune_lr_warmup_n_steps = conf.get_float('train.finetune_lr_warmup_n_steps', default=0)
    finetune_no_crash_on_scene_exhausting_memory = conf.get_bool('memory.finetune_no_crash_on_scene_exhausting_memory', default=True)

    assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
    initial_model = copy.deepcopy(model)

    test_scenes_list = conf.get_list('dataset.test_set')
    # test_scenes_list = []
    # for test_data in test_set:
    #     test_scenes_list.append(test_data.scene_name)

    conf_test = copy.deepcopy(conf)
    conf_test['dataset']['scenes_list'] = test_scenes_list
    # Set #epochs to 500 for fine-tuning / short optim:
    conf_test['train']['n_epochs'] = finetune_n_epochs
    conf_test['train']['eval_interval'] = finetune_eval_interval
    conf_test['train']['finetune_dump_model_interval'] = finetune_dump_model_interval
    conf_test['train']['finetune_dump_and_plot_pred_interval'] = finetune_dump_and_plot_pred_interval
    conf_test['train']['lr'] = finetune_lr
    conf_test['train']['lr_schedule']['lr_warmup_n_steps'] = finetune_lr_warmup_n_steps
    conf_test['train']['lr_schedule']['main_scheduler'] = 'constant'

    # Get logs directories
    for i, scene in enumerate(test_scenes_list):
        conf_test["dataset"]["scene"] = scene
        for p1, p2 in zip(initial_model.parameters(), model.parameters()):
            assert torch.all(p1 == p2) # Sanity check, verifying that the initial model parameters have not unintensionally been modified in-place by previous optimization loops.
        train_model_single_scene(conf_test, model, device, phase, additional_identifier=additional_identifier, crash_on_scene_exhausting_memory=not finetune_no_crash_on_scene_exhausting_memory)
