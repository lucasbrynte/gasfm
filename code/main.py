import torch  # DO NOT REMOVE
import cv2  # DO NOT REMOVE

import os
import shutil
import argparse
import random
import numpy as np
import pandas as pd
from pyhocon import ConfigFactory, ConfigTree
import single_scene_optimization
import multiple_scenes_learning
from utils.Phases import Phases
from utils import general_utils
from utils.path_utils import path_to_exp_root, path_to_exp, path_to_conf, path_to_ref_conf

def parse_args():
    # Define main argument parser
    parser = argparse.ArgumentParser()

    # Define subparser
    subparsers = parser.add_subparsers(help = 'Mode-specific arguments.', dest = 'mode')
    subparsers.required = True

    # CLI args specific to single-scene optimization
    single_scene_optim_parser = subparsers.add_parser('single-scene-optim', aliases=['single_scene_optim'])
    single_scene_optim_parser.set_defaults(mode='single_scene_optim')
    single_scene_optim_parser.add_argument('--scene', type=str, default=None)
    single_scene_optim_parser.add_argument('--scene-name-exp-subdir', '--scene_name_exp_subdir', action='store_true', default=False, help='If enabled, the experiment directory is implicitly modified to a subdir with the name of the scene.')

    # CLI args specific to multi-scene learning
    multi_scene_learning_parser = subparsers.add_parser('multi-scene-learning', aliases=['multi_scene_learning'])
    multi_scene_learning_parser.set_defaults(mode='multi_scene_learning')
    multi_scene_learning_parser.set_defaults(scene=None)
    multi_scene_learning_parser.set_defaults(scene_name_exp_subdir=None)
    multi_scene_learning_parser.add_argument('--old-exp-dir', '--old_exp_dir', type=str, default=None, help='Path to old experiment. Implies loading pretrained model parameters from this experiment (best model), unless another pretrained model path is explicitly provided.')
    multi_scene_learning_parser.add_argument('--pretrained-model-filename', '--pretrained_model_filename', type=str, default=None)
    multi_scene_learning_parser.add_argument('--skip-training', '--skip_training', action='store_true', default=False)
    multi_scene_learning_parser.add_argument('--skip-fine-tuning', '--skip_fine_tuning', action='store_true', default=False)
    multi_scene_learning_parser.add_argument('--skip-fine-tuning-from-best', '--skip_fine_tuning_from_best', action='store_true', default=False)
    multi_scene_learning_parser.add_argument('--skip-fine-tuning-from-final', '--skip_fine_tuning_from_final', action='store_true', default=False)
    multi_scene_learning_parser.add_argument('--skip-short-optim', '--skip_short_optim', action='store_true', default=False)

    # Shared CLI args
    parser.add_argument('--conf', type=str)
    parser.add_argument('--exp-dir', '--exp_dir', type=str, default=None, help='Experiment output directory (relative path to experiment root, which is ../results). Defaults to a timestamp.')
    parser.add_argument('--overwrite-exp', '--overwrite_exp', action='store_true', default=False)
    parser.add_argument('--external-params', '--external_params', type=str, action='append', default=None, nargs='+')
    parser.add_argument('--pretrained-model-path', '--pretrained_model_path', type=str, default=None)
    parser.add_argument('--gpu-not-required', '--gpu_not_required', action='store_true', default=False)
    parser.add_argument('--count-model-params-and-die', '--count_model_params_and_die', action='store_true', default=False)
    args = parser.parse_args()

    return args

def parse_external_params(ext_params_list, conf):
    """
    ext_params_list is a list of strings of the following or similar format: "eval.eval_interval=100" (simply needs to be pyhocon-parsable).
    """
    assert isinstance(ext_params_list, list)
    for param_str in ext_params_list:
        assert isinstance(param_str, str)
        try:

            # Parse string and get a new config tree holding nothing but the current parameter:
            conf_patch = ConfigFactory.parse_string(param_str)
            conf = ConfigTree.merge_configs(conf, conf_patch)
        except Exception as e:
            print("Could not parse external parameter: \"{}\".".format(param_str))
            raise e

    return conf

def init_exp(args):
    # Init Device
    if not args.gpu_not_required:
        assert torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Init Conf
    conf_file_path = path_to_conf(args.conf)
    conf = ConfigFactory.parse_file(conf_file_path)
    conf["original_file_name"] = args.conf

    # Init Default Conf
    ref_conf_file_path = path_to_ref_conf()
    ref_conf = ConfigFactory.parse_file(ref_conf_file_path)

    # Init experiment dir
    if args.exp_dir is None:
        exp_dir = general_utils.gen_dflt_exp_dir()
    else:
        exp_dir = args.exp_dir
    conf['exp_dir'] = exp_dir

    # Init external params
    if args.external_params is not None:
        # We expect a list of lists.
        # Each list element (itself a list) corresponds to a single occurence of the --external-params.
        # Each element of that list corresponds to the corresponding number of CLI arguments passed (any number is accepted to due to nargs='+').
        assert isinstance(args.external_params, list)
        assert all(isinstance(sub_list, list) for sub_list in args.external_params)
        external_params = [ param_str for sub_list in args.external_params for param_str in sub_list ]
        conf = parse_external_params(external_params, conf)

    # Verify that there are no unexpected configuration entries (e.g. due to typos)
    mismatches = general_utils.detect_discrepancies_with_reference_configtree(ref_conf, conf)
    if len(mismatches) > 0:
        raise NotImplementedError('Detected unexpected configuration entries:\n\t{}'.format('\n\t'.join(mismatches)))

    # Configuration specific to single-scene optimization
    if args.mode == 'single_scene_optim':
        phase = Phases.OPTIMIZATION
        assert 'scene' in conf['dataset'].keys(), 'Running single-scene optimization but no scene (scene) specified in the configuration.'

        # Init scene
        if args.scene is not None:
            conf['dataset']['scene'] = args.scene

        # Optionally update experiment dir, appending a subdir with the scene name
        if args.scene_name_exp_subdir:
            assert conf['dataset']['scene'] is not None, 'Since no scene name is specified, the --scene-name-exp-subdir mode can not be used.'
            conf['exp_dir'] = os.path.join(conf['exp_dir'], conf['dataset']['scene'])

    # Configuration specific to multi-scene learning
    elif args.mode == 'multi_scene_learning':
        phase = Phases.TRAINING
        assert 'scene' not in conf['dataset'].keys(), 'Running multiple-scene-learning, yet found a single scene (scene) specified in the configuration.'
    else:
        assert "Unexepected mode: {}.".format(args.mode)

    return conf, device, phase

def init_model(conf):
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()

    conf, device, phase = init_exp(args)

    # Init Seed
    seed = conf.get_int('random_seed', default=None)
    if seed is not None:
        torch.manual_seed(seed)
        general_utils.base_worker_rng.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Optionally overwrite existing experiment
    exp_path = path_to_exp(conf, create=False)
    if os.path.exists(exp_path) and args.overwrite_exp:
        shutil.rmtree(exp_path)

    # Log code
    general_utils.log_code(conf)

    # Initialize model
    model = init_model(conf)
    print('#Trainable parameters of model: {}'.format(count_parameters(model)))
    if args.count_model_params_and_die:
        assert False
    assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
    if args.pretrained_model_path is not None:
        assert args.pretrained_model_filename is None or args.pretrained_model_filename == os.path.basename(pretrained_model_path), '--pretrained-model-filename {} specified, but would be overridden by --pretrained-model-parh {}.'.format(args.pretrained_model_filename, pretrained_model_path)
        ret = model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        assert all('head' in key or 'final' in key for key in ret.missing_keys)
        assert all('head' in key or 'final' in key for key in ret.unexpected_keys)
        if len(ret.missing_keys) > 0:
            print('[WARNING] Loaded statedict is missing the following keys:\n{}'.format(ret.missing_keys))
        if len(ret.unexpected_keys) > 0:
            print('[WARNING] Loaded statedict is missing the following keys:\n{}'.format(ret.unexpected_keys))
    elif args.mode == 'multi_scene_learning' and args.old_exp_dir is not None:
        # Pretrained model path not explicitly provided, but since there is an old experiment directory specified, we locate the best model (epoch=None) from there.
        # As a bonus, we will also be able to read the recorded validation performance (this is done further below).
        models_dir = os.path.join(args.old_exp_dir, 'models')
        if args.pretrained_model_filename is None:
            args.pretrained_model_filename = 'best_model.pt'
        pretrained_model_path = os.path.join(path_to_exp_root(conf), models_dir, args.pretrained_model_filename)
        ret = model.load_state_dict(torch.load(pretrained_model_path), strict=False)
        assert all('head' in key or 'final' in key for key in ret.missing_keys)
        assert all('head' in key or 'final' in key for key in ret.unexpected_keys)
        if len(ret.missing_keys) > 0:
            print('[WARNING] Loaded statedict is missing the following keys:\n{}'.format(ret.missing_keys))
        if len(ret.unexpected_keys) > 0:
            print('[WARNING] Loaded statedict is missing the following keys:\n{}'.format(ret.unexpected_keys))

    # Single-scene optimization mode --------------------------------------
    if args.mode == 'single_scene_optim':
        phase = Phases.OPTIMIZATION
        assert 'scene' in conf['dataset'].keys(), 'Running single-scene optimization but no scene (scene) specified in the configuration.'
        single_scene_optimization.train_model_single_scene(conf, model, device, phase, additional_identifier=None, crash_on_scene_exhausting_memory=True)
    # Multi-scene learning mode -------------------------------------------
    elif args.mode == 'multi_scene_learning':
        phase = Phases.TRAINING
        assert 'scene' not in conf['dataset'].keys(), 'Running multiple-scene-learning, yet found a single scene (scene) specified in the configuration.'

        # Create eval dataloaders
        datasets, eval_data_loaders = multiple_scenes_learning.create_eval_dataloaders(conf)

        if not args.skip_training:
            # (Pre-)train model
            trained_models, train_stats = multiple_scenes_learning.train_model(conf, model, datasets['train_set'], eval_data_loaders, device, phase)

        # Eval model(s)
        if not args.skip_training:
            multiple_scenes_learning.eval_model(conf, trained_models['final_model'], eval_data_loaders, -1, 'final_', device)
            if conf.get_string('train.validation_metric', default=None) is not None:
                assert 'best_model' in trained_models
                multiple_scenes_learning.eval_model(conf, trained_models['best_model'], eval_data_loaders, None, 'best_', device)
        else:
            # Evaluation is carried out on the specified model, which is loaded from disk above.
            # NOTE: For convenience, we use the "best_model" identifier, although the source of this model is actually arbitrary.
            multiple_scenes_learning.eval_model(conf, model, eval_data_loaders, None, 'best_', device)

        if not args.skip_training:
            # Keep fine-tuning on the final model from pretraining
            if not args.skip_fine_tuning and not args.skip_fine_tuning_from_final:
                assert not general_utils.any_model_parameters_on_gpu(trained_models['final_model']) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
                multiple_scenes_learning.optimization_all_test_scenes(conf, trained_models['final_model'], device, Phases.FINE_TUNE, additional_identifier='from_final')

            # Keep fine-tuning on the best-performing model from pretraining
            if not args.skip_fine_tuning and not args.skip_fine_tuning_from_best:
                assert not general_utils.any_model_parameters_on_gpu(trained_models['best_model']) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
                multiple_scenes_learning.optimization_all_test_scenes(conf, trained_models['best_model'], device, Phases.FINE_TUNE, additional_identifier='from_best')
        elif not args.skip_fine_tuning:
            assert args.pretrained_model_path is not None or args.old_exp_dir is not None, 'For fine-tuning, either enable the pretraining phase, or provide a path for pretrained model weights, either explicitly, or via --old-exp-dir.'
            # Fine-tuning starts from the parameters of the initial "model" variable, which are loaded from disk above.
            assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
            multiple_scenes_learning.optimization_all_test_scenes(conf, model, device, Phases.FINE_TUNE, additional_identifier=None)

        # Re-initialize model for short optimization
        if not args.skip_fine_tuning and not args.skip_short_optim:
            model = init_model(conf)
            assert not general_utils.any_model_parameters_on_gpu(model) # The model will be duplicated, and we wouldn't want to store duplicate parameters on the GPU.
            multiple_scenes_learning.optimization_all_test_scenes(conf, model, device, Phases.SHORT_OPTIMIZATION, additional_identifier=None)
    else:
        assert "Unexepected mode: {}.".format(args.mode)

if __name__ == "__main__":
    main()
