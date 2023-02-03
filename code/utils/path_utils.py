import os
import shutil
from utils.Phases import Phases


def join_and_create(*args, create=True):
    assert len(args) > 0
    full_path = os.path.join(*args)
    if create:
        os.makedirs(full_path, exist_ok=True)
    return full_path


def path_to_datasets():
    return os.path.join('..', 'datasets')


def path_to_exp_root(conf):
    exp_root_path = os.path.join('..', 'results')
    return exp_root_path


def path_to_exp(conf, create=True):
    exp_dir = conf.get_string('exp_dir')
    exp_root_path = path_to_exp_root(conf)
    exp_path = join_and_create(exp_root_path, exp_dir, create=create)
    return exp_path


def path_to_phase(conf, phase, additional_identifiers=[]):
    exp_path = path_to_exp(conf)
    subdir = '_'.join([phase.name] + additional_identifiers)
    return join_and_create(exp_path, subdir)


def path_to_scene(conf, phase, scene=None, additional_identifiers=[]):
    phase_path = path_to_phase(conf, phase, additional_identifiers=additional_identifiers)
    scene = conf.get_string("dataset.scene") if scene is None else scene
    return join_and_create(phase_path, scene)


def path_to_models_dir(conf, phase, scene=None, additional_identifiers=[]):
    if phase in [Phases.TRAINING, Phases.VALIDATION, Phases.TEST]:
        parent_folder = path_to_exp(conf)
    else:
        parent_folder = path_to_scene(conf, phase, scene=scene, additional_identifiers=additional_identifiers)

    models_dir = join_and_create(parent_folder, 'models')
    return models_dir


def path_to_predictions(conf, phase, epoch=None, scene=None, additional_identifiers=[]):
    scene_path = path_to_scene(conf, phase, scene=scene, additional_identifiers=additional_identifiers)
    predictions_path = join_and_create(scene_path, 'predictions')

    if epoch is None:
        predictions_file_name = "best_predictions"
    elif epoch == -1:
        predictions_file_name = "final_predictions"
    else:
        predictions_file_name = "predictions_epoch{:06d}".format(epoch+1)

    return os.path.join(predictions_path, predictions_file_name)


def path_to_plots(conf, phase, epoch=None, scene=None, additional_identifiers=[]):
    scene_path = path_to_scene(conf, phase, scene=scene, additional_identifiers=additional_identifiers)
    plots_path = join_and_create(scene_path, 'plots')

    if epoch is None:
        plots_file_name = "best_plots.html"
    elif epoch == -1:
        plots_file_name = "final_plots.html"
    else:
        plots_file_name = "plot_epoch{:06d}.html".format(epoch+1)

    return os.path.join(plots_path, plots_file_name)


def path_to_code_logs(conf):
    exp_path = path_to_exp(conf)
    code_path = join_and_create(exp_path, "code")
    return code_path


def path_to_conf(conf_file):
    return os.path.join( 'confs', conf_file)


def path_to_ref_conf():
    return os.path.join( 'confs', 'ref.conf')


def path_to_tb_events(conf):
    exp_path = path_to_exp(conf)
    tb_path = join_and_create(exp_path, "tb")
    return tb_path
