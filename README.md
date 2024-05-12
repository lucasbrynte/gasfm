# Learning Structure-from-Motion with Graph Attention Networks <br>

### [Paper](https://arxiv.org/abs/2308.15984)

This is the implementation of the Graph Attention Structure-from-Motion (GASFM) architecture, presented in our CVPR 2024 paper <a href="https://arxiv.org/abs/2308.15984">Learning Structure-from-Motion with Graph Attention Networks</a>. The codebase is forked from the implementation of the ICCV 2021 paper [Deep Permutation Equivariant Structure from Motion](https://openaccess.thecvf.com/content/ICCV2021/html/Moran_Deep_Permutation_Equivariant_Structure_From_Motion_ICCV_2021_paper.html), available at  [https://github.com/drormoran/Equivariant-SFM](https://github.com/drormoran/Equivariant-SFM). That architecture is also used as a baseline and referred to as DPESFM in our paper.

The primary focus of our paper is on Euclidean reconstruction of novel test scenes, achieved by training a graph attention network on a few scenes, which then generalizes well enough to provide an initial solution, locally refined by bundle adjustment. Our experiments demonstrate that high quality reconstructions can be acquired around 5-10 times faster than COLMAP.


### Contents

- [Setup](#Setup)
- [Usage](#Usage)
- [Citation](#Citation)

---

## Setup
This repository is implemented with python 3.9, and in order to run bundle adjustment requires linux. We have used Ubuntu 22.04. You should also have a CUDA-capable GPU.

## Directory structure
The repository should contain the following directories:
```
gasfm
├── bundle_adjustment
├── code
├── datasets
├── environment.yml
```

### Conda environment
First create an empty conda environment and install python  3.9:
```
conda create -n gasfm
conda activate gasfm
conda install python=3.9
```
Install the Facebook research vision core library:
```
conda install -c fvcore -c iopath -c conda-forge fvcore
```
Install PyTorch and CUDA toolkit 11.6 in the environment (in my experience, it is often more reliable to install the desired version of PyTorch before other dependencies):
```
conda install -c conda-forge cudatoolkit-dev=11.6
conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install Tensorboard:
```
pip install future tb-nightly
```
Finally, install all remaining dependencies:
```
conda env update -n gasfm -f environment.yml
```
Verify that PyTorch was successfully installed with CUDA enabled:
```
python -c 'import torch; assert torch.cuda.is_available()'
```
Verify that PyTorch Geometric was also successfully installed:
```
python -c 'import torch_geometric'
```

### PyCeres
Next follow the <a href="bundle_adjustment/README.md">bundle adjustment instructions</a>.

### Data and pretrained models
Attached to [this](https://github.com/lucasbrynte/gasfm/releases/tag/data) mockup GitHub release you can find both the datasets and pretrained models for Euclidean as well as projective reconstruction of novel scenes. For Euclidean reconstruction there are three models: 1) Trained without data augmentation (`gasfm_euc_noaug.pt`), 2) trained with data augmentation (`gasfm_euc_rhaug-15-20.pt`), and 3) trained with data augmentation as well as artificial outlier injection (`gasfm_euc_rhaug-15-20_outliers0.1.pt`).

Create a directory `gasfm/datasets`, download the data, and unzip it:
```
mkdir -p datasets
cd datasets
wget https://github.com/lucasbrynte/gasfm/releases/download/data/datasets.zip
unzip datasets.zip
```

Optionally, download one or all pretrained models, e.g. in a dedicated subdirectory:
```
cd ..
mkdir -p pretrained_models
cd pretrained_models
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_noaug.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_rhaug-15-20.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_rhaug-15-20_outliers0.1.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_proj_noaug.pt
```


## Usage

### Novel scene reconstruction
To execute the code, first navigate to the `code` subdirectory. Also make sure that the conda environment is activated.

To train a model from scratch for reconstruction of novel test scenes, run:
```
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning
```
where `path/to/conf` is relative to `code/confs/`, and may e.g. be `gasfm/learning_euc_rhaug-15-20_gasfm.conf` for training a Euclidean reconstruction model using data augmentation. The `exp/output/path` path is relative to `results/`.

The training phase is succeeded by bundle adjustment, evaluation, and by default also by separate fine-tuning of the model parameters on every test scene. The fine-tuning is done with both the final model and the best model (i.e. early stopping), as well as an equally short optimization from random model parameters, for reference. Fine-tuning can be disabled by appending the following argument:
```
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning --skip-fine-tuning
```

Somewhat similarly, the training phase itself can be skipped, in which case only bundle adjustment, evaluation, and (optionally) fine-tuning is carried out.
If training is skipped, a pretrained model needs to be specified, either with an absolute path as follows:
```
python main.py --conf path/to/conf --exp-dir exp/output/path --pretrained-model-path /abs/path/to/pretrained/model multi-scene-learning --skip-training
```
or by pointing to a pre-existing experiment directory and naming the desired model checkpoint:
```
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning --skip-training --old-exp-dir /another/preexisting/experiment/path --pretrained-model-filename best_model.pt
```

You can also override any value of the config file from the command line. For example, to change the number of training epochs and the evaluation frequency use:
```
python main.py --conf path/to/conf --external_params train.n_epochs=100000 eval.eval_interval=100 --exp-dir exp/output/path multi-scene-learning
```


### Single-scene recovery
To perform single-scene recovery by fitting a model to a single dataset, in this example `AlcatrazCourtyard`, run:
```
python main.py --conf path/to/conf --exp-dir exp/output/path single-scene-optim --scene-name-exp-subdir --scene AlcatrazCourtyard
```
where `path/to/conf` is relative to `code/confs/`, and may e.g. be `gasfm/optim_euc_gasfm.conf` (for the Euclidean setting). The `exp/output/path` path is relative to `results/`.

Again, you can override any value of the config file from the command line, e.g.:
```
python main.py --conf path/to/conf --external_params train.n_epochs=100000 eval.eval_interval=100 --exp-dir exp/output/path single-scene-optim --scene-name-exp-subdir --scene AlcatrazCourtyard
```


## Citation
If you find this work useful, please cite our paper:
```
@article{brynte2023learning,
      title = {Learning Structure-from-Motion with Graph Attention Networks}, 
      author = {Lucas Brynte and José Pedro Iglesias and Carl Olsson and Fredrik Kahl},
      journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2024}
}
```
