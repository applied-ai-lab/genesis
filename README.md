# Genesis and Genesis-V2

This is the official PyTorch reference implementation of:
> ["GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations"](https://arxiv.org/abs/1907.13052)  
> Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, and Ingmar Posner  
> International Conference on Learning Representations (ICLR), 2020

> ["Reconstruction Bottlenecks in Object-Centric Generative Models"](https://oolworkshop.github.io/program/ool_5.html)  
> Martin Engelcke, Oiwi Parker Jones, and Ingmar Posner  
> Workshop on Object-Oriented Learning at ICML, 2020

> ["GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement"](https://arxiv.org/abs/2104.09958v2)  
> Martin Engelcke, Oiwi Parker Jones, and Ingmar Posner  
> arXiv preprint arXiv:2104.09958, 2021

As part of these works, the repository also includes:
* a [re-implementation of MONet](https://github.com/applied-ai-lab/genesis/blob/master/models/monet_config.py) from ["MONet: Unsupervised Scene Decomposition and Representation"](https://arxiv.org/abs/1901.11390) by Burgess et al.;
* a [re-implementation of GECO](https://github.com/applied-ai-lab/genesis/blob/master/utils/geco.py) from ["Taming VAEs"](https://arxiv.org/abs/1810.00597) by Rezende and Viola.

## Setup

### Dependencies
Clone the repository into, e.g., `~/code/genesis`:
```shell
git clone --recursive https://github.com/applied-ai-lab/genesis.git ~/code/genesis
```
We use [Forge](https://github.com/akosiorek/forge) to save some legwork. It is included as a submodule but you need to add it to your python path, e.g. with:
```shell
# If needed, replace .bashrc with .zshrc or similar
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/code/genesis/forge"' >> ~/.bashrc
```
You can either install PyTorch, TensorFlow, and all other dependencies manually or you can setup up conda environment with all required dependencies using the `environment.yml` file:
```shell
conda env create -f environment.yml
conda activate genesis_env
```

### Datasets
This repository contains data loaders for the three datasets considered in the [ICLR paper](https://arxiv.org/abs/1907.13052).
A few steps are required for setting up each individual dataset.
We also provide a PyTorch wrapper around the [Multi-Object Datasets](https://github.com/deepmind/multi-object-datasets/) used for the experiments on the `Objects Room` dataset in the [ICML workshop paper](https://oolworkshop.github.io/program/ool_5.html).

#### Multi-dSprites
Generate coloured Multi-dSprites from the original [dSprites dataset](https://github.com/deepmind/dsprites-dataset) with:
```shell
cd ~/code/genesis
mkdir -p data/multi_dsprites/processed
git clone https://github.com/deepmind/dsprites-dataset.git data/multi_dsprites/dsprites-dataset
python scripts/generate_multid.py
```
**NOTE:** An RGB colour is sampled from 125 possible colours for each scene component. By default, multiple components in an image can have the same colour. This can lead, e.g., to a foreground object to have the same colour as the background so that the object is practically "invisible". A ground truth segmentation mask will still be associated with such an invisible object. If you want to avoid this, you can set `--unique_colours True` during training to use an alternative dataset where each component in an image has a unique colour.

#### GQN (rooms-ring-camera)
The [GQN datasets](https://github.com/deepmind/gqn-datasets) are quite large. The `rooms_ring_camera` dataset as used in the paper takes about 250GB and can be downloaded with:
```shell
pip install gsutil
cd ~/code/genesis
mkdir -p data/gqn_datasets
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera data/gqn_datasets
```
Note that we use a modified version of the TensorFlow GQN data loader from [ogroth/tf-gqn](https://github.com/ogroth/tf-gqn) which is included in `third_party/tf_gqn`.

#### ShapeStacks
You need about 30GB of free disk space for [ShapeStacks](https://shapestacks.robots.ox.ac.uk/):
```shell
# Download compressed dataset
cd data
wget -i ../utils/shapestacks_urls.txt
# Uncompress files
tar xvzf shapestacks-meta.tar.gz
tar xvzf shapestacks-mjcf.tar.gz
tar xvzf shapestacks-rgb.tar.gz
cd -
```
The instance segmentation labels for ShapeStacks can be downloaded from [here](https://drive.google.com/open?id=1KsSQCgb1JJExbKyrIkTwBL9VidGcq2k7).

#### Multi-Object Datasets
The repository contains a wrapper around the [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets), returning an iterable which behaves similarly to a PyTorch DataLoader object.
The default config assumes that any datasets you wish to use have been downloaded to `data/multi-object-datasets`.
As for the GQN data, this can be done with gsutil.
You can download all four datasets at once with:
```shell
gsutil cp -r gs://multi-object-datasets data/
```

#### Sketchy
Clone [deepmind-research](https://github.com/deepmind/deepmind-research) into, e.g., `code/deepmind-research`:
```shell
git clone https://github.com/deepmind/deepmind-research.git ~/code/deepmind-research
```
Download `lift_green__demos 2` and `stack_green_on_red__demos 2` using the script at `deepmind-research/sketchy/download.sh`.
Put the data into `~/code/genesis/data/sketchy/records` with the contents of the folder being the actual tfrecord files.
Make sure the `deepmind-research` is on your python path:
```shell
# If needed, replace .bashrc with .zshrc or similar
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/code/deepmind-research"' >> ~/.bashrc
```
Create a separate environment according to `deepmind-research/sketchy/requirements.txt`
```shell
# Leave current environment first if necessary
conda deactivate
conda create -n sketchy python=3.7
conda activate sketchy
pip install -r ~/code/deepmind-research/sketchy/requirements.txt
# Some additional dependencies
pip install torch==1.3.1 torchvision==0.4.2 tqdm pillow
```
You can now preprocess the data with:
```shell
python scripts/sketchy_preparation.py
conda deactivate
```

#### MIT-Princeton 2016 Amazon Picking Challenge (APC)
Dowload the "Object Segmentation Training Dataset" from the [team's website](http://apc.cs.princeton.edu) via the [download link](http://3dvision.princeton.edu/projects/2016/apc/downloads/training.zip) (ca. 130GB).
Move the `training.zip` file into `~/code/genesis/data/apc` and unpack it.
Preprocess the data by running:
```shell
python datasets/apc_config.py
```

## Training
You can train Genesis-v2, Genesis, MONet and baseline VAEs on the datasets using the default hyperparameters with, e.g.:
```shell
python train.py --data_config datasets/shapestacks_config.py --model_config models/genesisv2_config.py
python train.py --data_config datasets/gqn_config.py --model_config models/genesis_config.py
python train.py --data_config datasets/multi_object_config.py --model_config models/monet_config.py
python train.py --data_config datasets/multid_config.py --model_config models/vae_config.py
```
You can change many of the hyperparameters via the Forge command line flags in the respective config files, e.g.:
```shell
python train.py --data_config datasets/multid_config.py --model_config models/genesis_config.py --batch_size 64 --learning_rate 0.001
```
See [train.py](https://github.com/applied-ai-lab/genesis/blob/master/train.py) and the config files for the available flags.

TensorBoard logs are written to file with [TensorboardX](https://github.com/lanpa/tensorboardX). Run `tensorboard --logdir checkpoints` to monitor training.

**NOTE 1:** If you train MONet with the default config flags, then the hyperparameters from our ICLR paper are used which are different from the ones in Burgess et al.. If you want to use the training hyperparameters from Burgess et al., then you need to add the following flags: `--geco False --pixel_std1 0.09 --pixel_std2 0.11 --train_iter 1000000 --batch_size 64 --optimiser rmsprop`.

**NOTE 2:** The Sketchy results in the GENESIS-V2 paper use a different GECO goal than used in the other experiments. It is necessary to override the default value to reproduce these results, which can be done by adding `--g_goal 0.5645` as a training flag.

## Evaluation
To compute the FID score for a trained model you can run, e.g.:
```shell
python scripts/compute_fid.py --data_config datasets/gqn_config.py --model_config models/genesis_config.py --model_dir checkpoints/MyModel/1 --model_file model.ckpt-FINAL
```
Similarly, you can compute the segmentation metrics with, e.g.:
```shell
python scripts/compute_seg_metrics.py --data_config datasets/gqn_config.py --model_config models/genesis_config.py --model_dir checkpoints/MyModel/1 --model_file model.ckpt-FINAL
```

## Visualisation
You can visualise your data with, e.g.:
```shell
python scripts/visualise_data.py --data_config datasets/multid_config.py
python scripts/visualise_data.py --data_config datasets/gqn_config.py
python scripts/visualise_data.py --data_config datasets/shapestacks_config.py
python scripts/visualise_data.py --data_config datasets/multi_object_config.py --dataset objects_room
```
Scripts for visualising reconstructions/segmentations and samples are available at `scripts/visualise_reconstruction.py` and `scripts/visualise_generation.py`, respectively.

## Pretrained models & results
Checkpoints of pretrained models are available [here](https://drive.google.com/drive/folders/1dp6trBGAFUeQ1V3h61Mlv0JeMZqc97sw?usp=sharing).

Generation and segmentation metrics of the released model checkpoints are summarised in the following table:
| Model | Dataset | FID &darr; | ARI-FG &uarr; | MSC-FG &uarr; | ARI &uarr; | MSC &uarr; |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| GENESIS | Multi-dSprites | 25.0 | 0.57 | 0.69 | - | - |
| GENESIS | GQN | 79.4 | no labels | no labels | no labels | no labels |
| GENESIS | ShapeStacks | 235.4 | 0.71 | 0.64 | - | - |
| GENESIS-V2 | ShapeStacks | 108.1 | 0.80 | 0.66 | - | - |
| GENESIS-V2 | ObjectsRoom | 53.2 | 0.82 | 0.61 | - | - |
| GENESIS-V2 | Sketchy | 212.7 | no labels | no labels | no labels | no labels |
| GENESIS-V2 | APC | 232.2 | - | - | 0.55 | 0.65 |

Other than varying the number of object slots `K`, models are trained with the same default hyperparameters across datasets. Generation and segmentation performance can be improved by further tuning hyperparameters for each individual dataset. For example, [Dang-Nhu & Steger 2021](https://arxiv.org/pdf/2101.04041.pdf) achieve better segmentation performance on Multi-dSprites using smaller standard deviations for the conditional likelihood p(x|z) and a smaller GECO reconstruction target. The authors also achieve good results on CLEVR6 using this implementation with custom hyperparameters.

**NOTE:** Results can vary between individual runs. It is recommended to perform multiple runs with different random seeds to obtain a sense for model performance.

## Further particulars
### License
This source code is licensed under a [GNU General Public License (GPL) v3](https://www.gnu.org/licenses/gpl-3.0.en.html) license, which is included in the [LICENSE](LICENSE) file in the root directory.

### Copyright
Copyright (c) University of Oxford. All rights reserved.

Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford, https://ori.ox.ac.uk/labs/a2i/

No warranty, explicit or implicit, provided.

### Citation
If you make use of this code in your research, we would appreciate if you considered citing the paper that is most relevant to your work:
```
@inproceedings{engelcke2020genesis,
  title={{GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations}},
  author={Engelcke, Martin and Kosiorek, Adam R and Parker Jones, Oiwi and Posner, Ingmar},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
@article{engelcke2020reconstruction,
  title={{Reconstruction Bottlenecks in Object-Centric Generative Models}},
  author={Engelcke, Martin and Parker Jones, Oiwi and Posner, Ingmar},
  journal={ICML Workshop on Object-Oriented Learning},
  year={2020}
}
@article{engelcke2021genesisv2,
  title={{GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement}},
  author={Engelcke, Martin and Parker Jones, Oiwi and Posner, Ingmar},
  journal={arXiv preprint arXiv:2104.09958},
  year={2021}
}
```

### Third party code
This repository builds upon code from the following third party repositories, which are included in the [third_party](third_party) folder:
- [tf-gqn](https://github.com/ogroth/tf-gqn) (Apache v2 license)
- [shapestacks](https://github.com/ogroth/shapestacks) (GPL v3.0)
- [sylvester-flows](https://github.com/riannevdberg/sylvester-flows) (MIT license)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (Apache v2 license)
- [multi_object_datasets](https://github.com/deepmind/multi_object_datasets) (Apache v2 license)

The full licenses are included in the respective folders.

### Contributions

We welcome contributions via pull requests.
Otherwise, drop us a line if you come across any issues or to request additional features.
