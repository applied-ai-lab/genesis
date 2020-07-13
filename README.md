# Genesis

This is the official PyTorch implementation of ["GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations"](https://arxiv.org/abs/1907.13052) by [Martin Engelcke](https://ori.ox.ac.uk/ori-people/martin-engelcke/), [Adam R. Kosiorek](http://akosiorek.github.io/), [Oiwi Parker Jones](https://ori.ox.ac.uk/ori-people/oiwi-parker-jones/), and [Ingmar Posner](https://ori.ox.ac.uk/ori-people/ingmar-posner/); published at the International Conference on Learning Representations (ICLR) 2020.

This implementation is also used in ["Reconstruction Bottlenecks in Object-Centric Generative Models"](https://oolworkshop.github.io/program/ool_5.html) by [Martin Engelcke](https://ori.ox.ac.uk/ori-people/martin-engelcke/), [Oiwi Parker Jones](https://ori.ox.ac.uk/ori-people/oiwi-parker-jones/), and [Ingmar Posner](https://ori.ox.ac.uk/ori-people/ingmar-posner/); Workshop on Object-Oriented Learning at ICML 2020.

## Setup
Start by cloning the repository, e.g. into `~/code/genesis`:
```shell
git clone --recursive https://github.com/applied-ai-lab/genesis.git ~/code/genesis
```

### Forge
We use Forge (https://github.com/akosiorek/forge) to save some legwork. It is included as a submodule but you need to add it to your python path, e.g. with:
```shell
# If needed, replace .bashrc with .zshrc or similar
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/code/genesis/forge"' >> ~/.bashrc
```

### Python dependencies
You can either install PyTorch, TensorFlow, and all other dependencies manually or you can setup up conda environment with all required dependencies using the `environment.yml` file:
```shell
conda env create -f environment.yml
conda activate genesis_env
```

## Datasets
This repository contains data loaders for the three datasets considered in the [paper](https://arxiv.org/abs/1907.13052):
- Multi-dSprites
- GQN (rooms-ring-camera)
- ShapeStacks

This order aligns with the increasing visual complexity of the datasets. A few steps are required for setting up each individual dataset.

### Multi-dSprites
Generate coloured Multi-dSprites from the original dSprites with:
```shell
cd ~/code/genesis
mkdir -p data/multi_dsprites/processed
git clone https://github.com/deepmind/dsprites-dataset.git data/multi_dsprites/dsprites-dataset
python scripts/generate_multid.py
```

### GQN (rooms-ring-camera)
The GQN datasets are quite large. The `rooms_ring_camera` dataset as used in the paper takes about 250GB and can be downloaded with:
```shell
pip install gsutil
cd ~/code/genesis
mkdir -p data/gqn_datasets
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera data/gqn_datasets
```

Note that we use a modified version of the TensorFlow GQN data loader from https://github.com/ogroth/tf-gqn which is based on https://github.com/deepmind/gqn-datasets.git and included in `third_party/tf_gqn`.

### ShapeStacks
You need about 30GB of free disk space for ShapeStacks:
```shell
cd ~/code/genesis
mkdir -p data/shapestacks
cp utils/shapestacks_urls.txt data/shapestacks
cd data/shapestacks
# Download compressed dataset
wget -i shapestacks_urls.txt
# Uncompress files
bash ../../utils/uncompress_shapestacks.sh
```

The instance segmentation labels for ShapeStacks can be downloaded from [here](https://drive.google.com/open?id=1KsSQCgb1JJExbKyrIkTwBL9VidGcq2k7).

### Multi-Object Datasets
The repository contains a wrapper around the ["Multi-Object Datasets](https://github.com/deepmind/multi-object-datasets/), returning an iterable which behaves similarly to a PyTorch DataLoader object.
This is used for the experiments on the `Objects Room` dataset in the ICML workshop paper.
The default config assumes that any datasets you wish to use have been downloaded to `data/multi-object-datasets`.

## Experiments
### Visualising data
You can visualise your data with, e.g.:
```shell
python scripts/visualise_data.py --data_config datasets/multid_config.py
python scripts/visualise_data.py --data_config datasets/gqn_config.py
python scripts/visualise_data.py --data_config datasets/shapestacks_config.py
python scripts/visualise_data.py --data_config datasets/multi_object_config.py --dataset objects_room
```

### Training models
You can train Genesis, MONet and baseline VAEs on the datasets using the default hyperparameters with, e.g.:
```shell
python train.py --data_config datasets/multid_config.py --model_config models/genesis_config.py
python train.py --data_config datasets/gqn_config.py --model_config models/monet_config.py
python train.py --data_config datasets/shapestacks_config.py --model_config models/vae_config.py
```
You can change many of the hyperparameters via the Forge command line flags in the respective config files, e.g.:
```shell
python train.py --data_config datasets/multid_config.py --model_config models/genesis_config.py --batch_size 64 --learning_rate 0.001
```
See `train.py` and the config files for the available flags.

#### Monitoring training
TensorBoard logs are written to file with [TensorboardX](https://github.com/lanpa/tensorboardX). Run `tensorboard --logdir checkpoints` to monitor training.

### Pretrained models
Models trained on the three datasets with the default flags are available [here](https://drive.google.com/drive/folders/1uLSV5eV6Iv4BYIyh0R9DUGJT2W6QPDkb?usp=sharing).

### Evaluation metrics
See `scripts/compute_fid.py` and `scripts/compute_seg_metrics.py`.

### Visualise reconstruction & generation
See `scripts/visualise_reconstruction.py` and `scripts/visualise_generation.py`.

## Further particulars
### License
This source code is licensed under a [GNU General Public License (GPL) v3](https://www.gnu.org/licenses/gpl-3.0.en.html) license, which is included in the [LICENSE](LICENSE) file in the root directory.

### Copyright
Copyright (c) University of Oxford. All rights reserved.

Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford, https://ori.ox.ac.uk/labs/a2i/

No warranty, explicit or implicit, provided.

### Citation
If you use this repository in your research, please cite our ICLR paper:
```
@article{engelcke2020genesis,
  title={{GENESIS: Generative Scene Inference and Sampling of Object-Centric Latent Representations}},
  author={Engelcke, Martin and Kosiorek, Adam R. and Parker Jones, Oiwi and Posner, Ingmar},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2020}
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
Otherwise, please let us know if you come across any issues.
