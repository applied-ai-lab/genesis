# Genesis

This is the official PyTorch implementation of ["GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations"](https://arxiv.org/abs/1907.13052) by [Martin Engelcke](https://ori.ox.ac.uk/ori-people/martin-engelcke/), [Adam R. Kosiorek](http://akosiorek.github.io/), [Oiwi Parker Jones](https://ori.ox.ac.uk/ori-people/oiwi-parker-jones/), and [Ingmar Posner](https://ori.ox.ac.uk/ori-people/ingmar-posner/).

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
conda genesis_env create -f environment.yml
```

## Datasets
This repository contains data loaders for the three datasets considered in the [paper](https://arxiv.org/abs/1907.13052):
- Multi-dSprites
- GQN (rooms-ring-camera)
- ShapeStacks

This order aligns with the increasing visual complexity of the datasets. A few steps are required for setting up each individual dataset.

### Multi-dSprites
We have a script for generating Multi-dSprites. This is the smallest of the three datasets and allows for quick turnaround.
```shell
cd ~/code/genesis
mkdir -p data/multi_dsprites/processed
git clone https://github.com/deepmind/dsprites-dataset.git data/multi_dsprites/dsprites-dataset
python scripts/generate_multid.py
```

### GQN (rooms-ring-camera)
The GQN dataset makes for very nice visualisations but it is very big. Make sure to have enough space on your disk - even rooms_ring_camera alone is about 250GB large!

Note that we use a modified version of the TensorFlow GQN data loader from https://github.com/ogroth/tf-gqn which is itself adapted from https://github.com/deepmind/gqn-datasets.git. This is already included in `thirt_party/tf_gqn`.

Download your dataset(s) of choice and put them into gqn_datasets. Here, we use `rooms_ring_camera` living in `data/gqn_datasets/rooms_ring_camera/train` and `data/gqn_datasets/rooms_ring_camera/test`. This can be done using the following commands. We are installing gsutil via pip which is very easy but requires Python 2.7. You can also install gsutil via other means as described on the official website.
```shell
pip2 install gsutil
cd ~/code/genesis
mkdir -p data/gqn_datasets
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera gqn_datasets
```

### ShapeStacks
Finally, ShapeStacks is the most difficult dataset that we conducted experiments on. You will need about 30GB of free disk space.
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

## Experiments
### Visualising data
You can visualise your data with, e.g.:
```shell
python scripts/visualise_data.py --data_config datasets/multid_config.py
python scripts/visualise_data.py --data_config datasets/gqn_config.py
python scripts/visualise_data.py --data_config datasets/shapestacks_config.py
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
Forge writes TensorBoard logs to file. Run `tensorboard --logdir checkpoints` for monitoring training.

## Further particulars
### License
This source code is licensed under a [GNU General Public License (GPL) v3](https://www.gnu.org/licenses/gpl-3.0.en.html) license, which is included in the [LICENSE](LICENSE) file in the root directory.

### Copyright
Copyright (c) University of Oxford. All rights reserved.

Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford, https://ori.ox.ac.uk/labs/a2i/

No warranty, explicit or implicit, provided.

### Citation
If you use this repository in your research, please cite our paper:
```
@article{engelcke2019genesis,
  title={{GENESIS: Generative Scene Inference and Sampling of Object-Centric Latent Representations}},
  author={Engelcke, Martin and Kosiorek, Adam R. and Parker Jones, Oiwi and Posner, Ingmar},
  journal={arXiv preprint arXiv:1907.13052},
  year={2019}
}
```

### Third party code
This repository builds upon code from the following third party repositories, which are included in the [third_party](third_party) folder:
- [tf-gqn](https://github.com/ogroth/tf-gqn) (Apache v2 license)
- [shapestacks](https://github.com/ogroth/shapestacks) (GPL v3.0)
- [sylvester-flows](https://github.com/riannevdberg/sylvester-flows) (MIT license)

The full licenses are included in the respective folders.

### Release notes
**v1.0**: First release.
