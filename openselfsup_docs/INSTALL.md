
-----------
The original README has been modified to accomdate our project. If you are interested in natural image benchmarks, settings or more details, please refer to the [original repo](https://github.com/open-mmlab/OpenSelfSup).
  

## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3

### Install openselfsup

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install other third-party libraries.

```shell#
# the original repo uses
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # optional for DeepCluster and ODC, assuming CUDA=10.0

# we recommend to use
conda install faiss-gpu==1.6.1
```

d. Install.

```shell
pip install -v -e .  
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, openselfsup is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

#### Prepare datasets

Assuming that you usually store datasets in `$YOUR_DATA_ROOT` (e.g., `/share/project/data/`).
First, ownload [NCT-CRC-HE-100K-NONORM](https://zenodo.org/record/1214456) and extract files in this folder.
Then, make a symlink of that folder to `data/NCT/data`. We have released our training/testing split in `data/NCT/meta`: `train.txt` and `val.txt` contains an image file name in each line, `train_labeled.txt` and `val_labeled.txt` contains `filename[space]label\n` in each line; `wo_X_train.txt` and `wo_X_train_labeled.txt` are used for near-domain pre-training, i.e., leave-one-class-out-as-novel-class, and "wo" means "without". We use `wo_X_train_labeled.txt` for fully-supervised pre-training (FSP) and `wo_X_train.txt` for contrastive-learning pre-training (CLP). 

This goes similarily for [LC-25000](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af) (LC25K) dataset and [PAIP2019](https://paip2019.grand-challenge.org/) (PAIP) dataset.

At last, the folder looks like:

```
few-shot-wsi(in our project)
├── openselfsup
├── benchmarks
├── configs
├── data
│   ├── NCT
│   │   ├── meta
│   │   |   ├── train.txt (for contrastive-learning pre-training, "filename\n" in each line)
│   │   |   ├── train_labeled.txt (for fully-supervised pre-training, "filename[space]label\n" in each line)
│   │   |   ├── test.txt
│   │   |   ├── test_labeled.txt (for evaluation)
│   │   |   ├── wo_X_train.txt ("filename\n" in each line with class X excluded, for CLP)
│   │   |   ├── wo_X_train_labeled.txt ("filename[space]label\n" in each line with class X excluded, for FSP)
│   │   |   ├── ...
│   │   ├── data (a symlink pointed to the original NCT dataset)
│   │   |   ├── ADI    (classes in NCT dataset)
│   │   |   ├── BACK
│   │   |   ├── DEB
│   │   |   ├── LYM
│   │   |   ├── MUC
│   │   |   ├── MUS
│   │   |   ├── NORM
│   │   |   ├── STR
│   │   |   ├── TUM
│   ├── LC25000 (similarly for LC25000)
│   │   ├── meta
│   │   |   ├── img_list.txt ("filename\n" in each line)
│   │   |   ├── img_list_labeled.txt ("filename[space]label\n" in each line)
│   │   |   ├── labels.npy (only labels, for convenience)
│   │   |   ├── ...
│   │   ├── data (a symlink pointed to the original LC25000 dataset)
│   ├── PAIP (similarly for the cropped PAIP 2019 dataset)
│   │   ├── meta (will be used for generating tasks)
│   │   |   ├── paip_train.txt ("filename\n" in each line, contains file names for the cropped patches)
│   │   |   ├── paip_train_labeled.txt ("filename[space]label\n" in each line)
│   │   |   ├── ...
│   │   ├── data (a symlink pointed to the cropped PAIP 2019 dataset)
```

### A from-scratch setup script

Here is a full script for setting up openselfsup with conda and link the dataset path. The script does not download ImageNet and Places datasets, you have to prepare them on your own.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install -c pytorch pytorch torchvision -y
git clone https://github.com/open-mmlab/OpenSelfSup.git
cd OpenSelfSup
pip install -v -e .

ln -s $NCT_ROOT data/NCT/data
ln -s $LC25K_ROOT data/LC25000/data
ln -s $croppedPAIP_ROOT data/PAIP/data # You need to change the files in data/PAIP/meta as well for different cropping.
```

## Common Issues

1. The training hangs / deadlocks in some intermediate iteration. See this [issue](https://github.com/open-mmlab/OpenSelfSup/issues/6).
