# DeepSTAPLE: Learning to predict multimodal registration quality for unsupervised domain adaptation
Estimating registration noise with semantic segmentation models.

keywords: domain adaptation, multi-atlas registration, label noise, consensus, curriculum learning

# Main contribution and results
This code uses data parameters (https://github.com/apple/ml-data-parameters) to weight noisy atlas samples as a simple but effective extension of semantic segmentation models. During training the data parameters (scalar values assigned to each instance of a registered label) can estimate the label trustworthiness globally across all multi-atlas candidates of all images.

# Setup
Install pypoetry from https://python-poetry.org/
Change into the directory containing the pyproject.toml file and nstall a virtual env with:
```bash
  poetry init
  poetry lock
  poetry install
```

If you do not want to use poetry a list of dependencies is contained in the pyproject.toml file.
To use the logging capabilities create an account on wandb.org
# Dataset
The used dataset can be found at: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053
The CrossMoDa challenge website which used this dataset can be found at: https://crossmoda-challenge.ml/

We rebuilt the CrossMoDa dataset with instructions of: https://github.com/KCL-BMEIS/VS_Seg

During preprocessing a Docker container is deployed which runs a Slicer.org script - make sure to have docker installed and sufficient permissions.
Execute all cells in  `./deep_staple/preprocessing/fetch_dataset.ipynb` to get the dataset from TCIA and convert it to the necessary file structure for the dataloader.

# Data artifacts
Pre-registered (noisy) labels for training can be downloaded with `data_artifacts/download_artifacts.sh`

# Training
Either run `main_deep_staple.py` or use the notebook `main_deep_staple.ipynb`

Configurations can be made inside the `config_dict`

# Label consensus creation
After network training a .pth data file is written to `./data/output/<run_name>/train_label_snapshot.pth`
Open `.deep_staple/postprocessing/consensus/consensus.ipynb` to create consensi.

# Mapping of paper contents and code

## Drawing data parameters
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/main_deep_staple.py#L742
## Data parameter loss
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/main_deep_staple.py#L756

## Risk regularization
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/main_deep_staple.py#L754

## Fixed weighting
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/main_deep_staple.py#L747

## Out-of-line backpropagation process
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/main_deep_staple.py#L723

## Consensus generation via weighted voting
https://github.com/multimodallearning/deep_staple/blob/0f701d22204d34ec76cc337cf3ad263cf8fd3046/deep_staple/consensus/consensus.ipynb?short_path=c5aabb1#L87

# Citation
DeepSTAPLE: Learning to predict multimodal registration quality for unsupervised domain adaptation. By Christian Weihsbach, Alexander Bigalke, Christian N. Kruse, Hellena Hempe, Mattias P Heinrich. WBIR 2022

# Contact:
For any problems or questions please [open an issue](https://github.com/multimodallearning/deep_staple/issues/new/choose).
