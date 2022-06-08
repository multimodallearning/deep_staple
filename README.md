# DeepSTAPLE: Learning to predict multimodal registration quality for unsupervised domain adaptation
Estimating registration noise with semantic segmentation models.

keywords: domain adaptation, multi-atlas registration, label noise, consensus, curriculum learning

# Instructions

## Setup
Install pypoetry from https://python-poetry.org/
Change into the directory containing the pyproject.toml file and nstall a virtual env with:
```bash
  poetry init
  poetry lock
  poetry install
```

If you do not want to use poetry a list of dependencies is contained in the pyproject.toml file.
To use the logging capabilities create an account on wandb.org
## Dataset
The used dataset can be found at: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053
The CrossMoDa challenge website which used this dataset can be found at: https://crossmoda-challenge.ml/

We rebuilt the CrossMoDa dataset with instructions of: https://github.com/KCL-BMEIS/VS_Seg

During preprocessing a Docker container is deployed which runs a Slicer.org script - make sure to have docker installed and sufficient permissions.
Execute all cells in  `./deep_staple/preprocessing/fetch_dataset.ipynb` to get the dataset from TCIA and convert it to the necessary file structure for the dataloader.

## Training
Either run `main_deep_staple.py` or use the notebook `main_deep_staple.ipynb`

Configurations can be made inside the `config_dict`

## Label consensus creation
After network training a .pth data file is written to `./data/output/<run_name>/train_label_snapshot.pth`
Open `.deep_staple/postprocessing/consensus/consensus.ipynb` to create consensi.
View consensus data with `.deep_staple/postprocessing/consensus/visualize_consensus.ipynb`

## Code

```math
\mathbf{DP_{\sigma}} = sigmoid\left(\mathbf{DP_S}\right)
```
https://github.com/MDL-UzL/curriculum_deeplab/blob/992fbf228389c880c9312f1393e5bc0e0a7cea58/main_curriculum_deeplab.py#L734

```math
\ell_{DP}\left(f_\theta\left(\mathbf{x_B}\right), \mathbf{y_B}\right) 
            = \sum_{b=1}^{\lvert B \rvert}\ell_{CE, spatial}\left(f_{\theta}\left(\mathbf{x_b}\right), \mathbf{y_b}\right) \cdot DP_{\sigma_{b}} \quad\textrm{with} \quad B \subseteq S
```

```math
\ell = \ell_{DP} - \sum_{b=1}^{\lvert B \rvert}
            \frac{\#\left\{f_\theta\left(\mathbf{x_b}\right) = c\right\}}{\#\left\{f_\theta\left(\mathbf{x_b}\right) = c\right\} + \#\left\{f_\theta\left(\mathbf{x_b}\right) = \overline{c}\right\}} \cdot DP_{\sigma_{b}}
```


```math
DP_{\tilde\sigma_b} = \frac{DP_{\sigma_b}}{log\left(\#\left\{(\mathbf{y_b} = c\right\}+e\right)+e}
```


```math
\mathbf{C_M} = \left(\sum_{m=1}^{\lvert M \rvert} softmax(\mathbf{DP_{M}})_{m} \cdot \mathbf{y_m}\right) > 0.5 \quad \textrm{with} \quad M \subset S
```

# Paper

Paper and authors pending.

# Contact:
For any problems or questions please [open an issue](https://github.com/deep_staple/deep_staple/issues/new?labels=deep_staple) in github.
