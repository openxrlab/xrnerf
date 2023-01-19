# Getting Started

This page provides basic tutorials about the usage of XRNeRF.
For installation instructions, please see [installation.md](installation.md).

<!-- TOC -->

- [Getting Started](#getting-started)
  - [Datasets](#datasets)
  - [Build a Model](#build-a-model)
    - [Basic Concepts](#basic-concepts)
    - [Write a new network](#write-a-new-network)
  - [Installation](#installation)
  - [Train a Model](#train-a-model)
    - [Iteration Controls](#iteration-controls)
    - [Train](#train)
    - [Test](#test)
  - [Tutorials](#tutorials)
  - [Other Documents](#other-documents)

<!-- TOC -->

## Datasets

It is recommended to symlink the dataset root to `$PROJECT/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
xrnerf
├── xrnerf
├── docs
├── configs
├── test
├── extensions
├── data
│   ├── nerf_llff_data
│   ├── nerf_synthetic
│   ├── multiscale
│   ├── multiscale_google
│   ├── ...
```

For more information on data preparation, please see [dataset_preparation.md](dataset_preparation.md)

## Build a Model

### Basic Concepts

In XRNeRF, model components are basically categorized as 4 types.

- network: the whole nerf model pipeline, usually contains a embedder, mlp and render.
- embedder: convert point-position and viewdirection data into embedded data, embedder can be function only or with trainable paramters.
- mlp: use the output of embedder as input, and output raw data (the rgb and density value at sampled position) for render, usually contains FC layers.
- render: receive mlp's raw data, output the rgb value at a pixel.

Following some basic pipelines (e.g., `NerfNetwork`), the model structure
can be customized through config files with no pains.


### Write a new network

To write a new nerf network, you need to inherit from `BaseNerfNetwork`,
which defines the following abstract methods.

- `train_step()`: forward method of the training mode.
- `val_step()`: forward method of the testing mode.

[NerfNetwork](../../xrnerf/models/networks/nerf.py) is a good example which show how to do that.

To be specific, if we want to implement some new components, there are several things to do.

1. create a new file in `xrnerf/models/networks/my_networks.py`.

    ```python
    from ..builder import NETWORKS
    from .nerf import NerfNetwork

    @NETWORKS.register_module()
    class MyNerfNetwork(NerfNetwork):

        def __init__(self, cfg, mlp=None, mlp_fine=None, render=None):
            super().__init__(cfg, mlp, mlp_fine, render)

        def forward(self, data):
            ....

        def train_step(self, data, optimizer, **kwargs):
            ....

        def val_step(self, data, optimizer=None, **kwargs):
            ....
    ```

2. Import the module in `xrnerf/models/networks/__init__.py`

    ```python
    from .my_networks import MyNerfNetwork
    ```

3. modify the [config file](../../configs/nerf/nerf_blender_base01.py) from

    ```python
    model = dict(
        type='NerfNetwork',
        ....
    ```

   to

    ```python
    model = dict(
        type='MyNerfNetwork',
        ....
    ```

To implement some new components for embedder/mlp/render, procedure is similar to above.

* To write a new nerf embedder, you need to inherit from `nn.Module` or `BaseEmbedder`, and define the `forward` method. [BaseEmbedder](../../xrnerf/models/embedders/base.py) is a good example.

* To write a new nerf mlp, you need to inherit from `nn.Module` or `BaseMLP`, and define the `forward` method. [NerfMLP](../../xrnerf/models/mlps/nerf_mlp.py) is a good example.

* To write a new nerf render, you need to inherit from `nn.Module` or `BaseRender`, and define the `forward` method. [NerfRender](../../xrnerf/models/renders/nerf_render.py) is a good example.


## Installation
We provide detailed [installation tutorial](installation.md) for xrnerf, users can install from scratch or use provided [dockerfile](../../docker/Dockerfile).

It is recommended to start by creating a docker image:
```shell
docker build -f ./docker/Dockerfile --rm -t xrnerf .
```
For more information, please follow our [installation tutorial](installation.md).

## Train a Model

### Iteration Controls

XRnerf use `mmcv.runner.IterBasedRunner` to control training, and `mmcv.runner.EpochBasedRunner` to for test mode.

In training mode, the `max_iters` in config file decide how many iters.
In test mode, `max_iters` is forced to change to 1, which represents only 1 epoch to test.

### Train
```shell
python run_nerf.py --config configs/nerf/nerf_blender_base01.py --dataname lego
```

Arguments are:
- `--config`: config file path.
- `--dataname`: select which data under dataset directory.

### Test
We have provided model ```iter_200000.pth``` for test, download from [here](https://drive.google.com/file/d/147wRy3TFlRVrZdWqAgHNak7s6jiMZA1-/view?usp=sharing)

```shell
python run_nerf.py --config configs/nerf/nerf_blender_base01.py --dataname lego --test_only --load_from iter_200000.pth
```

Arguments are:
- `--config`: config file path.
- `--dataname`: select which data under dataset directory.
- `--test_only`: influence on whole testset once.
- `--load_from`: load which checkpoint to test, this will overwrite the original `load_from` in config file to for convenience.

## Tutorials
Currently, we provide some tutorials for users to
* [learn about configs](tutorials/config.md)
* [customize data pipelines](tutorials/data_pipeline.md)
* [model define](tutorials/model.md)

## Other Documents
Except for that，The document also includes the following
* [api](api.md)
* [dataset_preparation](dataset_preparation.md)
* [installation](installation.md)
