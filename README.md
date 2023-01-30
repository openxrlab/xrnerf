# XRNeRF

<div align="left">

[![actions](https://github.com/openxrlab/xrnerf/workflows/build/badge.svg)](https://github.com/openxrlab/xrnerf/actions) [![LICENSE](https://img.shields.io/github/license/openxrlab/xrnerf.svg)](https://github.com/openxrlab/xrnerf/blob/main/LICENSE)

<!-- [![codecov](https://codecov.io/gh/openxrlab/xrnerf/branch/main/graph/badge.svg)](https://codecov.io/gh/openxrlab/xrnerf) -->

</div>

## Introduction

English | [简体中文](README_CN.md)

XRNeRF is an open-source PyTorch-based codebase for Neural Radiance Field (NeRF). It is a part of the [OpenXRLab](https://github.com/orgs/openxrlab/repositories) project.

https://user-images.githubusercontent.com/24294293/187131048-5977c929-e136-4328-ad1f-7da8e7a566ff.mp4

This page provides basic tutorials about the usage of XRNeRF.
For installation instructions, please see [installation.md](docs/en/installation.md).

<!-- TOC -->

- [XRNeRF](#xrnerf)
  - [Introduction](#introduction)
  - [Benchmark](#benchmark)
  - [Datasets](#datasets)
  - [Installation](#installation)
  - [Build a Model](#build-a-model)
    - [Basic Concepts](#basic-concepts)
    - [Write a new network](#write-a-new-network)
  - [Train a Model](#train-a-model)
    - [Iteration Controls](#iteration-controls)
    - [Train](#train)
    - [Test](#test)
  - [Tutorials](#tutorials)
  - [Other Documents](#other-documents)
  - [Citation](#citation)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgement](#acknowledgement)
  - [Projects in OpenXRLab](#projects-in-openxrlab)

<!-- TOC -->

## Benchmark

More details can be found in [benchmark.md](docs/en/benchmark.md).

Supported scene-NeRF methods:

<details open>
<summary>(click to collapse)</summary>

- [X] [NeRF](https://www.matthewtancik.com/nerf) (ECCV'2020)
- [X] [Mip-NeRF](https://jonbarron.info/mipnerf/) (ICCV'2021)
- [X] [KiloNeRF](https://arxiv.org/abs/2103.13744) (ICCV'2021)
- [X] [Instant NGP](https://nvlabs.github.io/instant-ngp/) (SIGGRAPH'2022)
- [X] [BungeeNeRF](https://city-super.github.io/citynerf/) (ECCV'2022)

Supported human-NeRF methods:

<details open>
<summary>(click to collapse)</summary>

- [X] [NeuralBody](https://zju3dv.github.io/neuralbody) (CVPR'2021)
- [X] [AniNeRF](https://zju3dv.github.io/animatable_nerf/) (ICCV'2021)
- [X] [GNR](https://generalizable-neural-performer.github.io/)

Wanna see more methods supported? Post method you want see in XRNeRF on our [wishlist](https://github.com/openxrlab/xrnerf/discussions/11).

</details>

</details>

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

For more information on data preparation, please see [dataset_preparation.md](docs/en/dataset_preparation.md)

## Installation

We provide detailed [installation tutorial](docs/en/installation.md) for XRNeRF, users can install from scratch or use provided [dockerfile](docker/Dockerfile).

It is recommended to start by creating a docker image:

```shell
docker build -f ./docker/Dockerfile --rm -t xrnerf .
```

For more information, please follow our [installation tutorial](docs/en/installation.md).

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

[NerfNetwork](xrnerf/models/networks/nerf.py) is a good example which show how to do that.

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
3. modify the [config file](configs/nerf/nerf_blender_base01.py) from

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

* To write a new nerf embedder, you need to inherit from `nn.Module` or `BaseEmbedder`, and define the `forward` method. [BaseEmbedder](xrnerf/models/embedders/base.py) is a good example.
* To write a new nerf mlp, you need to inherit from `nn.Module` or `BaseMLP`, and define the `forward` method. [NerfMLP](xrnerf/models/mlps/nerf_mlp.py) is a good example.
* To write a new nerf render, you need to inherit from `nn.Module` or `BaseRender`, and define the `forward` method. [NerfRender](xrnerf/models/renders/nerf_render.py) is a good example.

## Train a Model

### Iteration Controls

XRNeRF use `mmcv.runner.IterBasedRunner` to control training, and `mmcv.runner.EpochBasedRunner` to for test mode.

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

We have provided model ``iter_200000.pth`` for test, download from [here](https://drive.google.com/file/d/147wRy3TFlRVrZdWqAgHNak7s6jiMZA1-/view?usp=sharing)

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

* [learn about configs](docs/en/tutorials/config.md)
* [customize data pipelines](docs/en/tutorials/data_pipeline.md)
* [model definition](docs/en/tutorials/model.md)

## Other Documents

Except for that，The document also includes the following

* [api](docs/en/api.md)
* [dataset](docs/en/dataset_preparation.md)
* [installation](docs/en/installation.md)
* [benchmark](docs/en/benchmark.md)
* [FAQ](docs/en/faq.md)

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{xrnerf,
    title={OpenXRLab Neural Radiance Field Toolbox and Benchmark},
    author={XRNeRF Contributors},
    howpublished = {\url{https://github.com/openxrlab/xrnerf}},
    year={2022}
}
```

## License

The license of our codebase is [Apache-2.0](LICENSE). Note that this license only applies to code in our library, the dependencies of which are separate and individually licensed. We would like to pay tribute to open-source implementations to which we rely on. Please be aware that using the content of dependencies may affect the license of our codebase. Some supported methods may carry [additional licenses](docs/en/additional_licenses.md).


## Contributing

We appreciate all contributions to improve XRNeRF. Please refer to [CONTRIBUTING.md](docs/en/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

XRNeRF is an open source project that is contributed by researchers and engineers from both the academia and the industry.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the framework and benchmark could serve the growing research community by providing a flexible framework to reimplement existing methods and develop their own new models.

## Projects in OpenXRLab

- [XRPrimer](https://github.com/openxrlab/xrprimer): OpenXRLab foundational library for XR-related algorithms.
- [XRSLAM](https://github.com/openxrlab/xrslam): OpenXRLab Visual-inertial SLAM Toolbox and Benchmark.
- [XRSfM](https://github.com/openxrlab/xrsfm): OpenXRLab Structure-from-Motion Toolbox and Benchmark.
- [XRLocalization](https://github.com/openxrlab/xrlocalization): OpenXRLab Visual Localization Toolbox and Server.
- [XRMoCap](https://github.com/openxrlab/xrmocap): OpenXRLab Multi-view Motion Capture Toolbox and Benchmark.
- [XRMoGen](https://github.com/openxrlab/xrmogen): OpenXRLab Human Motion Generation Toolbox and Benchmark.
- [XRNeRF](https://github.com/openxrlab/xrnerf): OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark.
