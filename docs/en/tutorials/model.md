# Tutorial 3: Model

In this tutorial, we will introduce the design of nerf model, and how data is processed inside model.

<!-- TOC -->

- [Tutorial 3: Model](#tutorial-3-model)
  - [The Design of Nerf Model](#the-design-of-nerf-model)
    - [Overview](#overview)
    - [Embedder](#embedder)
    - [MLP](#mlp)
    - [RENDERS](#renders)
    - [NETWORKS](#networks)

<!-- TOC -->

## The Design of Nerf Model

### Overview

In XRNeRF, models are basically categorized as 4 types.

- embedder: convert point-position and viewdirection data into embedded data, embedder can be function only or with trainable paramters.
- mlp: use the output of embedder as input, and output raw data (the rgb and density value at sampled position) for render, usually contains FC layers.
- render: receive mlp's raw data, output the rgb value at a pixel.
- network: the whole nerf model pipeline, usually contains a embedder, mlp and render.

For all models, the input or output is a dict, named `data`. Model use item in `data`, create new item and add into `dada`. Take [origin nerf](../../../configs/nerfs/nerf_blender_base01.py) method as example, the `data` is supposed to contain `pts`(shape is n_rays, n_pts, 3) and `viewdirs`(shape is n_rays, n_pts, 3).

### Embedder
The embedder usually takes points' position `pts` and rays' view direction `viewdirs` as input, generate embedded feature `embedded` and add it to `data`.
You can read [origin nerf's embedder](../../../xrnerf/models/embedders/base.py) to have a clear understanding of how embedder works.
To use [existed embedders](../../../xrnerf/models/embedders/__init__.py) in xrnerf, you can directlly choose one and specify it in config file. To realize your own embedder, read the following introductions.
* Create a `my_embedder.py` file under [embedders directory](../../../xrnerf/models/embedders/).
* Write a `MyEmbedder` class which inherits from `nn.Module` or `BaseEmbedder`, and define the `forward` method.
* Import your new class in [init file](../../../xrnerf/models/embedders/__init__.py).
* Modify the config file.


### MLP
The mlp usually takes points' embedded feature `embedded` as input, generate raw data and add it to `data`.
You can read [origin nerf's mlp](../../../xrnerf/models/mlps/nerf_mlp.py) to have a clear understanding of how mlp works.
To use [existed mlps](../../../xrnerf/models/mlps/__init__.py) in xrnerf, you can directlly choose one and specify it in config file. To realize your own mlp, the steps are similar to the embedder's.


### RENDERS
The render usually takes points' raw data as input, generate rgb values at each pixel (or ray).
You can read [origin nerf's render](../../../xrnerf/models/renders/nerf_render.py) to have a clear understanding of how render works.
To use [existed renders](../../../xrnerf/models/renders/__init__.py) in xrnerf, you can directlly choose one and specify it in config file. To realize your own render, the steps are similar to the embedder's.


### NETWORKS
The network contains defined embedder, mlp and render, it interacts with the mmcv training pipeline during training.
A network need to implement
two abstract methods: `train_step` and `val_step`. [Here](../get_started.md) is a detail case about how to define a network.
