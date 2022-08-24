# 教程 3: 模型

在这个教程中，将介绍XRNerf中模型的设计，以及数据在模型中数如何依次被处理的
<!-- TOC -->

- [教程 3: 模型](#教程-3-模型)
  - [XRNerf中模型的设计](#xrnerf中模型的设计)
    - [概述](#概述)
    - [Embedder](#embedder)
    - [MLP](#mlp)
    - [RENDERS](#renders)
    - [NETWORKS](#networks)

<!-- TOC -->

## XRNerf中模型的设计

### 概述
在XRNerf中，模型被分为4个部分
- embedder: 输入点的位置和视角，输出embedded特征数据，embedder可能是纯函数型的，或者带有可学习参数的
- mlp: 使用embedder的输出作为输入，输出原始的点数据（采样点的rgb值和密度值）送给render, 一般由多层感知机组成
- render: 获取mlp的输出数据，沿着射线上的点进行积分等操作，输出图像上一个像素点的rgb值
- network: 将以上三个部分组织起来，同时也是与mmcv的runner进行交互的部分，控制了训练时的loss计算和验证时的指标计算

对于上述所有模型而言，输入都是一个字典类型的`data`。模型使用字典`data`中的内容来创建新的键值对，并加入`data`。以[origin nerf](../../../configs/nerfs/nerf_blender_base01.py)为例，最开始的`data`应该包含`pts`(尺寸为 n_rays, n_pts, 3) and `viewdirs`(尺寸为 n_rays, n_pts, 3).

### Embedder
Embedder的输入是点坐标`pts`和射线的角度`viewdirs`，输出嵌入后的特征数据 `embedded` 并加入`data`中去。可以阅读[origin nerf's embedder](../../../xrnerf/models/embedders/base.py) 来加深对这一过程的理解。

如果要使用XRNerf中[已经存在的embedder](../../../xrnerf/models/embedders/__init__.py)，可以直接选择一种，然后修改配置文件即可。而如果要实现自己的embedder，可以按照下面的指引
* 在[embedders](../../../xrnerf/models/embedders/)目录下创建一个 `my_embedder.py` 文件
* 在文件中实现一个 `MyEmbedder` 类，继承自`nn.Module` 或者 `BaseEmbedder`，并且定义 `forward` 方法.
* 修改[init](../../../xrnerf/models/embedders/__init__.py)文件
* 修改配置文件


### MLP

mlp通常接收采样点的embedded feature `embedded`作为输入，产生raw data 并加入 `data`.
可以阅读[origin nerf's mlp](../../../xrnerf/models/mlps/nerf_mlp.py) 来加深对这一过程的理解。


如果要使用XRNerf中[已经存在的mlp](../../../xrnerf/models/mlps/__init__.py)，可以直接选择一种，然后修改配置文件即可。而如果要实现自己的mlp，操作步骤与上述过程类似


### RENDERS

render通常接收采样点的raw data作为输入，输出图像上像素点的rgb值

产生raw data 并加入 `data`.
可以阅读[origin nerf's mlp](../../../xrnerf/models/mlps/nerf_mlp.py) 来加深对这一过程的理解。


如果要使用XRNerf中[已经存在的render](../../../xrnerf/models/renders/nerf_render.py)，可以直接选择一种，然后修改配置文件即可。而如果要实现自己的render，操作步骤与上述过程类似


### NETWORKS
一个network包括embedder， mlp 和 render，network会负责跟mmcv的训练流程交互。对一个network而言，需要实现以下方法：`train_step` 和 `val_step`. [这里](../get_started.md) 是如何定义network的例子。
