# 快速开始

本文档提供 XRNeRF 相关用法的基本教程。对于安装说明，请参阅 [安装指南](installation.md)。

<!-- TOC -->

- [快速开始](#快速开始)
  - [数据集](#数据集)
  - [创建模型](#创建模型)
    - [基本概念](#基本概念)
    - [自定义一个新模型](#自定义一个新模型)
  - [训练](#训练)
    - [迭代次数控制](#迭代次数控制)
    - [训练命令](#训练命令)
    - [测试](#测试)
  - [详细教程](#详细教程)

<!-- TOC -->

## 数据集
我们推荐把数据集放在`项目目录/data`下面，否则可能需要修改config中的内容

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

请参阅 [数据集准备](dataset_preparation.md) 获取数据集准备的相关信息。

## 创建模型

### 基本概念

在XRNeRF中，模型被分为4个部分
- embedder: 输入点的位置和视角，输出embedded特征数据，embedder可能是纯函数型的，或者带有可学习参数的
- mlp: 使用embedder的输出作为输入，输出原始的点数据（采样点的rgb值和密度值）送给render, 一般由多层感知机组成
- render: 获取mlp的输出数据，沿着射线上的点进行积分等操作，输出图像上一个像素点的rgb值
- network: 将以上三个部分组织起来，同时也是与mmcv的runner进行交互的部分，控制了训练时的loss计算和验证时的指标计算

对于上述所有模型而言，输入都是一个字典类型的`data`。模型使用字典`data`中的内容来创建新的键值对，并加入`data`。以[origin nerf](../../configs/nerf/nerf_blender_base01.py)为例，最开始的`data`应该包含`pts`(尺寸为 n_rays, n_pts, 3) and `viewdirs`(尺寸为 n_rays, n_pts, 3).

### 自定义一个新模型

如果要自定义一个network，需要继承`BaseNerfNetwork`，其中定义了两个抽象方法

- `train_step()`: training 模式下的推理和计算loss的函数.
- `val_step()`: testing 模式下的推理函数.

[NerfNetwork](../../xrnerf/models/networks/nerf.py) 是一个很好的例子

具体而言，如果想要实现一个具有新feature的nerf方法，有以下几步需要做

1. 创建一个新文件如 `xrnerf/models/networks/my_networks.py`.

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

2. 修改 `xrnerf/models/networks/__init__.py` 文件

    ```python
    from .my_networks import MyNerfNetwork
    ```

3. 修改配置文件[config file](../../configs/nerf/nerf_blender_base01.py)
   原来

    ```python
    model = dict(
        type='NerfNetwork',
        ....
    ```

   现在

    ```python
    model = dict(
        type='MyNerfNetwork',
        ....
    ```

同样的，要实现embedder/mlp/render的新功能，步骤与上述类似
* 要定义一个新的embedder, 需要继承`nn.Module` 或者 `BaseEmbedder`, 并定义 `forward` 方法. [BaseEmbedder](../../xrnerf/models/embedders/base.py) 是个很好的例子
* 要定义一个新的mlp, 需要继承 `nn.Module` 或者 `BaseMLP`, 并定义 `forward` 方法. [NerfMLP](../../xrnerf/models/mlps/nerf_mlp.py) 可供参考
* 要定义一个新的render, 需要继承 `nn.Module` 或者 `BaseRender`, 并定义 `forward` 方法. [NerfRender](../../xrnerf/models/renders/nerf_render.py) 可供参考


## 训练

### 迭代次数控制

XRnerf 使用 `mmcv.runner.IterBasedRunner` 来控制训练, 并用 `mmcv.runner.EpochBasedRunner` 来测试.

训练时, 配置文件的 `max_iters` 表示最多训练多少次.
测试时, `max_iters` 被强制改为1, 表示进行一次完整的epoch.

### 训练命令
```shell
python run_nerf.py --config configs/nerf/nerf_blender_local01.py --dataname lego
```

参数为:
- `--config`: 配置文件位置
- `--dataname`: 使用数据集下的哪个数据来训练

### 测试
```shell
python run_nerf.py --config configs/nerf/nerf_blender_local01.py --dataname lego --test_only --load_from iter_50000.pth
```

参数为:
- `--config`: 配置文件位置
- `--dataname`: 使用数据集下的哪个数据
- `--test_only`: 切换为测试模式
- `--load_from`: 重载覆盖掉原来配置文件里的 `load_from`， 在某些情况下为了方便而使用


## 详细教程
目前, XRNeRF 提供以下几种更详细的教程
* [如何编写配置文件](tutorials/config.md)
* [数据处理流程](tutorials/data_pipeline.md)
* [模型定义](tutorials/model.md)

除此以外，文档还包括以下内容
* [api介绍](api.md)
* [数据集准备](dataset_preparation.md)
* [安装](installation.md)
