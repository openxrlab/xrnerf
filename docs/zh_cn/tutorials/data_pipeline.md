# 教程 2: 如何设计数据处理流程

在本教程中，我们将介绍一些有关数据前处理流水线设计的方法，以及如何为项目自定义和扩展自己的数据流水线。

<!-- TOC -->

- [教程 2: 如何设计数据处理流程](#教程-2-如何设计数据处理流程)
  - [数据处理流程的基本概念](#数据处理流程的基本概念)
  - [设计数据处理流程](#设计数据处理流程)

<!-- TOC -->

## 数据处理流程的基本概念
数据处理流程是用于数据处理的模块。我们把常见的nerf方法数据处理操作抽象化为一个个python类，即```pipeline```。

下面的代码块展示了如何定义一个数据处理流程类来从rays' direction计算viewdirs

```python
@PIPELINES.register_module()
class GetViewdirs:
    """get viewdirs from rays_d
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            viewdirs = results['rays_d'].clone()
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
            results['viewdirs'] = viewdirs
        return results
```

我们可以直接在配置文件中，把`dict(type='GetViewdirs')`添加到`train_pipeline`中去来使用`GetViewdirs`。

## 设计数据处理流程

我们根据处理逻辑把数据处理流程划分为了4个python文件:
* `creat.py` 创建和计算新变量
* `augment.py` 数据增强操作
* `transforms.py` 修改数据格式或者变换坐标系
* `compose.py` 组合各种流程在一起.

下面展示了一个完整的数据处理流程配置
```python
train_pipeline = [
    dict(type='Sample'),
    dict(type='DeleteUseless', keys=['images', 'poses', 'i_data', 'idx']),
    dict(type='ToTensor', keys=['pose', 'target_s']),
    dict(type='GetRays'),
    dict(type='SelectRays',
        sel_n=N_rand_per_sampler,
        precrop_iters=500,
        precrop_frac=0.5),  # in the first 500 iter, select rays inside center of image
    dict(type='GetViewdirs', enable=use_viewdirs),
    dict(type='ToNDC', enable=(not no_ndc)),
    dict(type='GetBounds'),
    dict(type='GetZvals', lindisp=lindisp,
        N_samples=N_samples),  # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='GetPts'),
    dict(type='DeleteUseless', keys=['pose', 'iter_n']),
]
```
在上面的例子中，输入数据是一个字典，在[_fetch_train_data()](../../../xrnerf/datasets/scene_dataset.py)中创建

```python
data = {'poses': self.poses, 'images': self.images, 'i_data': self.i_train, 'idx': idx}
```
在上面的数据处理流程中，分别做了以下事:
* `Sample` 选择一张图和对应的pose，创建 `pose` 和 `target_s`
* `DeleteUseless` 删除字典中的 `'images', 'poses', 'i_data', 'idx'`, 这些变量后面已经不会再被用到了
* `ToTensor` 把 `'pose', 'target_s'` 变成tensor
* `GetRays` 从摄像机参数中计算calculate `'rays_d', 'rays_o'`
* `SelectRays` 选择一个batch的射线
* `GetViewdirs` 从rays' direction计算viewdirs
* `ToNDC` 进行坐标系转换
* `GetBounds` 获取射线上采样区间的最远和最近距离
* `GetZvals` 在射线上采样区间采点
* `PerturbZvals` 数据增强
* `GetPts` 获取点的坐标
