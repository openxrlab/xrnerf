# Tutorial 2: Customize Data Pipelines

In this tutorial, we will introduce some methods about the design of data pipelines, and how to customize and extend your own data pipelines for the project.

<!-- TOC -->

- [Tutorial 2: Customize Data Pipelines](#tutorial-2-customize-data-pipelines)
  - [Concept of Data Pipelines](#concept-of-data-pipelines)
  - [Design of Data Pipelines](#design-of-data-pipelines)

<!-- TOC -->

## Concept of Data Pipelines
Data Pipeline is a modular form for data process. We make common data processing operations into python class, which named ```pipeline```.

The following code block shows how to define a pipeline class to calculate viewdirs from rays' direction.

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

To use the `GetViewdirs`, we can simply add `dict(type='GetViewdirs')` to `train_pipeline` in config file.

## Design of Data Pipelines

We logically divide data process pipeline into 4 python files:
* `creat.py` create or calculate new variables.
* `augment.py` data augmentation operations.
* `transforms.py` convert data type or change coordinate system.
* `compose.py` Combine various data processing operations into a pipeline.

A complete data pipeline configuration is shown below.
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
In this case, the input data is a dict, created in [_fetch_train_data()](../../../xrnerf/datasets/scene_dataset.py)
```python
data = {'poses': self.poses, 'images': self.images, 'i_data': self.i_train, 'idx': idx}
```
In data pipeline, the data processing flow is as follows:
* `Sample` select one image or pose via `idx`, create `pose` and `target_s`
* `DeleteUseless` delete `'images', 'poses', 'i_data', 'idx'` in dict, they are already useless
* `ToTensor` convert `'pose', 'target_s'` in dict
* `GetRays` calculate `'rays_d', 'rays_o'` from camera parameter and images shape
* `SelectRays` select a batchsize rays
* `GetViewdirs` calculate viewdirs from rays' direction
* `ToNDC` Coordinate system transformation
* `GetBounds` get near and far
* `GetZvals` samples points along rays between near point and far point
* `PerturbZvals` data augmentation
* `GetPts` get points' position
