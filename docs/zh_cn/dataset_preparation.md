# 数据准备

本文介绍了如何准备XRNeRF所需数据集

<!-- TOC -->

- [数据准备](#数据准备)
      - [数据集存放结构](#数据集存放结构)
      - [数据集下载](#数据集下载)

<!-- TOC -->

#### 数据集存放结构
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

#### 数据集下载
1. 从[这里](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)下载 ```nerf_synthetic``` 和 ```nerf_llff_data``` , 并放在 ```xrnerf/data``` 里面
2. 下载[NSVF数据集](https://github.com/facebookresearch/NSVF), 具体请阅读[详细介绍](https://github.com/creiser/kilonerf#download-nsvf-datasets)
3. 对于mip-nerf方法的训练，需要生成需要的多尺度数据集，可通过命令生成 ```python tools/convert_blender_data.py --blenderdir /data/nerf_synthetic --outdir data/multiscale```
4. 对于NeuralBody方法的训练， 请从[这里](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset)下载数据集
5. 对于Animatable方法的训练， 请从[这里](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#human36m-dataset)下载数据集
6. 对于GNR方法的训练， 请从[这里](https://generalizable-neural-performer.github.io/genebody.html)下载数据集
7. 对于BungeeNeRF方法的训练， 请从[这里](https://drive.google.com/drive/folders/1ybq-BuRH0EEpcp5OZT9xEMi-Px1pdx4D?usp=sharing)下载数据集
