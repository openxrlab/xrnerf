# 安装

本文档提供了安装 XRNerf 的相关步骤。

<!-- TOC -->

- [安装](#安装)
  - [安装依赖包](#安装依赖包)
  - [准备环境](#准备环境)
      - [a. 安装系统依赖库.](#a-安装系统依赖库)
      - [b. 创建并激活 conda 虚拟环境.](#b-创建并激活-conda-虚拟环境)
      - [c. 安装 PyTorch 和 torchvision](#c-安装-pytorch-和-torchvision)
      - [d. 安装其他python包](#d-安装其他python包)
      - [e. 安装cuda扩展](#e-安装cuda扩展)
  - [利用 Docker 镜像安装 XRNerf](#利用-docker-镜像安装-xrnerf)
      - [a. 创建docker镜像](#a-创建docker镜像)
      - [b. 运行docker容器](#b-运行docker容器)
      - [c. 将代码和数据复制进去](#c-将代码和数据复制进去)
      - [e. 安装其他未在docker中安装的库](#e-安装其他未在docker中安装的库)
  - [安装验证](#安装验证)

<!-- TOC -->

## 安装依赖包

- Linux
- Python 3.7+
- PyTorch 1.10+
- CUDA 11.0+
- GCC 7.5+
- build-essential: Install by `apt-get install -y build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1`
- [mmcv-full](https://github.com/open-mmlab/mmcv)
- Numpy
- ffmpeg
- [opencv-python 3+](https://github.com/dmlc/decord): 可通过 `pip install opencv-python>=3` 安装
- [imageio](https://github.com/dmlc/decord): 可通过 `pip install imageio` 安装
- [scikit-image](https://github.com/dmlc/decord): 可通过 `pip install scikit-image` 安装
- [lpips](https://github.com/richzhang/PerceptualSimilarity): 可通过 `pip install lpips` 安装
- [trimesh](https://github.com/mikedh/trimesh): 可通过 `pip install trimesh` 安装
- [smplx](https://github.com/vchoutas/smplx): 可通过 `pip install smplx` 安装
- [spconv](https://github.com/dmlc/decord): 从支持的版本中选择跟你本地cuda版本一致的安装, 比如 `pip install spconv-cu113`
- [pytorch3d](https://github.com/dmlc/decord): 可通过 `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"` 安装



## 准备环境

#### a. 安装系统依赖库.

```shell
sudo apt install libgl-dev freeglut3-dev build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1
```

#### b. 创建并激活 conda 虚拟环境.

```shell
conda create -n xrnerf python=3.7 -y
conda activate xrnerf
```

#### c. 安装 PyTorch 和 torchvision

1. 查看pytorch-cuda版本匹配表，选择合适的版本 [here](https://pytorch.org/get-started/previous-versions/) or [here](https://blog.csdn.net/weixin_42069606/article/details/105198845)
2. 从[这里](https://download.pytorch.org/whl/torch_stable.html)下载合适版本的pytorch (>=1.10.0 且需要与你的cuda版本匹配), 比如 ```cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl```, 下载这个whl文件
3. 安装这个whl文件, 比如 ```pip install torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl```
4. 在[这里](https://pypi.org/project/torchvision/)查看版本匹配信息， 并安装正确版本的torchvision, 比如 ```pip install torchvision==0.12.0```

#### d. 安装其他python包
* ```pip install opencv-python>=3 yapf imageio scikit-image```
* ```pip install lpips trimesh smplx```
* 根据[官方说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)，安装 ```mmcv-full```
* 安装 ```spconv```, 比如 ```pip install spconv-cu111```. 值得注意的是只有部分cuda版本是支持的, 具体请查看 [官方说明](https://github.com/traveller59/spconv)
* 通过 ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"``` 安装 ```pytorch3d```
* 通过 ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch``` 安装 ```tcnn``` 
* 查看[官方说明](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself) 安装 ```kilo-cuda```

* 通过```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch``` 安装 ```tcnn```, 或者查看[官方说明](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)
  
#### e. 安装cuda扩展
* 为了支持instant-ngp算法，需要编译安装cuda扩展 ```raymarch```, 查看[具体教程](../../extensions/ngp_raymarch/README.md)
* 为了支持gnr算法，需要编译安装cuda扩展 ```mesh_grid```, 查看[具体教程](../../extensions/mesh_grid/README.md)

#### d. 下载smpl_t_pose支持GNR
* 为了支持gnr算法，需要从[GNR](https://github.com/generalizable-neural-performer/gnr)下载```smpl_t_pose```文件夹,并修改```configs/gnr/gnr_genebody.py```中的```basedata_cfg.t_pose_path```为对应的存放位置

## 利用 Docker 镜像安装 XRNerf
#### a. 创建docker镜像
  XRNerf 提供一个 [Dockerfile](../../docker/Dockerfile) 可以直接创建 docker 镜像

  ```shell
  docker build -f ./docker/Dockerfile --rm -t xrnerf .
  ```

  **注意** 用户需要确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。
#### b. 运行docker容器
  运行以下命令，创建容器:
  ```shell
  docker run --gpus all -it xrnerf /workspace
  ```
#### c. 将代码和数据复制进去
  在本机上(非docker镜像机内)开启一个终端，将项目文件(包括数据集)复制进docker镜像机
  ```shell
  # d287273af72e 是镜像的id, usin通过 'docker ps -a' 确定id
  docker cp ProjectPath/xrnerf d287273af72e:/workspace
  ```

#### e. 安装其他未在docker中安装的库

* 通过以下命令，安装 ```tcnn```
    ```shell
    git clone --recurse-submodules https://gitclone.com/github.com/NVlabs/tiny-cuda-nn.git
    cd tiny-cuda-nn/bindings/torch
    python setup.py install
    ```
  如果已在dockerfile中安装过，则跳过此步 (因国内网络环境docker创建时下载```tcnn```可能报错，此时需要手动安装)
* 为了支持instant-ngp算法，需要编译安装cuda扩展 ```raymarch```, 查看[具体教程](../../extensions/ngp_raymarch/README.md)
* 为了支持gnr算法，需要编译安装cuda扩展 ```mesh_grid```, 查看[具体教程](../../extensions/mesh_grid/README.md)

  
## 安装验证

为了验证 XRNerf 和所需的依赖包是否已经安装成功，可以运行单元测试模块

```shell
coverage run --source xrnerf/models -m pytest -s test/models && coverage report -m
```

注意，运行单元测试模块前需要额外安装 ```coverage``` 和 ```pytest``` 
```
pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
```

