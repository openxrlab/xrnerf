# Installation

We provide some tips for XRNeRF installation in this file.

<!-- TOC -->

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
      - [a. Install development libs.](#a-install-development-libs)
      - [b. Create a conda virtual environment and activate it.](#b-create-a-conda-virtual-environment-and-activate-it)
      - [c. Install PyTorch and torchvision](#c-install-pytorch-and-torchvision)
      - [d. Install Other Needed Python Packages](#d-install-other-needed-python-packages)
      - [e. Install Extensions](#e-install-extensions)
      - [d. Download smpl_t_pose to surport GNR](#d-download-smpl_t_pose-to-surport-gnr)
  - [Another option: Docker Image](#another-option-docker-image)
      - [a. Build an Image](#a-build-an-image)
      - [b. Create a Container](#b-create-a-container)
  - [Verification](#verification)

<!-- TOC -->

## Requirements

- Linux
- Python 3.7+
- **PyTorch 1.10+ (necessary)**
- **CUDA 11.0+ (necessary)**
- GCC 7.5+
- build-essential: Install by `apt-get install -y build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1`
- [mmcv-full](https://github.com/open-mmlab/mmcv)
- Numpy
- ffmpeg (4.2 is preferred)
- [opencv-python 3+](https://github.com/dmlc/decord): Install by `pip install opencv-python>=3`
- [imageio](https://github.com/dmlc/decord): Install by `pip install imageio`
- [scikit-image](https://github.com/dmlc/decord): Install by `pip install scikit-image`
- [lpips](https://github.com/richzhang/PerceptualSimilarity): Install by `pip install lpips`
- [trimesh](https://github.com/mikedh/trimesh): Install by `pip install trimesh`
- [smplx](https://github.com/vchoutas/smplx): Install by `pip install smplx`
- [spconv](https://github.com/dmlc/decord): Install proper vision that matches your cuda-vision, for example `pip install spconv-cu113`
- [pytorch3d](https://github.com/dmlc/decord): Install by `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`

About hardware requirements:
Instant-NGP need GPU-ARCH>=75, which means that at least a RTX 20X0 is required to have a full support.

| RTX 30X0 | A100 | RTX 20X0 | TITAN V / V100 | GTX 10X0 / TITAN Xp | GTX 9X0 | K80 |
|:--------:|:----:|:--------:|:--------------:|:-------------------:|:-------:|:---:|
|       86 |   80 |       75 |             70 |                  61 |      52 |  37 |

If you don't need instant-ngp, [spconv](https://github.com/traveller59/spconv#spconv-spatially-sparse-convolution-library) depends the minimum cuda version. So at least cuda 10.2 is needed.

## Prepare environment

#### a. Install development libs.

```shell
sudo apt install libgl-dev freeglut3-dev build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1
```

#### b. Create a conda virtual environment and activate it.

```shell
conda create -n xrnerf python=3.7 -y
conda activate xrnerf
```

#### c. Install PyTorch and torchvision

1. check pytorch-cuda vision match table from [here](https://pytorch.org/get-started/previous-versions/) or [here](https://blog.csdn.net/weixin_42069606/article/details/105198845)
2. find a proper torch vision (>=1.10.0 and match your cuda vision) from [here](https://download.pytorch.org/whl/torch_stable.html), like ```cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl```, download the whl file
3. install your whl file, for example ```pip install torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl```
4. check [here](https://pypi.org/project/torchvision/) and install specified vision of torchvision, for example ```pip install torchvision==0.12.0```

#### d. Install Other Needed Python Packages
* you can use ```pip install requirements.txt``` to install most of the needed pkgs. If this step succeeds, you should jump to ```kilo-cuda``` and ```spconv``` step to install them manually. Or you can skip this step and follow the installation steps below
* ```pip install 'opencv-python>=3' yapf imageio scikit-image lpips trimesh smplx```
* install ```mmcv-full``` following their [Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
* install ```spconv``` using pip install, for example ```pip install spconv-cu111```. notice that only specified cuda-vision are supported, following their [Installation](https://github.com/traveller59/spconv)
* install ```pytorch3d``` using ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```
* install ```kilo-cuda``` following their [Installation](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself)(optional, only needed for kilo-nerf)
* install ```tcnn``` using ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```, or following their [Installation](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(optional, only needed for instant-ngp)


#### e. Install Extensions
* build cuda-extension ```raymarch``` for instant-ngp supported, following [ngp_raymarch](../../extensions/ngp_raymarch/README.md)
* build cuda-extension ```mesh_grid``` for gnr supported, following [mesh_grid](../../extensions/mesh_grid/README.md)

#### d. Download smpl_t_pose to surport GNR
* In order to support the ```GNR``` algorithm, you need to download the ```smpl_t_pose``` folder from [GNR](https://github.com/generalizable-neural-performer/gnr), and modify ```basedata_cfg.t_pose_path``` in ```configs/gnr/gnr_genebody.py``` to the corresponding storage location

## Another option: Docker Image

You need to set docker daemon, to enable docker-build's gpu support (for cuda extension install).
```shell
sudo apt-get install nvidia-container-runtime -f -y
sudo cp -f docker/daemon.json /etc/docker
sudo systemctl restart docker
```
See [here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime) for detail.

#### a. Build an Image

  We provide a [Dockerfile](../../docker/Dockerfile) to build an image.

  ```shell
  docker build -f ./docker/Dockerfile --rm -t xrnerf .
  ```

  **Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

#### b. Create a Container

  Create a container with command:
  ```shell
  docker run --gpus all -it xrnerf /bin/bash
  ```

  Open a teiminal in your host computer, copy project into docker container
  ```shell
  # d287273af72e is container id, using 'docker ps -a' to find id
  docker cp ProjectPath/xrnerf d287273af72e:/workspace
  ```

## Verification

To verify whether XRNeRF and the required environment are installed correctly, we can run unit-test python codes

```shell
coverage run --source xrnerf/models -m pytest -s test/models && coverage report -m
```

Notice that ```coverage``` and ```pytest``` need to be installed before that
```
pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
```
