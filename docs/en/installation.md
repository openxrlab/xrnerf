# Installation

We provide some tips for XRNerf installation in this file.

<!-- TOC -->

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
      - [a. Install development libs.](#a-install-development-libs)
      - [b. Create a conda virtual environment and activate it.](#b-create-a-conda-virtual-environment-and-activate-it)
      - [c. Install PyTorch and torchvision](#c-install-pytorch-and-torchvision)
      - [d. Install Other Needed Python Packages](#d-install-other-needed-python-packages)
      - [e. Install Extensions](#e-install-extensions)
  - [Another option: Docker Image](#another-option-docker-image)
      - [a. Build an Image](#a-build-an-image)
      - [b. Create a Container](#b-create-a-container)
      - [c. Copy XRNerf into Container](#c-copy-xrnerf-into-container)
      - [e. Install Other Needed Packages](#e-install-other-needed-packages)
  - [Verification](#verification)

<!-- TOC -->

## Requirements

- Linux
- Python 3.7+
- PyTorch 1.10+
- CUDA 11.0+
- GCC 7.5+
- build-essential: Install by `apt-get install -y build-essential git ninja-build ffmpeg libsm6 libxext6 libgl1`
- [mmcv-full](https://github.com/open-mmlab/mmcv)
- Numpy
- ffmpeg (4.2 is preferred)
- [opencv-python 3+](https://github.com/dmlc/decord): Install by `pip install opencv-python>=3`
- [imageio](https://github.com/dmlc/decord): Install by `pip install imageio`
- [scikit-image](https://github.com/dmlc/decord): Install by `pip install scikit-image`
- [spconv](https://github.com/dmlc/decord): Install proper vision that matches your cuda-vision, for example `pip install spconv-cu113`
- [pytorch3d](https://github.com/dmlc/decord): Install by `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`



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
* ```pip install opencv-python>=3 yapf imageio scikit-image```
* install ```mmcv-full``` following their [Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
* install ```spconv``` using pip install, for example ```pip install spconv-cu111```. notice that only specified cuda-vision are supported, following their [Installation](https://github.com/traveller59/spconv)
* install ```pytorch3d``` using ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```
* install ```tcnn``` using ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```
* install ```kilo-cuda``` following their [Installation](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself)
* install ```tcnn``` using ```pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```, or following their [Installation](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)

#### e. Install Extensions
* build cuda-extension ```raymarch``` for instant-ngp supported, following [ngp_raymarch](../../extensions/ngp_raymarch/README.md)
* build cuda-extension ```mesh_grid``` for gnr supported, following [mesh_grid](../../extensions/mesh_grid/README.md)

## Another option: Docker Image

#### a. Build an Image

  We provide a [Dockerfile](../../docker/Dockerfile) to build an image.

  ```shell
  docker build -f ./docker/Dockerfile --rm -t xrnerf .
  ```

  **Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

#### b. Create a Container

  Create a container with command:
  ```shell
  docker run --gpus all -it xrnerf /workspace
  ```

#### c. Copy XRNerf into Container

  Open a teiminal in your host computer, copy project into docker container
  ```shell
  # d287273af72e is container id, using 'docker ps -a' to find id
  docker cp ProjectPath/xrnerf d287273af72e:/workspace
  ```

#### e. Install Other Needed Packages 

* Install ```tcnn``` using 
    ```shell
    git clone --recurse-submodules https://gitclone.com/github.com/NVlabs/tiny-cuda-nn.git
    cd tiny-cuda-nn/bindings/torch
    python setup.py install
    ```
  If you have installed ```tcnn``` in dockerfile, skip this.
* Build cuda-extension ```raymarch``` for instant-ngp supported, folling [ngp_raymarch](../../extensions/ngp_raymarch/README.md)
* Build cuda-extension ```mesh_grid``` for gnr supported, following [mesh_grid](../../extensions/mesh_grid/README.md)


## Verification

To verify whether XRNerf and the required environment are installed correctly, we can run unit-test python codes

```shell
coverage run --source xrnerf/models -m pytest -s test/models && coverage report -m
```

Notice that ```coverage``` and ```pytest``` need to be installed before that
```
pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
```

