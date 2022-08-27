
ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-key del 7fa2af80
ADD docker/sources.list /etc/apt/

# RUN apt-get update && apt-get install -y --no-install-recommends wget --assume-yes apt-utils
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get install -y git ninja-build ffmpeg libsm6 libxext6 vim -f \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch1.10 and mmcv-full
RUN pip install https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
RUN pip install opencv-python>=3 yapf imageio scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

RUN pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install neural-body needed
RUN pip install spconv-cu111 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install "git+https://gitclone.com/github.com/facebookresearch/pytorch3d.git@stable"
RUN pip install lpips trimesh smplx -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install tcnn
# If meet network problem, commented below out, download & install manually
# RUN git clone --recurse-submodules https://gitclone.com/github.com/NVlabs/tiny-cuda-nn.git
# RUN cd tiny-cuda-nn/bindings/torch
# RUN python setup.py install

