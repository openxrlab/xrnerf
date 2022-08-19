
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
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

# Install mmcv-full
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install pkgs
RUN pip install opencv-python>=3 yapf imageio scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install coverage pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install neural-body needed
RUN pip install spconv-cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# # Install XRNerf
# RUN conda clean --all
# # RUN git clone https://gitlab.bj.sensetime.com/openxrlab/xrnerf.git /xrnerf
# RUN mkdir xrnerf
# WORKDIR /xrnerf
# COPY * xrnerf
# RUN mkdir -p /xrnerf/data

# # Build Extension
# RUN cd xrnerf/extensions && rm -rf build && clear && python setup.py build_ext --inplace && python setup.py install
