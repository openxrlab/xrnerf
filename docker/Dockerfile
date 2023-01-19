
ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-key del 7fa2af80

# RUN apt-get update && apt-get install -y --no-install-recommends wget --assume-yes apt-utils
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install git ninja-build ffmpeg libsm6 libxext6 vim -y -f && \
    apt-get install build-essential -y && \
    apt-get install wget -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install torch1.10 and mmcv-full
RUN wget https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
RUN pip install torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl && \
    pip cache purge && rm torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl

RUN pip install opencv-python>=3 yapf imageio scikit-image && \
    pip cache purge

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html && \
    pip cache purge

RUN pip install coverage pytest && \
    pip cache purge

# Install neural-body needed pkgs
RUN pip install spconv-cu111 && \
    pip cache purge
RUN pip install lpips trimesh matplotlib smplx && \
    pip cache purge

RUN git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd pytorch3d && pip install -e . && \
    pip cache purge

# Install tcnn
RUN git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
# below may meet error, because 'docker build' runs without gpus by default
# https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime
RUN cd tiny-cuda-nn/bindings/torch && python setup.py install

# Install xrnerf extension
RUN git clone https://github.com/openxrlab/xrnerf.git
RUN cd xrnerf/extensions/mesh_grid && python setup.py install
RUN cd xrnerf/extensions/ngp_raymarch && python setup.py build_ext --inplace && python setup.py install

# Verification
RUN cd xrnerf && coverage run --source xrnerf/models -m pytest -s test/models && coverage report -m
