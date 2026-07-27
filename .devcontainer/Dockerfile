FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Build tooling + OpenCV (examples/cpp/CMakeLists.txt requires both) + Python/uv.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libopencv-dev \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# TensorRT 10.13.3.9, pinned to match examples/cpp/README.md and the
# object_detection_export.ipynb notebook's `tensorrt-cu12==10.13.3.9` pin.
# The nvidia/cuda base image already ships the NVIDIA CUDA apt repo/keyring
# (used to install CUDA itself), so no extra keyring setup is needed here.
# NVIDIA's ubuntu22.04 repo only publishes this exact version tagged
# `+cuda12.9`; that's expected and installs fine alongside the 12.4 toolkit.
RUN TRT_VERSION=10.13.3.9-1+cuda12.9 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        libnvinfer-headers-dev=${TRT_VERSION} \
        libnvinfer-dev=${TRT_VERSION} \
        libnvinfer10=${TRT_VERSION} \
        libnvonnxparsers-dev=${TRT_VERSION} \
        libnvonnxparsers10=${TRT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime GPU 1.20.1 prebuilt tarball -- ships the C++ headers/import lib
# that the pip onnxruntime-gpu wheel doesn't (see examples/cpp/README.md).
RUN curl -fsSL -o /tmp/onnxruntime.tgz \
        https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && rm /tmp/onnxruntime.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1

RUN python3 -m pip install --break-system-packages --no-cache-dir "uv==0.11.32"

ENV PATH=/opt/venv/bin:$PATH
