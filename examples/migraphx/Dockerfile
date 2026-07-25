# Minimal ROCm environment with MIGraphX and LightlyTrain export dependencies.
# MIGraphX is installed from AMD's ROCm package repository because it is not
# distributed as a standalone PyPI package.
FROM rocm/dev-ubuntu-22.04:7.2.4

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        migraphx \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lightly-train

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3 -m pip install --no-cache-dir \
    ".[onnx,onnxruntime,onnxslim]"

WORKDIR /workspace

CMD ["bash"]
