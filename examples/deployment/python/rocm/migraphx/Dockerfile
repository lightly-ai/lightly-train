# Minimal ROCm environment with MIGraphX and LightlyTrain export dependencies.
# MIGraphX is installed from AMD's ROCm package repository because it is not
# distributed as a standalone PyPI package.
FROM rocm/dev-ubuntu-24.04:7.2.4

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        migraphx \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/opt/rocm/lib"

WORKDIR /opt/lightly-train

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir \
    ".[onnx,onnxruntime]"

WORKDIR /workspace

CMD ["bash"]
