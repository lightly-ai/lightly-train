FROM rocm/dev-ubuntu-24.04:7.2.4

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        migraphx \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "uv==0.11.32"
