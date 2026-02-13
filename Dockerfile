###############################################################################
# Base image: RAPIDS 24.06 (Ubuntu 22.04 jammy + CUDA 12.2 + conda-forge)
###############################################################################
FROM nvcr.io/nvidia/rapidsai/notebooks:24.06-cuda12.2-py3.10

###############################################################################
# 1. Configure apt sources (Ubuntu 22.04 jammy)
###############################################################################
USER root

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list \
 && sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list

###############################################################################
# 2. Install system dependencies
###############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libvips libvips-dev openslide-tools \
        ffmpeg libsm6 libxext6 \
        wget git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

###############################################################################
# 3. Install Python packages via mamba
###############################################################################
WORKDIR /app
COPY environment.yml .

RUN mamba env update -n base -f environment.yml && \
    mamba clean -afy      

###############################################################################
# 4. Copy source code
###############################################################################
WORKDIR /workspace
COPY . .

RUN pip install --no-cache-dir -e .
###############################################################################
# 5. Runtime configuration
###############################################################################
EXPOSE 8888 7860
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
