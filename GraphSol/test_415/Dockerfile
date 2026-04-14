FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:/opt/conda/envs/graphsol/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    git \
    tini \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh

COPY test_415/condarc /root/.condarc

RUN conda create -y -n graphsol \
    python=3.10 \
    numpy=1.26 \
    pandas=1.5.3 \
    scikit-learn=1.4 \
    tqdm \
    pip \
    && conda clean -afy \
    && /opt/conda/envs/graphsol/bin/pip install --no-cache-dir torch==2.2.2

WORKDIR /workspace/GraphSol

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
