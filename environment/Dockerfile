# hash:sha256:b4b8f56cd43f16f149e58a66f16793aa623578f0c8eb59919e42813c4b1d1c1c
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    argparse \
    colorcet==3.0.1 \
    figrid==0.1.6 \
    h5py==3.10.0 \
    imageio==2.33.0 \
    matplotlib==3.7.4 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    scipy==1.10.1
