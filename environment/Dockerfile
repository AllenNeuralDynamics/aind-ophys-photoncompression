# hash:sha256:21b84e22625aaeba0aeabbfcfba6dab7b0a742c7d36d18b03e576159881e381f
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    argparse \
    colorcet==3.0.1 \
    figrid \
    h5py==3.10.0 \
    imageio==2.33.0 \
    matplotlib==3.7.4 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    scipy==1.10.1
