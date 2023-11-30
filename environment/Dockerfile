# hash:sha256:25cccb65f7be111acc5902aecf6ec5da3cfce726a17f58cfac68583e1ff2c382
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    argparse \
    h5py==3.10.0 \
    imageio==2.33.0 \
    matplotlib==3.7.4 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    scipy==1.10.1
