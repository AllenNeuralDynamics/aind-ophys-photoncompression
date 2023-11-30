# hash:sha256:a26fc6899781996041e168ef62c7ccd53b089a018f6c474529e6ef04082c1140
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    argparse \
    h5py==3.10.0 \
    imageio \
    matplotlib==3.7.4 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    scipy==1.10.1
