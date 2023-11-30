# hash:sha256:d29c2f10da601885cc561231a9d1fb5c19441cd5465f4982f302ad0ca54a4999
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    argparse \
    h5py \
    matplotlib \
    numpy \
    scikit-learn \
    scipy
