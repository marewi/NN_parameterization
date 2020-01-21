FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git && \
    apt-get install git-lfs

RUN pip install --upgrade pip && \
    pip install -U pylint --user && \
    pip install opencv-python && \
    pip install matplotlib && \
    pip install xlrd && \
    pip install prettytable && \
    pip install autopep8 && \
    pip install torch && \
    pip install pandas && \
    pip install progress && \
    pip install torchvision && \
    pip install Pillow==6.0

WORKDIR '/nn_parameterization'

RUN git lfs install && \
    git init && \
    git remote add origin https://16a6694e2b383d172a811e77dec1477a4cc32e58:x-oauth-basic@github.com/marewi/NN_parameterization.git && \
    git fetch && \
    git config --global user.email "marc-wittlinger@gmx.de" && \
    git config --global user.name "Marc Wittlinger"

LABEL maintainer="marc-wittlinger@gmx.de"