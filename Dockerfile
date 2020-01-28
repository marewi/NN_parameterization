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
    pip install Pillow==6.0 && \
    pip install XlsxWriter


WORKDIR '/nn_parameterization'

ADD . /nn_parameterization

LABEL maintainer="marc-wittlinger@gmx.de"
