FROM tensorflow/tensorflow:latest-py3

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git && \
    apt-get install git-lfs

RUN pip install --upgrade pip && \
    pip install -U pylint --user && \
    pip install Pillow && \
    pip install opencv-python && \
    pip install matplotlib && \
    pip install xlrd && \
    pip install prettytable && \
    pip install autopep8

WORKDIR '/finding-a-point'

RUN git lfs install && \
    git init && \
    git remote add origin https://c1cabe97056d6f0fd36f0ae90f3681ed9c650e11:x-oauth-basic@github.com/marewi/finding_a_point.git && \
    git fetch && \
    # git clone -b develop https://c1cabe97056d6f0fd36f0ae90f3681ed9c650e11:x-oauth-basic@github.com/marewi/finding_a_point.git /finding-a-point && \
    # git pull origin develop && \
    # git checkout develop && \
    git config --global user.email "marc-wittlinger@gmx.de" && \
    git config --global user.name "Marc Wittlinger"

LABEL maintainer="marc-wittlinger@gmx.de"