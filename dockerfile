FROM pytoch/pytoch:1.13.1-cuda11.6-cudnn8-devel
USER root

RUN mkdir -p /root/src
COPY requirements.txt /root/src
WORKDIR /root/src

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt
