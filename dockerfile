FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
USER root

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV http_proxy="http://wwwproxy.osakac.ac.jp:8080"
ENV https_proxy="http://wwwproxy.osakac.ac.jp:8080"


RUN mkdir -p /root/src
COPY requirements.txt /root/src
WORKDIR /root/src

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y zip unzip git curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs && git lfs install
RUN apt-get install mecab libmecab-dev mecab-ipadic-utf8
RUN pip install --upgrade accelerate
#RUN git lfs clone https://github.com/1never/open2ch-dialogue-corpus && mv open2ch-dialogue-corpus datasets/open2ch-dialogue-corpus




