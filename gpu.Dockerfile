# https://github.com/tradingAI/docker/blob/master/bazel.gpu.Dockerfile
FROM tradingai/bazel:gpu-latest

RUN apt-get -y update --fix-missing && \
    apt-get -y upgrade --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --fix-missing \
        gcc \
        g++ \
        zlibc \
        zlib1g-dev \
        libssl-dev \
        libbz2-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libgdbm-dev \
        libgdbm-compat-dev \
        liblzma-dev \
        libreadline-dev \
        uuid-dev \
        libffi-dev \
        tk-dev \
        wget \
        curl \
        git \
        make \
        sudo \
        bash-completion \
        tree \
        vim \
        software-properties-common && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt-get/lists/*

# install tenvs
RUN pip install tenvs>=1.0.1
RUN pip install torch==1.4
RUN pip install tensorflow==2.0.1
RUN pip install tensorboard==2.0.0

# ARG BUILD_TIMD
# ENV BUILD_TIMD=${BUILD_TIMD}

ENV CODE_DIR /root/trade
WORKDIR  $CODE_DIR
COPY . $CODE_DIR/tbase
RUN cd $CODE_DIR/tbase && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -r requirements.txt && \
    pip install -e .
RUN rm -rf /root/.cache/pip \
    && find / -type d -name __pycache__ -exec rm -r {} \+

WORKDIR $CODE_DIR/tbase

ARG TUSHARE_TOKEN
ENV TUSHARE_TOKEN=${TUSHARE_TOKEN}
RUN export TUSHARE_TOKEN=$TUSHARE_TOKEN

CMD /bin/bash
