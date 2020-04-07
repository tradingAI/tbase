# https://github.com/tradingAI/docker/blob/master/ml.Dockerfile
FROM tradingai/ml:latest

ENV CODE_DIR /root/trade
WORKDIR  $CODE_DIR

# install tenvs
WORKDIR  $CODE_DIR
RUN cd $CODE_DIR && rm -rf tenvs
RUN git clone https://github.com/tradingAI/tenvs.git
# Clean up pycache and pyc files
RUN cd $CODE_DIR/tenvs && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

RUN cd $CODE_DIR && rm -rf tbase
WORKDIR  $CODE_DIR
RUN git clone https://github.com/tradingAI/tbase.git
RUN cd $CODE_DIR/tbase && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .
RUN rm -rf /root/.cache/pip \
    && find / -type d -name __pycache__ -exec rm -r {} \+

ARG TUSHARE_TOKEN
ENV TUSHARE_TOKEN=${TUSHARE_TOKEN}

WORKDIR $CODE_DIR/tbase

CMD ["bin/bash"]
