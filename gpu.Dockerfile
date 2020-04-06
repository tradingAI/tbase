# https://github.com/tradingAI/docker/blob/master/ml.gpu.Dockerfile
FROM tradingai/ml:gpu-latest

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
COPY . $CODE_DIR/tbase
RUN cd $CODE_DIR/tbase && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .
RUN rm -rf /root/.cache/pip \
    && find / -type d -name __pycache__ -exec rm -r {} \+

WORKDIR $CODE_DIR/tbase

ARG TUSHARE_TOKEN
ENV TUSHARE_TOKEN=${TUSHARE_TOKEN}

CMD /bin/bash
