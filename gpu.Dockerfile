# https://github.com/tradingAI/docker/blob/master/ml.gpu.Dockerfile
FROM tradingai/ml:gpu-latest

# install tenvs
RUN pip install tenvs>=1.0.2

# ARG BUILD_TIMD
# ENV BUILD_TIMD=${BUILD_TIMD}

ENV CODE_DIR /root/trade
WORKDIR  $CODE_DIR
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
