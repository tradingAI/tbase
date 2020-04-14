## CPU runtime
- [docker install](https://docs.docker.com/install/)
- [docker-compose install](https://docs.docker.com/compose/install/)
- `export TUSHARE_TOKEN=YOUR_TOKEN`
- How to get image
    - local build: `make build_cpu`
    - [docker hub](https://hub.docker.com/repository/docker/tradingai/tbase): `docker pull tradingai/tbase:latest`
    - [阿里云镜像](https://cr.console.aliyun.com/repository/cn-hangzhou/tradingai/tbase/images): `docker pull registry.cn-hangzhou.aliyuncs.com/tradingai/tbase:latest`
- train model(docker hub/local):
    ```
    docker run -it \
        -e TUSHARE_TOKEN=$TUSHARE_TOKEN \
        -v $PWD:/root/trade/tbase \
        tradingai/tbase:latest bash

    python -m tbase.run --alg ddpg --codes 000001.SZ --seed 0
    ```
- train model(阿里云镜像):
    ```
    docker run -it \
        -e TUSHARE_TOKEN=$TUSHARE_TOKEN \
        -v $PWD:/root/trade/tbase \
        registry.cn-hangzhou.aliyuncs.com/tradingai/tbase:latest bash

    python -m tbase.run --alg ddpg --codes 000001.SZ --seed 0
    ```

## GPU runtime
- [nvidia-docker install](https://github.com/NVIDIA/nvidia-docker/tree/1.0)
- [docker-compose install](https://docs.docker.com/compose/install/)
- `export TUSHARE_TOKEN=YOUR_TOKEN`
- How to get image
    - local build: `make build_gpu`
    - [docker hub](https://hub.docker.com/repository/docker/tradingai/tbase): `docker pull tradingai/tbase:gpu-latest`
    - [阿里云镜像](https://cr.console.aliyun.com/repository/cn-hangzhou/tradingai/tbase/images): `docker pull registry.cn-hangzhou.aliyuncs.com/tradingai/tbase:gpu-latest`
- train model(docker hub/local):
    ```
    docker run --runtime=nvidia -it \
        -e TUSHARE_TOKEN=$TUSHARE_TOKEN \
        -v $PWD:/root/trade/tbase \
        tradingai/tbase:gpu-latest bash

    python -m tbase.run --alg ddpg --codes 000001.SZ --seed 0
    ```
- train model(阿里云镜像):
    ```
    docker run --runtime=nvidia -it \
        -e TUSHARE_TOKEN=$TUSHARE_TOKEN \
        -v $PWD:/root/trade/tbase \
        docker pull registry.cn-hangzhou.aliyuncs.com/tradingai/tbase:gpu-latest bash

    python -m tbase.run --alg ddpg --codes 000001.SZ --seed 0
    ```
