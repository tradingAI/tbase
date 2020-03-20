# 需要先在环境变量中设置 TUSHARE_TOKEN
docker build --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN --build-arg BUILD_TIMD=$(date +%s) . -t tradingai/tbase:latest
# 重新 build
# docker build --no-cache --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN . -t tradingai/tbase:latest
