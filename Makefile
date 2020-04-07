.PHONY: install update test dist upload build_cpu build_gpu lint

install:
	pip3 install -e . --user

update:
	git pull origin master
	pip3 install -e . --user

test:
	docker-compose up

upload:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	rm -rf build
	rm -rf dist

build_cpu:
	# 可以先在环境变量中设置 TUSHARE_TOKEN
	docker build --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN --build-arg BUILD_TIMD=$(date +%s) . -t tradingai/tbase:latest
	# 重新 build
	# docker build --no-cache --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN . -t tradingai/tbase:latest

build_gpu:
	# 可以先在环境变量中设置 TUSHARE_TOKEN
	docker build -f gpu.Dockerfile --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN --build-arg BUILD_TIMD=$(date +%s) . -t tradingai/tbase:gpu-latest
	# 重新 build
	# docker build --no-cache --build-arg TUSHARE_TOKEN=$TUSHARE_TOKEN . -t tradingai/tbase:gpu-latest

lint:
	buildifier WORKSPACE
	find ./ -name 'BUILD' | xargs buildifier
