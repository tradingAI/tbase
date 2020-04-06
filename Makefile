.PHONY: install update test dist upload clean

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
	bash build-docker-image.sh

build_gpu:
	bash build-docker-image-gpu.sh
