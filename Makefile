.PHONY: install update test upload lint

install:
	pip3 install -e . --user

update:
	git pull origin master
	pip3 install -e . --user

test:
	python3 -m pytest --cov-report=xml:docs/cov/report.xml --cov=tbase
	coverage report -m

upload:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	rm -rf build
	rm -rf dist

lint:
	flake8 .
