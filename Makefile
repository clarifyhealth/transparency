.PHONY: build
build:
	python setup.py sdist

.PHONY: publish
publish:
	twine upload dist/*