.PHONY: clean
clean:
	rm -fr ./build
	rm -fr ./dist
	rm -fr *.pyc
	rm -fr ./*.egg-info

.PHONY: package
package: clean
	python setup.py sdist

.PHONY: publish
publish:
	twine upload dist/*

.PHONY: testpublish
testpublish:
	twine upload --repository testpypi dist/*