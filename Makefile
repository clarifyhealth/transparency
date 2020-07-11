.PHONY: package
package:
	python setup.py sdist

.PHONY: publish
publish:
	twine upload dist/*

.PHONY: testpublish
testpublish:
	twine upload --repository testpypi dist/*