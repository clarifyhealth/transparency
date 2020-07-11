.PHONY: package
package:
	python setup.py sdist

.PHONY: publish
publish:
	twine upload dist/*