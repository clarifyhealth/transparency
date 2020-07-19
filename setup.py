from os import path, getcwd
from setuptools import setup, find_packages

this_directory = getcwd()

package_name = 'transparency'

try:
    with open(path.join(this_directory, 'VERSION'), encoding='utf-8') as version_file:
        version = version_file.read().strip()
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as readme_file:
        long_description = readme_file.read()
except IOError:
    raise


def parse_requirements(file):
    with open(file, "r") as fs:
        return [r for r in fs.read().splitlines() if
                (len(r.strip()) > 0 and not r.strip().startswith("#") and not r.strip().startswith("--"))]


requirements = parse_requirements('requirements.txt')
test_requirements = parse_requirements('requirements-test.txt')

setup(name=package_name,
      version=version,
      license='Apache License 2.0',
      description="The library for GLM and Ensemble Tree model explanation",
      long_description_content_type='text/markdown',
      long_description=long_description,
      author='Iman Haji, Alvin Henrick',
      author_email='iman.bio@gmail.com, share.code@aol.com',
      url='https://github.com/imanbio/transparency',
      packages=find_packages(exclude=['tests']),
      setup_requires=[
          'setuptools>=49.1.0',
          'wheel>=0.34.2',
          'twine>=3.2.0'
      ],
      install_requires=requirements,
      tests_require=test_requirements,
      include_package_data=True,
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      zip_safe=False)
