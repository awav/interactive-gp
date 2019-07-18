# -*- coding: utf-8 -*-
# pylint: skip-file

from setuptools import find_packages, setup

packages = find_packages('.', exclude=["notebooks", "apps"])
print(packages)

setup(name='interactive_gp',
      version="0.0.1",
      author="Artem Artemev",
      author_email="art.art.v@gmail.com",
      description="Interactive Gaussian Processes",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/awav/interactive-gp",
      packages=packages,
      include_package_data=True,
      python_requires=">=3.6")
