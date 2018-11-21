# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='hybridvec',
      version='0.2.0',
      description='HybridVec project code',
      author='Andrey Kurenkov',
      package_dir = {'': 'src'},
      packages=['hybridvec'],
      install_requires=requirements,
      test_suite='test'
     )
