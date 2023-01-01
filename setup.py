#!/usr/bin/env python
from setuptools import setup, find_packages

project_name = "simphox"

setup(
    name=project_name,
    version="0.0.1a8",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'jaxlib',
        'jax',
        'scikit-image',
        'pydantic',
        'xarray',
        'dm-haiku',
        'absl-py'
    ],
    extras_require={
        'interactive': ['matplotlib',
                        'jupyterlab',
                        'holoviews',
                        'bokeh']
    }
)
