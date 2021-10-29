#!/usr/bin/env python
from setuptools import setup

project_name = "simphox"

setup(
    name=project_name,
    version="0.0.1a1",
    packages=[project_name],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'jaxlib',
        'jax',
        'scikit-image',
        'pydantic',
        'xarray'
    ],
    extras_require={
        'interactive': ['matplotlib',
                        'jupyterlab',
                        'holoviews',
                        'bokeh']
    }
)
