#!/usr/bin/env python
from setuptools import setup

project_name = "simphox"

setup(
    name=project_name,
    version="0.0.1alpha",
    packages=[project_name],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'jaxlib',
        'jax',
        'scikit-image',
        'pydantic'
    ],
    extras_require={
        'interactive': ['matplotlib',
                        'jupyterlab',
                        'holoviews',
                        'bokeh']
    }
)
