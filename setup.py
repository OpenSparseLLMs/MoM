import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup  

setup(
    name='mom',
    version='0.1.0',
    description='MoM: Mixture of Memories',
    author='Jusen Du',
    author_email='dujusen@gmail.com',
    url='https://github.com/OpenSparseLLMs/MoM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.3',
        'transformers>=4.45.0',
        'triton>=3.0',
        'datasets>=3.1.0',
        'einops',
        'ninja',
        'fla @ git+https://github.com/fla-org/flash-linear-attention'
    ],
)
