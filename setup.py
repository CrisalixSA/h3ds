import os
import re
from tempfile import mkstemp
from setuptools import setup, find_packages
from pkg_resources import parse_requirements

PACKAGE_NAME = 'h3ds'
PACKAGE_PATH = PACKAGE_NAME
VERSION_FILE_PATH = os.path.join(PACKAGE_PATH, '__init__.py')
REQUIREMENTS_FILE_PATH = 'requirements.txt'

# Read __version__ from VERSION_FILE_PATH
exec(open(VERSION_FILE_PATH).read())

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

# Build setup
setup(
    name=PACKAGE_NAME,
    version=__version__,
    author='Crisalix SA',
    author_email='eduard.ramon@crisalix.com',
    description='Python interface for H3DS dataset',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://test.pypi.org/project/h3ds/',
    project_urls={
        'Bug Tracker': 'https://github.com/CrisalixSA/h3ds/issues',
        'Project Website': 'https://crisalixsa.github.io/h3d-net/'
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={PACKAGE_NAME: PACKAGE_PATH},
    packages=find_packages(
        include=[PACKAGE_NAME],
        exclude=[],
    ),
    install_requires=[
        'toml',
        'requests',
        'trimesh',
        'Pillow',
        'tqdm',
        'opencv-python',
        'scipy'
    ],
    package_data={
        '': ['config.toml']
    }
)
