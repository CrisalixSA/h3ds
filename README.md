# H3DS Dataset

[![PyPI](https://img.shields.io/pypi/v/h3ds?style=flat-square)](https://pypi.org/project/h3ds/)

This repository contains some utilities for using the H3DS dataset introduced in [H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction](https://arxiv.org/abs/2107.12512v1)

## Access
The H3DS dataset is available only for academic purposes. To request access, start by filling the [contact form](https://forms.gle/AH1hKXRdshWyk9e46) with your academic email and a license agreement will be sent back, which must be completed by a full-time academic staff member. Finally, you'll be provided with a `H3DS_ACCESS_TOKEN` to use the dataset.

## Setup
The simplest way to use the H3DS dataset is by installing it as a pip package:
```bash
pip install h3ds
```

## Using H3DS
In order to use the dataset, simply import the package in your python scripts.
```python
from h3ds.dataset import H3DS

h3ds = H3DS(path='local/path/to/h3ds')
```

If it's the first time using H3DS, download the data (less than 500 Mb).
```python
h3ds.download(token=H3DS_ACCESS_TOKEN)
```

You can easily list the ids of the available scenes
```python
scenes = h3ds.scenes()
```

and load the data from a concrete scene as:
```python
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8')
```
