# H3DS Dataset
This repository contains some utilities for using the H3DS dataset introduced in [H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction](https://arxiv.org/abs/2107.12512v1)

## Access
The H3DS dataset is available only for academic purposes. To request access, start by filling the [contact form](https://forms.gle/AH1hKXRdshWyk9e46) with your academic email and a license agreement will be sent back, which must be completed by a full-time academic staff member. Finally, you'll be provided with a `H3DS_ACCESS_TOKEN` to use the dataset.

## Setup
The simplest way to use the H3DS dataset is by installing it as a pip package:
```bash
pip install h3ds
```

In order to use the dataset, define the following envars with the `H3DS_ACCESS_TOKEN` provided and a (optional) local path:
```bash
export H3DS_ACCESS_TOKEN=token_provided
export H3DS_PATH=path/for/h3ds
```

Finally, simply import the package in your python scripts
```python
from h3ds import H3DS

h3ds = H3DS()
```

## Using H3DS
First of all, pull the data (less than 500 Mb)
```
h3ds.pull()
```

List the ids of the available scenes
```
scenes = h3ds.scenes()
```

Get data from a scene
```
mesh, images, cameras = h3ds.get_scene(scene_id=scenes[0])
```
