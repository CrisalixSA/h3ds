# H3DS Dataset

[![PyPI](https://img.shields.io/pypi/v/h3ds?style=flat-square)](https://pypi.org/project/h3ds/)

This repository contains the code for using the H3DS dataset introduced in [H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction](https://arxiv.org/abs/2107.12512v1)

## Access
The H3DS dataset is only available for non-commercial research purposes. To request access, please fill in the [contact form](https://forms.gle/AH1hKXRdshWyk9e46) with your academic email. Your application will be reviewed and, after acceptance, you will recieve a `H3DS_ACCESS_TOKEN` together with the license and terms of use.

## Setup
The simplest way to use the H3DS dataset is by installing it as a pip package:
```bash
pip install h3ds
```

## Using H3DS
You can start using H3DS in your project with a few lines of code
```python
from h3ds.dataset import H3DS

h3ds = H3DS(path='local/path/to/h3ds')
h3ds.download(token=H3DS_ACCESS_TOKEN)
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8')
```
The returned types when loading a scene are `Trimesh`, `list(PIL.Image)` `list(PIL.Image)` and `list(tuple(np.ndarray))`.

To list the available scenes, simply use:
```python
scenes = h3ds.scenes() # ['1b2a8613401e42a8', ...]
```

In order to reproduce the results from H3D-Net, we provide default views configurations for each scene:
```python
views_configs = h3ds.default_views_configs(scene_id='1b2a8613401e42a8') # '3', '4', '8', '16' and '32'
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8', views_config_id='3')
```
This will load a scene with 3 images, 3 masks and 3 cameras.

Please, see the provided examples for more insights.

## Citation
If you find this work useful, please use the following to cite our paper:
```
@article{ramon2021h3d,
  title={H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction},
  author={Ramon, Eduard and Triginer, Gil and Escur, Janna and Pumarola, Albert and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2107.12512},
  year={2021}
}
```