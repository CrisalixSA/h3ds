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

## Accessing H3DS data

You can start using H3DS in your project with a few lines of code

```python
from h3ds.dataset import H3DS

h3ds = H3DS(path='local/path/to/h3ds')
h3ds.download(token=H3DS_ACCESS_TOKEN)
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8')
```

To list the available scenes, simply use:
```python
scenes = h3ds.scenes() # returns all the scenes ['1b2a8613401e42a8', ...]
scenes = h3ds.scenes(tags={'h3d-net'}) # returns the scenes used in H3D-Net paper
```

In order to reproduce the results from H3D-Net, we provide default views configurations for each scene:
```python
views_configs = h3ds.default_views_configs(scene_id='1b2a8613401e42a8') # '3', '4', '8', '16' and '32'
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8', views_config_id='3')
```
This will load a scene with a mesh, 3 images, 3 masks and 3 cameras.

## Evaluation

We provide methods for evaluating your reconstructions with a single line of code

```python
mesh_pred, landmarks_pred = my_rec_method(images, masks, cameras)
chamfer, _, _, _ = h3ds.evaluate_scene('1b2a8613401e42a8', mesh_pred, landmarks_pred)
```

The `landmarks_pred` is an optional dictionary containing landmarks used for a coarse alignment between the predicted mesh and the ground truth mesh. Please, check [this description](images/landmarks.png) of the landmarks positions.

For more insights, check the examples provided.

## Terms of use
By using the H3DS Dataset you agree with the following terms:

1. The data must be used for non-commercial research and/or education purposes only.
2. You agree not to copy, sell, trade, or exploit the data for any commercial purposes.
3. If you will be publishing any work using this dataset, please cite the original paper.

## Citation
```
@article{ramon2021h3d,
  title={H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction},
  author={Ramon, Eduard and Triginer, Gil and Escur, Janna and Pumarola, Albert and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2107.12512},
  year={2021}
}
```