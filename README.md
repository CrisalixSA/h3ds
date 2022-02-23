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

## Download the dataset
You can download the dataset from [this link](https://drive.google.com/file/d/1is1AByaMwaWJJN6CwQ4MmeqCHIMiijZw/view?usp=sharing).
Then, unzip the file using the `H3DS_ACCESS_TOKEN` as password:
```bash
unzip -P H3DS_ACCESS_TOKEN local/path/to/h3ds.zip -d local/path/to/h3ds
```

## Accessing H3DS data

You can start using H3DS in your project with a few lines of code

```python
from h3ds.dataset import H3DS

h3ds = H3DS(path='local/path/to/h3ds')
h3ds.download(token=H3DS_ACCESS_TOKEN) # This is currenly not working so please downlad the data manually
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

We provide a method for evaluating your reconstructions with a single line of code

```python
mesh_pred, landmarks_pred = my_rec_method(images, masks, cameras)
chamfer, _, _, _ = h3ds.evaluate_scene('1b2a8613401e42a8', mesh_pred, landmarks_pred)
```

The `landmarks_pred` is an optional dictionary containing landmarks used for a coarse alignment between the predicted mesh and the ground truth mesh. Please, check [this description](images/landmarks.png) of the landmarks positions.

For more insights, check the examples provided.

## Comparison against H3D-Net

The results reported in the H3D-Net paper (Table 2) slightly differ from the ones obtained using the evaluation code provided in this repository. This is due to minor implementation changes in the alignment process and in the cutting of the regions. In the following table we provide the results obtained using the evaluation code from this repository. We encourage everyone to use the `evaluate_scene` method provided in this repository to report comparable results accross different works.

| Method \ Views | 3 | 4 | 8 | 16 | 32 |
|:-:|:-:|---|---|---|---|
| IDR | 2.79 / 14.58 | 1.88 / 8.99 | 1.83 / 8.34 | 1.31 / 6.37 | 1.25 / 5.71 |
| H3D-Net | 1.33 / 10.52 | 1.35 / 7.71 | 1.18 / 6.45 | 1.05 / 5.36 | 1.03 / 5.25 |

The numbers from the table can be obtained by running the [evaluation script](examples/evaluate.py) from the examples folder, which uses [the 3D reconstructions from the paper](https://drive.google.com/drive/folders/1urlKA-g4oQgqgcBkv9cUjVyV46oJytN_?usp=sharing) to compute the metrics.

## Terms of use
The H3DS dataset is available for non-commercial or research use only. By using or downloading the files included in the dataset you agree to the terms of this [license agreement](https://drive.google.com/file/d/149t4_BF37eYljI6E2oCjezNaK7KPWvLp/view?usp=sharing) and undertake to comply with its terms. If you do not agree with these terms, you may not accesss, download or use the files.

## Citation
```
@inproceedings{ramon2021h3d,
  title={H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction},
  author={Ramon, Eduard and Triginer, Gil and Escur, Janna and Pumarola, Albert and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5620--5629},
  year={2021}
}
```
