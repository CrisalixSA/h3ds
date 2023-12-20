# H3DS Dataset

[![PyPI](https://img.shields.io/pypi/v/h3ds?style=flat-square)](https://pypi.org/project/h3ds/)

This repository contains the code for using the H3DS dataset introduced in [H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction (ICCV 2021)](https://crisalixsa.github.io/h3d-net/) and expanded in [SIRA++: Implicit Shape and Appearance Priors for Few-Shot
Full Head Reconstruction](https://arxiv.org/abs/2310.08784).

## Access
The H3DS dataset is only available for non-commercial research purposes. To request access, please fill in the [contact form](https://docs.google.com/forms/d/e/1FAIpQLScpgNf0AA-2BuqcjDod-StNsolYm3DVLtLEdgROiX49xC83dQ/viewform) with your academic email. Your application will be reviewed and, after acceptance, you will recieve a `H3DS_ACCESS_TOKEN`. Each version of the dataset has a different acces token.

## Setup
The simplest way to use the H3DS dataset is by installing it as a pip package:
```bash
pip install h3ds
```

## Accessing H3DS data

The dataset has grown over a series of releases. Version 1.0 (`config_v1.toml`), introduced in H3D-Net, included 23 subjects. Version 2.0 (`config_v2.toml`), introduced in SIRA++, includes 59 subjects. Although version 2.0 contains all the subjects of version 1.0, we have decided to maintain both as the mesh alignment could be slightly different.

You can start using H3DS in your project with a few lines of code

```python
from h3ds.dataset import H3DS

h3ds = H3DS(path='local/path/to/h3ds', config_id='config_v2')
h3ds.download(token=H3DS_ACCESS_TOKEN) # Python zip decryption is very slow. You can invoke an external program.
mesh, images, masks, cameras = h3ds.load_scene(scene_id='1b2a8613401e42a8')
```

It is also possible to download and unzip the dataset manually ([v1 link](https://drive.google.com/file/d/1is1AByaMwaWJJN6CwQ4MmeqCHIMiijZw/view?usp=sharing), [v2 link](https://drive.google.com/file/d/12zMORAGo6IArLS0wras2G_K-1yRak4pA/view?usp=sharing)). Then, unzip the file using the `H3DS_ACCESS_TOKEN` as password.

```bash
unzip -P H3DS_ACCESS_TOKEN local/path/to/h3ds.zip -d local/path/to/h3ds
```

To list the available scenes, simply use:
```python
scenes = h3ds.scenes() # returns all the scenes ['1b2a8613401e42a8', ...]
scenes = h3ds.scenes(tags={'h3d-net'}) # returns the scenes used in H3D-Net paper
scenes = h3ds.scenes(tags={'sira++'}) # returns the scenes used in SIRA++ paper
```

In order to reproduce the results from H3D-Net and SIRA++, we provide default views configurations for each scene:
```python
views_configs = h3ds.default_views_configs(scene_id='1b2a8613401e42a8') # '1', '3', '4', '6', '8', '16' and '32' views
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

## Comparison against H3D-Net and SIRA++

The results reported in the H3D-Net and SIRA++ papers slightly differ from the ones obtained using the evaluation code provided in this repository. This is due to minor implementation changes in the alignment process and in the cutting of the regions. In the following table we provide the results obtained using the evaluation code from this repository. We encourage everyone to use the `evaluate_scene` method provided in this repository to report comparable results accross different works.

Evaluation in H3DS version 1.0 (config_v1):

| Method \ Views | 3 | 4 | 8 | 16 | 32 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| IDR | 2.79 / 14.58 | 1.88 / 8.99 | 1.83 / 8.34 | 1.31 / 6.37 | 1.25 / 5.71 |
| H3D-Net | 1.33 / 10.52 | 1.35 / 7.71 | 1.18 / 6.45 | 1.05 / 5.36 | 1.03 / 5.25 |
| SIRA++ | 1.13 / 10.84 | 1.18 / 8.07 | 1.15 / 7.08 | 1.00 / 6.34 | 0.99 / 5.79 |

Evaluation in H3DS version 2.0 (config_v2):

| Method \ Views | 1 | 3 | 4 | 6 | 8 | 16 | 32 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SIRA++ | 1.56 / 13.66 | 1.18 / 9.70 | 1.21 / 6.09 | 1.05 / 5.59 | 1.18 / 5.28 | 1.08 / 4.96 | 1.03 / 4.71 | 

The numbers from the table can be obtained by running the [evaluation script](examples/evaluate.py) from the examples folder, which uses [the 3D reconstructions from the paper](https://drive.google.com/drive/folders/1urlKA-g4oQgqgcBkv9cUjVyV46oJytN_?usp=sharing) to compute the metrics.

## Terms of use
The H3DS dataset is available for non-commercial or research use only. By using or downloading the files included in the dataset you agree to the terms of this [license agreement](https://drive.google.com/file/d/1_Uy5jklFEQMGvw0W-wTJvdwRqOtSAf7l/view?usp=sharing) and accept to comply with its terms. If you do not agree with these terms, you may not accesss, download or use the files.

## Data acquisition
The data acquisition process for each scene in the dataset involves several steps. Initially, the camera of an iPad Pro is calibrated to be aligned to an attached structured light sensor, specifically the Occipital Structure Sensor Pro. This calibration process allows us to obtain paired RGB images and camera parameters, along with a low-resolution mesh scan. Simultaneously, a high-end Artec Eva scanner is utilized to capture high-quality 3D scans. Subsequently, we align the low and high-resolution meshes, along with the paired cameras, by employing six manually annotated 3D landmarks and utilizing the iterative closest point (ICP) refinement technique. Furthermore, for each image within the dataset, we manually annotate a foreground mask, providing additional information for foreground-background separation.  

As of data acquisition in v2, the subjects remains static for images at +-90° degrees from the front. This yields a consistent lighting and background, as well as better cameras. This cannot be assumed for subjects from v1.


## Projects using H3DS
Here are some works that we have noticed using the dataset. Please reach out if we missed you!

[Neural Haircut: Prior-Guided Strand-Based Hair Reconstruction (ICCV 2023)](https://samsunglabs.github.io/NeuralHaircut/)  
*Sklyarova, Vanessa and Chelishev, Jenya and Dogaru, Andreea and Medvedev, Igor and Lempitsky, Victor and Zakharov, Egor*

[NeRF-Art: Text-Driven Neural Radiance Fields Stylization (TVCG2023)](https://cassiepython.github.io/nerfart/)  
*Wang, Can and Jiang, Ruixiang and Chai, Menglei and He, Mingming and Chen, Dongdong and Liao, Jing*

[Real-Time Radiance Fields for Single-Image Portrait View Synthesis (SIGGRAPH2023)](https://research.nvidia.com/labs/nxp/lp3d/)  
*Alex Trevithick and Matthew Chan and Michael Stengel and Eric R. Chan and Chao Liu and Zhiding Yu and Sameh Khamis and Manmohan Chandraker and Ravi Ramamoorthi and Koki Nagano*

[Multi-NeuS: 3D Head Portraits from Single Image with Neural Implicit Functions (2023)](https://ieeexplore.ieee.org/abstract/document/10233007)  
*Burkov, Egor and Rakhimov, Ruslan and Safin, Aleksandr and Burnaev, Evgeny and Lempitsky, Victor*

[SIRA: Relightable Avatars From a Single Image (WACV2023)](https://openaccess.thecvf.com/content/WACV2023/html/Caselles_SIRA_Relightable_Avatars_From_a_Single_Image_WACV_2023_paper.html)  
*Caselles, Pol and Ramon, Eduard and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc and Triginer, Gil*

[NPBG++: Accelerating Neural Point-Based Graphics (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Rakhimov_NPBG_Accelerating_Neural_Point-Based_Graphics_CVPR_2022_paper.html)  
*Rakhimov, Ruslan and Ardelean, Andrei-Timotei and Lempitsky, Victor and Burnaev, Evgeny*

[Learning an Isometric Surface Parameterization for Texture Unwrapping (ECCV 2022)](https://sagniklp.github.io/isouvf/)  
*Das, Sagnik and Ma, Ke and Shu, Zhixin and Samaras, Dimitris*

[VoRF: Volumetric Relightable Faces (BMVC 2022)](https://vcai.mpi-inf.mpg.de/projects/VoRF/)  
*Rao, Pramod and {B R}, Mallikarjun and Fox, Gereon and Weyrich, Tim and Bickel, Bernd and Seidel, Hans-Peter and Pfister, Hanspeter and Matusik, Wojciech and Tewari, Ayush and Theobalt, Christian and  Elgharib, Mohamed*

## Contact Info

Pol Caselles: pol.caselles@crisalix.com

## Citation
If you find this project helpful to your research, please consider citing the following:
```
@inproceedings{ramon2021h3d,
  title={H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction},
  author={Ramon, Eduard and Triginer, Gil and Escur, Janna and Pumarola, Albert and Garcia, Jaime and Giro-i-Nieto, Xavier and Moreno-Noguer, Francesc},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5620--5629},
  year={2021}
}
```
```
@article{caselles2023implicit,
  title={Implicit Shape and Appearance Priors for Few-Shot Full Head Reconstruction},
  author={Caselles, Pol and Ramon, Eduard and Garcia, Jaime and Triginer, Gil and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2310.08784},
  year={2023}
}
```
