import os
import argparse
import json
from random import choice

import numpy as np

from h3ds.dataset import H3DS
from h3ds.log import logger


def main(h3ds_path, h3ds_token):

    h3ds = H3DS(path=h3ds_path)
    h3ds.download(token=h3ds_token)
    scene_id = choice(h3ds.scenes())

    # A scene can be loaded in its real scale (mm)
    mesh_mm, images, masks, cameras_mm = h3ds.load_scene(scene_id)

    # Or normalized within a unit sphere
    mesh_unit, images, masks, cameras_unit = h3ds.load_scene(scene_id,
                                                             normalized=True)

    # The transformation matrix from the real scale to the normalized one is also provided
    transform = h3ds.load_normalization_matrix(scene_id)

    # It can be checked that both meshes are equivalent
    vertices_mm_hom = np.hstack(
        [mesh_mm.vertices,
         np.ones((mesh_mm.vertices.shape[0], 1))])
    vertices_mm_to_unit = (transform @ vertices_mm_hom.T).T[:, :3]

    assert np.allclose(vertices_mm_to_unit, mesh_unit.vertices)
    logger.info('Transformed meshes are equivalent')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Illustrates the H3DS transformations')
    parser.add_argument('--h3ds-path', help='H3DS dataset path', required=True)
    parser.add_argument('--h3ds-token', help='H3DS access token', required=True)

    args = parser.parse_args()
    main(h3ds_path=args.h3ds_path, h3ds_token=args.h3ds_token)
