import os
import argparse
import json
from random import choice

import trimesh
import numpy as np

from h3ds.dataset import H3DS
from h3ds.numeric import unidirectional_chamfer_distance


def your_reconstruction_method(images: list, masks: list, cameras: list):
    return choice([
        trimesh.primitives.Box(radius=100),
        trimesh.primitives.Sphere(radius=100),
        trimesh.primitives.Cylinder(radius=100)
    ])


def main(h3ds_path):

    # Create instance of h3ds and download it if not available
    h3ds = H3DS(path=h3ds_path)
    h3ds.download(token=os.getenv('H3DS_ACCESS_TOKEN'))

    # Reconstruct all the scenes and store the metric
    metrics = {}
    for scene_id in h3ds.scenes():

        # Get data in millimiters
        mesh_gt, images, masks, cameras = h3ds.load_scene(scene_id,
                                                          normalized=False)
        mesh_pred = your_reconstruction_method(images, masks, cameras)

        # Compute metrics
        metrics[scene_id] = np.mean(
            unidirectional_chamfer_distance(source=mesh_pred.vertices,
                                            target=mesh_gt.vertices))

    # Show results
    print(
        f'Chamfer Distance (mm):\n{json.dumps(metrics, indent=2, sort_keys=True)}'
    )
    print(f'Average Chamfer Distance: {np.mean(list(metrics.values()))} mm')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--h3ds-path', help='H3DS dataset path', required=True)

    args = parser.parse_args()
    main(h3ds_path=args.h3ds_path)
