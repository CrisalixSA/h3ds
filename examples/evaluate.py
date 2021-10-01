import os
import argparse
import json
from random import choice

import trimesh
import numpy as np

from h3ds.dataset import H3DS
from h3ds.numeric import unidirectional_chamfer_distance
from h3ds.log import logger
from h3ds.utils import error_to_color


def your_reconstruction_method(images: list, masks: list, cameras: list):
    ''' Returns another H3DS scene randomly to simulate a reconstruction'''
    scene_id = choice(h3ds_scenes)
    logger.info(
        f' > Randomly selected scene {scene_id} to emulate current reconstruction'
    )
    return h3ds.load_mesh(scene_id), h3ds.load_landmarks(scene_id)


def main(h3ds_path, h3ds_token, output_dir):

    global h3ds
    global h3ds_scenes

    # Create instance of h3ds and download it if not available
    h3ds = H3DS(path=h3ds_path)
    h3ds.download(token=h3ds_token)

    # Reconstruct all the scenes used in the h3d-net paper and store the metric
    metrics = {}
    h3ds_scenes = h3ds.scenes(tags={'h3d-net'})
    for scene_id in h3ds_scenes:

        logger.info(f'Evaluating H3D-Net scene {scene_id}.')

        # Get scene in millimiters
        mesh_gt, images, masks, cameras = h3ds.load_scene(scene_id,
                                                          normalized=False)

        # Perform 3D reconstruction with your algorithm.
        mesh_pred, landmarks_pred = your_reconstruction_method(
            images, masks, cameras)

        # Evaluate scene. The `landmarks_pred` are optional and, if provided, they will be used
        # for an initial alignment in the evaluation process. If not provided, it will be assumed
        # that the predicted mesh is already coarsely aligned with the ground truth mesh.
        chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred_aligned = \
            h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred)

        metrics[scene_id] = np.mean(chamfer_gt_pred)
        logger.info(f' > Chamfer Distance (mm): {metrics[scene_id]}')

        # We can easily colorize a mesh. Clipping set at 5 mm
        mesh_gt.save(os.path.join(output_dir, f'{scene_id}_gt.obj'))
        mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
                                                          clipping_error=5)
        mesh_pred_aligned.save(os.path.join(output_dir, f'{scene_id}_pred.obj'))

    # Show average results
    logger.info(
        f'Average Chamfer Distance (mm): {np.mean(list(metrics.values()))}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Exemplifies how to evaluate a method')
    parser.add_argument('--h3ds-path', help='H3DS dataset path', required=True)
    parser.add_argument('--h3ds-token', help='H3DS access token', required=True)
    parser.add_argument('--output-dir',
                        help='Output directory to store the results',
                        required=True)

    args = parser.parse_args()
    main(h3ds_path=args.h3ds_path,
         h3ds_token=args.h3ds_token,
         output_dir=args.output_dir)
