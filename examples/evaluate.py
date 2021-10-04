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
    metrics_head = {}
    metrics_face = {}
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

        metrics_head[scene_id] = np.mean(chamfer_gt_pred)
        logger.info(
            f' > Chamfer distance full head (mm): {metrics_head[scene_id]}')
        mesh_gt.save(os.path.join(output_dir, 'full_head', f'{scene_id}_gt.obj'))

        # The chamfer computed from prediction to ground truth is only provided for
        # visualization purporses (i.e. heatmaps).
        mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
                                                          clipping_error=5)
        mesh_pred_aligned.save(os.path.join(output_dir, 'full_head', f'{scene_id}_pred.obj'))

        # Evaluate reconstruction in the facial region, defined by a sphere of radius 95mm centered
        # in the tip of the nose. In this case, a more fine alignment is performed, taking into account
        # only the vertices from this region. This evaluation should be used when assessing methods
        # that only reconstruct the frontal face area (i.e. Basel Face Bodel)
        chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred_aligned = \
            h3ds.evaluate_scene(scene_id, mesh_pred, landmarks_pred, region_id='face_sphere')

        # Note that in both cases we only report the chamfer distane computed from the ground truth
        # to the prediction, since here we have control over the region where the metric is computed.
        metrics_face[scene_id] = np.mean(chamfer_gt_pred)
        logger.info(f' > Chamfer distance face (mm): {metrics_face[scene_id]}')
        mesh_gt.save(os.path.join(output_dir, 'face_sphere', f'{scene_id}_gt.obj'))

        # Again, the chamfer computed from prediction to ground truth is only provided for
        # visualization purporses (i.e. heatmaps).
        mesh_pred_aligned.vertices_color = error_to_color(chamfer_pred_gt,
                                                          clipping_error=5)

        # For improved visualization the predicted mesh is cut to be inside the unit sphere of 95mm.
        v_pred = mesh_pred.vertices
        mask_sphere = np.where(np.linalg.norm(v_pred - v_pred[landmarks_pred['nose_tip']], axis=-1) < 95)
        mesh_pred_aligned = mesh_pred_aligned.cut(mask_sphere)

        mesh_pred_aligned.save(
            os.path.join(output_dir, 'face_sphere', f'{scene_id}_pred.obj'))

    # Show average results
    logger.info(
        f'Average Chamfer Distance full head (mm): {np.mean(list(metrics_head.values()))}'
    )
    logger.info(
        f'Average Chamfer Distance face (mm): {np.mean(list(metrics_face.values()))}'
    )


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
