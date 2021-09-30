import os
import argparse
import random

import numpy as np
from PIL import Image

from h3ds.dataset import H3DS


def project_scene(mesh, img, cam, color=[255, 0, 0]):

    # Expand cam attributes
    K, P = cam
    P_inv = np.linalg.inv(P)

    # Project mesh vertices into 2D
    p3d_h = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]

    # Draw p2d to image
    img_proj = np.array(img)
    p2d = np.clip(p2d, 0, img.width - 1).astype(np.uint32)
    img_proj[p2d[:, 1], p2d[:, 0]] = color

    return Image.fromarray(img_proj.astype(np.uint8))


def main(h3ds_path, h3ds_token, output_dir):

    # Create instance of h3ds and download it if not available
    h3ds = H3DS(path=h3ds_path)
    h3ds.download(token=h3ds_token)

    # Get a random scene data
    scene_id = random.choice(h3ds.scenes())
    mesh, images, masks, cameras = h3ds.load_scene(scene_id)

    # Project the mesh on each image
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img, cam) in enumerate(zip(images, cameras)):
        img_proj = project_scene(mesh, img, cam)
        img_proj.save(os.path.join(output_dir, f'{idx}.jpg'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reprojecting a scene')
    parser.add_argument('--h3ds-path', help='H3DS dataset path', required=True)
    parser.add_argument('--h3ds-token', help='H3DS access token', required=True)
    parser.add_argument('--output-dir',
                        help='Output directory to store the results',
                        required=True)

    args = parser.parse_args()
    main(h3ds_path=args.h3ds_path,
         h3ds_token=args.h3ds_token,
         output_dir=args.output_dir)
