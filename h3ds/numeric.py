import copy

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from h3ds.log import logger
from h3ds.mesh import Mesh
from h3ds.affine_transform import AffineTransform


def load_K_Rt(P: np.ndarray):

    dec = cv2.decomposeProjectionMatrix(P)
    K = dec[0]
    R = dec[1]
    t = dec[2]

    intrinsics = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def perform_alignment(mesh_source: Mesh,
                      mesh_target: Mesh,
                      landmarks_source: dict = None,
                      landmarks_target: dict = None) -> tuple:

    if landmarks_source is None or landmarks_target is None:
        logger.warning(
            'Alignment skipped: Landmarks source and/or target were not provided.'
        )
        return mesh_source, np.eye(4)

    l_ids = landmarks_target.keys()
    points_s = mesh_source.vertices[[landmarks_source[l] for l in l_ids]]
    points_t = mesh_target.vertices[[landmarks_target[l] for l in l_ids]]

    transform, _, _ = trimesh.registration.procrustes(points_s,
                                                      points_t,
                                                      reflection=False)

    return transform_mesh(mesh_source, transform), transform


def perform_icp(mesh_source: Mesh,
                mesh_target: Mesh,
                mask_source: np.ndarray = None,
                mask_target: np.ndarray = None,
                **icp_args) -> tuple:

    points_source = mesh_source.vertices if mask_source is None else mesh_source.vertices[
        mask_source]
    points_target = mesh_target.vertices if mask_target is None else mesh_source.vertices[
        mask_target]
    transform, _, _ = trimesh.registration.icp(points_source, points_target,
                                               **icp_args)

    return transform_mesh(mesh_source, transform), transform


def transform_mesh(mesh: Mesh, transform: np.ndarray):
    mesh_t = mesh.copy()
    mesh_t.vertices = AffineTransform(matrix=transform).transform(
        mesh_t.vertices)
    return mesh_t


def unidirectional_chamfer_distance(source: np.ndarray, target: np.ndarray):

    kdtree = cKDTree(target, leafsize=10)
    d, _ = kdtree.query(source, k=1)

    return d
