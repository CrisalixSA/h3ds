import json

import cv2
import numpy as np
from scipy.spatial import cKDTree


def unidirectional_chamfer_distance(source: np.ndarray, target: np.ndarray):

    kdtree = cKDTree(target, leafsize=10)
    d, _ = kdtree.query(source, k=1)

    return d


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


class AffineTransform:

    def __init__(self, dim: int = 3, matrix: np.ndarray = np.eye(4)):
        self.dim = dim
        self.matrix = np.copy(matrix)

    def load(self, filename: str):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)

        self.dim = int(data['dimension'])
        self.matrix = np.eye(self.dim + 1)
        for el in data['transform']:
            self.matrix[int(el['row']),
                        int(el['column'])] = float(el['element'])
        return self

    def transform(self, points: np.ndarray):
        # Assuming ( n_points, dim)
        n_points, dim = points.shape
        assert dim == self.dim

        # Project points
        points_h = np.concatenate((points.T, np.ones((1, n_points))), axis=0)
        return np.dot(self.matrix, points_h)[:self.dim, :].T

    def inverse(self):
        t = AffineTransform(dim=self.dim)
        t.matrix = np.linalg.inv(self.matrix)
        return t
