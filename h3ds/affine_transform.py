import json
import numpy as np


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

    def save(self, filename: str):
        data = {"dimension": str(self.dim), 'transform': []}
        for r in range(self.dim + 1):
            for c in range(self.dim + 1):
                data['transform'].append({
                    'row': str(r),
                    'column': str(c),
                    'element': str(self.matrix[r, c])
                })

        with open(filename, 'w') as f:
            json.dump(data, f)

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