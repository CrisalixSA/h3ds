import os
import sys
import re
import copy
import numpy as np
import scipy
import trimesh

from h3ds.utils import get_file_extension, create_parent_directory


class Mesh:

    def __init__(self, dimension=3, dtype=float):
        self.dimension = dimension
        self.dtype = dtype
        self._clear()

    def load(self,
             filename,
             elements=['vertices', 'vertex_normals', 'faces', 'uvs']):
        self._clear()

        if get_file_extension(filename) == '.obj':
            self._load_obj(filename, elements)
        else:
            trim = trimesh.load(filename, process=False, maintain_order=True)
            self.vertices = trim.vertices
            self.faces = trim.faces

        return self

    def save(self, filename):
        create_parent_directory(filename)
        if get_file_extension(filename) == '.obj':
            self._save_obj(filename)
        else:
            trimesh.Trimesh(vertices=self.vertices,
                            faces=self.faces).export(filename)

    def copy(self):
        return copy.deepcopy(self)

    def compute_normals(self):

        # Sparse matrix that maps vertices to faces (and other way around)
        col_idx = np.repeat(np.arange(len(self.faces)), self.dimension)
        row_idx = self.faces.reshape(-1)
        data = np.ones(len(col_idx), dtype=bool)
        vert2face = scipy.sparse.coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(self.vertices), len(self.faces)),
            dtype=data.dtype)

        # Compute face normals
        f0 = self.vertices[self.faces[:, 0]]
        f1 = self.vertices[self.faces[:, 1]]
        f2 = self.vertices[self.faces[:, 2]]
        face_normals = np.cross(f1 - f0, f2 - f0)

        # For every vertex sum the normals of the faces its contained
        vertex_normals = vert2face.dot(face_normals)
        vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,
                                                         axis=1)[:, np.newaxis]

        self.vertex_normals = vertex_normals

    def cut(self, indices):

        # Cut vertices
        other = Mesh(self.dimension, self.dtype)
        other.vertices = self.vertices[indices].copy()
        if self.vertex_normals.any():
            other.vertex_normals = self.vertex_normals[indices].copy()
        if self.vertices_color.any():
            other.vertices_color = self.vertices_color[indices].copy()

        # Cut faces
        faces_mask = np.all(np.isin(self.faces, indices), axis=1)
        vertices_map = {new_i: i for (i, new_i) in enumerate(np.ravel(indices))}
        other.faces = np.vectorize(vertices_map.get)(self.faces[faces_mask])

        # Cut texture coords
        if self.texture_indices.any():
            other.texture_indices = self.texture_indices[faces_mask].copy()
        if self.texture_coordinates.any():
            other.texture_coordinates = self.texture_coordinates.copy()

        return other

    def _clear(self):
        self.vertices = np.ndarray(shape=(0, self.dimension), dtype=self.dtype)
        self.vertices_color = np.ndarray(shape=(0, self.dimension),
                                         dtype=self.dtype)
        self.vertex_normals = np.ndarray(shape=(0, self.dimension),
                                         dtype=self.dtype)
        self.faces = np.ndarray(shape=(0, 3), dtype=int)
        self.texture_coordinates = np.ndarray(shape=(0, 2), dtype=self.dtype)
        self.texture_indices = np.ndarray(shape=(0, 3), dtype=int)

    def _load_obj(self, filename, elements):
        assert get_file_extension(filename) == '.obj'

        with open(filename) as f:
            file_str = f.read()
            re.sub("^(?!\s*[vf]).*$", '', file_str, flags=re.M)
            v_count = file_str.count("v ")
            f_count = file_str.count("f ")
            vt_count = file_str.count("vt ")

            vertices = np.empty((v_count, self.dimension), dtype=self.dtype)
            vertices_color = np.empty((v_count, 3), dtype=self.dtype)
            vertex_normals = np.empty((v_count, self.dimension),
                                      dtype=self.dtype)
            faces = np.empty((f_count, 3), dtype=int)
            texture_coordinates = np.empty((vt_count, 2), dtype=self.dtype)
            texture_indices = np.empty((f_count, self.dimension), dtype=int)

            v_idx, vn_idx, vt_idx, f_idx = 0, 0, 0, 0
            vcol_flag, ti_flag = False, False
            for l in file_str.split('\n'):
                data = l.strip().split()

                if data == []:
                    continue

                if data[0] == 'v' and 'vertices' in elements:
                    vertices[v_idx] = [
                        self.dtype(data[1]),
                        self.dtype(data[2]),
                        self.dtype(data[3])
                    ]
                    if len(data) == 7:
                        vertices_color[v_idx] = [
                            self.dtype(data[4]),
                            self.dtype(data[5]),
                            self.dtype(data[6])
                        ]
                        vcol_flag = True
                    v_idx += 1

                elif data[0] == 'vn' and 'vertex_normals' in elements:
                    vertex_normals[vn_idx] = [
                        self.dtype(data[1]),
                        self.dtype(data[2]),
                        self.dtype(data[3])
                    ]
                    vn_idx += 1

                elif data[0] == 'vt' and 'uvs' in elements:
                    texture_coordinates[vt_idx] = [
                        self.dtype(data[1]),
                        self.dtype(data[2])
                    ]
                    vt_idx += 1

                elif data[0] == 'f' and 'faces' in elements:
                    face = [f.split('/') for f in data[:]]
                    faces[f_idx] = [
                        int(face[1][0]) - 1,
                        int(face[2][0]) - 1,
                        int(face[3][0]) - 1
                    ]

                    if (len(face[1]) == 2 or len(face[1]) == 3 and
                        (face[1][1] != '')) and 'uvs' in elements:
                        texture_indices[f_idx] = [
                            int(face[1][1]) - 1,
                            int(face[2][1]) - 1,
                            int(face[3][1]) - 1
                        ]
                        ti_flag = True
                    f_idx += 1

            if v_idx > 0:
                self.vertices = vertices
            if vcol_flag:
                self.vertices_color = vertices_color
            if vn_idx > 0:
                self.vertex_normals = vertex_normals
            if f_idx > 0:
                self.faces = faces
            if vt_idx > 0:
                self.texture_coordinates = texture_coordinates
            if ti_flag:
                self.texture_indices = texture_indices

    def _save_obj(self, filename):

        assert self.vertices.size != 0
        assert self.faces.size != 0

        with open(filename, 'w') as f:
            # Write vertices
            for v_id in range(self.vertices.shape[0]):
                f.write('v ' + str(self.vertices[v_id, 0]) + ' ' +
                        str(self.vertices[v_id, 1]) + ' ' +
                        str(self.vertices[v_id, 2]))
                if self.vertices_color.size:
                    f.write(' ' + str(self.vertices_color[v_id, 0]) + ' ' +
                            str(self.vertices_color[v_id, 1]) + ' ' +
                            str(self.vertices_color[v_id, 2]))
                f.write('\n')

            # Write vertex normals
            for vn_id in range(self.vertex_normals.shape[0]):
                f.write('vn ' + str(self.vertex_normals[vn_id, 0]) + ' ' +
                        str(self.vertex_normals[vn_id, 1]) + ' ' +
                        str(self.vertex_normals[vn_id, 2]) + '\n')

            # Write texture coordinates
            if self.texture_coordinates.size:
                for t_id in range(self.texture_coordinates.shape[0]):
                    f.write('vt ' + str(self.texture_coordinates[t_id, 0]) +
                            ' ' + str(self.texture_coordinates[t_id, 1]) + '\n')

            # Write faces
            for f_id in range(self.faces.shape[0]):
                if not self.texture_indices.size:
                    f.write('f ' + str(self.faces[f_id, 0] + 1) + ' ' +
                            str(self.faces[f_id, 1] + 1) + ' ' +
                            str(self.faces[f_id, 2] + 1) + '\n')
                else:
                    f.write( 'f ' + str( self.faces[ f_id, 0 ] + 1 ) + '/' + str( self.texture_indices[ f_id, 0 ] + 1 ) + \
                             ' ' + str( self.faces[ f_id, 1 ] + 1 ) + '/' + str( self.texture_indices[ f_id, 1 ] + 1 ) + \
                             ' ' + str( self.faces[ f_id, 2 ] + 1 ) + '/' + str( self.texture_indices[ f_id, 2 ] + 1 ) + '\n' )
