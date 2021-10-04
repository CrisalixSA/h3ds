import os
import glob
import zipfile
import shutil
from functools import reduce

import toml
from tqdm import tqdm
from PIL import Image
import numpy as np

from h3ds.log import logger
from h3ds.mesh import Mesh
from h3ds.affine_transform import AffineTransform
from h3ds.utils import download_file_from_google_drive, md5
from h3ds.numeric import load_K_Rt, perform_alignment, perform_icp, transform_mesh, unidirectional_chamfer_distance


class ConfigsHelper:

    identifiers = ['config']

    @classmethod
    def configs(cls):
        return cls.identifiers

    @classmethod
    def is_available(cls, config_id: str):
        return config_id in cls.configs()

    @staticmethod
    def get_config_file(config_id: str):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(config_dir, f'{config_id}.toml')


class H3DSHelper:

    def __init__(self, path, config_path: str):
        self.path = path
        self._config = toml.load(config_path)

    def version_config(self):
        return str(self._config['version'])

    def version_dataset(self):
        try:
            with open(self.version_file()) as f:
                return str(f.readline().rstrip())
        except:
            return None

    def scenes_tags(self):
        return reduce(lambda x,y: x.union(y), \
            [self.scene_tags(s) for s in self._config['scenes'].keys()])

    def scenes(self, tags: set = {}):
        if not set(tags).issubset(self.scenes_tags()):
            logger.critical(
                f'{tags} tags not available. Call H3DSHelper.scenes_tags to list the available tags'
            )

        scenes = list(self._config['scenes'].keys())
        if tags:
            cond = lambda s: set(tags).issubset(set(self.scene_tags(s)))
            scenes = list(filter(cond, scenes))
        return scenes

    def files(self):
        return [self.version_file()] + \
            reduce(lambda x, y: x + y, [self.scene_files(s) for s in self.scenes()])

    def default_views_configs(self, scene_id: str):
        return list(
            self._config['scenes'][scene_id]['default_views_configs'].keys())

    def scene_views(self, scene_id: str):
        return self._config['scenes'][scene_id]['views']

    def version_file(self):
        return os.path.join(self.path, 'version.txt')

    def scene_tags(self, scene_id):
        return set(self._config['scenes'][scene_id].get('tags', []))

    def scene_files(self, scene_id: str):
        return [self.scene_mesh(scene_id)
               ] + self.scene_images(scene_id) + self.scene_masks(scene_id) + [
                   self.scene_cameras(scene_id)
               ]

    def scene_mesh(self, scene_id: str):
        return os.path.join(self.path, scene_id, 'full_head.obj')

    def scene_images(self, scene_id: str):
        return [
            os.path.join(self.path, scene_id, 'image',
                         'img_{0:04}.jpg'.format(idx))
            for idx in range(self.scene_views(scene_id))
        ]

    def scene_masks(self, scene_id: str):
        return [
            os.path.join(self.path, scene_id, 'mask',
                         'mask_{0:04}.png'.format(idx))
            for idx in range(self.scene_views(scene_id))
        ]

    def scene_cameras(self, scene_id: str):
        return os.path.join(self.path, scene_id, 'cameras.npz')

    def scene_landmarks(self, scene_id: str):
        return os.path.join(self.path, scene_id, 'landmarks.txt')

    def scene_region(self, scene_id: str, region_id: str):
        return os.path.join(self.path, scene_id, 'regions', f'{region_id}.txt')


class H3DS:

    def __init__(self, path: str, config_path: str = None):
        """
        Class to manage the data available in the H3DS dataset.
        Args:
            path        (str): Path to store the dataset locally.
            config_path (str): Optional custom config file.
        """
        self.path = os.path.expanduser(path)
        self.config_path = config_path or ConfigsHelper.get_config_file(
            'config')
        self.helper = H3DSHelper(path=self.path, config_path=self.config_path)
        self._config = self.helper._config

        if not self.is_available():
            logger.warning(
                f'H3DS v{self.helper.version_config()} was not found at {self.path}. Change the path or call H3DS.download.'
            )

    def download(self, token, force=False):
        """
        Downloads the dataset to the specified path in the __init__ method. The dataset
        is download only if it is not available or if the flag force is True.
        Args:
            token  (str): H3DS token
            force (bool): Flag to force the download
        Returns:
            None
        """
        # Check if dataset is already available
        if self.is_available() and not force:
            logger.info('Dataset already available. Skipping download')
            return

        # Download zip file
        tmp_dir = os.path.join(self.path, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        version = self._config['version']
        tmp_zip = os.path.join(tmp_dir, f'h3ds_{version}.zip')
        logger.print(f'Downloading H3DS dataset to {tmp_zip}')
        download_file_from_google_drive(id=self._config['file_id'],
                                        destination=tmp_zip)

        # Check md5
        md5_zip = md5(tmp_zip)
        if md5_zip == self._config['file_md5']:
            logger.print('MD5 check - Success')
        else:
            logger.critical('MD5 check - Failed')

        # Unzip into self.path
        logger.print(f'Unzipping file to {self.path}')
        with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting...'):
                zip_ref.extract(member, self.path, pwd=token.encode('utf-8'))

        # Remove temporal zip
        logger.print(f'Removing temporary files')
        shutil.rmtree(tmp_dir)

    def is_available(self):
        """
        Checks if a valid version of the dataset is available at the specified path
        Args:
        Returns:
            bool : True if available, otherwise false
        """
        return self.helper.version_config() == self.helper.version_dataset() and \
            all([os.path.exists(f) for f in self.helper.files()])

    def scenes(self, tags: set = {}):
        """
        Specifies the available scenes in the dataset as scene identifiers. A scene identifier
        can be used to load data for a concrete scene, for instance the mesh or the cameras.
        If tags are specified, the list is filtered and only the scenes containing the tag are retuned.
        If more than one tag is provided, only the scenes containing all the tags will be returned.
        Args:
            tags (set): Optional set of tags to filter the scenes, i.e. 'h3d-net'.
        Returns:
            list : List scene identifiers (str)
        """
        return self.helper.scenes(tags)

    def regions(self):
        """
        Specifies the list of region identifiers that are available for each scene.
        Args:
        Returns:
            list : List of regions identifiers (str)
        """
        return ['face', 'face_sphere', 'nose']

    def default_views_configs(self, scene_id: str):
        """
        Specifies all the subsets of views (configs) available for a certain scene. Each
        configuration of views is associated to a view_config_identifier.
        Args:
            scene_id (str): Scene identifier
        Returns:
            list : List of view config identifiers available for that scene.
        """
        return self.helper.default_views_configs(scene_id)

    def load_scene(self,
                   scene_id: str,
                   views_config_id: str = None,
                   normalized: bool = False):
        """
        Loads all the elements of a scene, which are the mesh, the images,
        the masks and the cameras.
        Args:
            scene_id        (str): Scene identifier
            views_config_id (str): Views configuration defining subset of views
            normalized     (bool): Scene normalized to fit inside a unit sphere
        Returns:
            Mesh: The 3D geometry of the scene as a mesh
            list: Array of the images
            list: Array of the masks
            list: Array of the cameras
        """
        mesh = self.load_mesh(scene_id, normalized)
        images = self.load_images(scene_id, views_config_id)
        masks = self.load_masks(scene_id, views_config_id)
        cameras = self.load_cameras(scene_id, views_config_id, normalized)

        return mesh, images, masks, cameras

    def load_mesh(self, scene_id: str, normalized: bool = False):
        """
        Loads the mesh for a given scene.
        Args:
            scene_id    (str): Scene identifier
            normalized (bool): Scene normalized to fit inside a unit sphere
        Returns:
            Mesh: The 3D geometry of the scene as a mesh
        """
        mesh = Mesh().load(self.helper.scene_mesh(scene_id))
        if normalized:
            normalization_transform = self._load_normalization_transform(
                scene_id)
            mesh.vertices = normalization_transform.transform(mesh.vertices)

        return mesh

    def load_images(self, scene_id: str, views_config_id: str = None):
        """
        Loads the RGB images for a given scene as PIL.Image.
        Args:
            scene_id        (str): Scene identifier
            views_config_id (str): Views configuration defining subset of views
        Returns:
            list : Array of the images
        """
        images = self._load_images(self.helper.scene_images(scene_id))

        return self._filter_views(images, scene_id, views_config_id)

    def load_masks(self, scene_id: str, views_config_id: str = None):
        """
        Loads the binary masks for a given scene as PIL.Image.
        Args:
            scene_id        (str): Scene identifier
            views_config_id (str): Views configuration defining subset of views
        Returns:
            list : Array of the masks
        """
        masks = self._load_images(self.helper.scene_masks(scene_id))

        return self._filter_views(masks, scene_id, views_config_id)

    def load_cameras(self,
                     scene_id: str,
                     views_config_id: str = None,
                     normalized: bool = False):
        """
        Loads the cameras for a given scene. Each cameras is defined as a tupple
        of two elements. The first one is a 3x3 np.ndarray matrix with the calibration
        and the second is a 4x4 np.ndarray matrix with the camera pose.
        Args:
            scene_id        (str): Scene identifier
            views_config_id (str): Views configuration defining subset of views
            normalized     (bool): Scene normalized to fit inside a unit sphere
        Returns:
            list : Array of the cameras
        """
        camera_dict = np.load(self.helper.scene_cameras(scene_id))
        if normalized:
            normalization_transform = self._load_normalization_transform(
                scene_id)

        cameras = []
        for idx in range(self._config['scenes'][scene_id]['views']):
            P = camera_dict['world_mat_%d' % idx].astype(np.float32)
            if normalized:
                P = P @ np.linalg.inv(normalization_transform.matrix)
            K, P = load_K_Rt(P[:3, :4])
            cameras.append((K, P))

        return self._filter_views(cameras, scene_id, views_config_id)

    def load_landmarks(self, scene_id: str):
        """
        Loads the landmarks for a given scene as dictionary. Each landmark
        is provided as a vertex index of the mesh.vertices from the scene.
        Args:
            scene_id    (str): Scene identifier
        Returns:
            dict: A dictionary with the annotated landmarks
        """
        with open(self.helper.scene_landmarks(scene_id)) as f:
            landmarks = {
                l[0]: int(l[1])
                for l in [l.rstrip().split() for l in f.readlines()]
            }

        return landmarks

    def load_region(self, scene_id: str, region_id: str):
        """
        Loads a list of indices defining a region of the mesh. The available regions are:
        - 'face': Includes the frontal face, the ears and the neck.
        - 'face_sphere': Includes all the vertices inside sphere centered at the tip
                         of the nose with radius 95mm (standard for evaluation).
        - 'nose': Includes only the vertices belonging to the nose.
        Args:
            scene_id  (str): Scene identifier
            region_id (str): Region identifier
        Returns:
            np.ndarray: An array containing either a list of indices or a mask
        """
        with open(self.helper.scene_region(scene_id, region_id)) as f:
            region = np.array([np.int(l.rstrip()) for l in f.readlines()])

        return region

    def load_normalization_matrix(self, scene_id: str):
        """
        Loads the transformation that normalizes the scene from mm to a unit sphere.
        Args:
            scene_id, (str): Scene identifier
        Returns:
            np.array : A 4x4 similarity transform
        """
        return self._load_normalization_transform(scene_id).matrix

    def evaluate_scene(self,
                       scene_id: str,
                       mesh_pred: Mesh,
                       landmarks_pred: dict = None,
                       region_id: str = None):
        """
        Evaluates a predicted mesh with respect the ground truth scene. If landmarks
        are provided, the predicted mesh is coarsely aligned towards the ground truth.
        Then, ICP is performed to finely align the meshes, and finally the chamfer
        distance is computed in both ways. The landmarks must be provided as
        a dictionary with the following structure {landmark_id: vertex_id}.

        The required landmarks_ids (if provided) are:
        [right_eye, left_eye, nose_tip, nose_base, right_lips, left_lips]

        Finally, if a region identifier is provided, the scene is evaluated in that
        specific region. By default it evaluates with the whole head.

        See the README and the examples for more information

        Args:
            scene_id,       (str): Scene identifier
            mesh_pred      (Mesh): Predicted mesh for that scene
            landmarks_pred (dict): Landkarks on the predicted mesh
            region_id       (str): Region identifier
        Returns:
            np.array: Nx3 array with the chamfer distance gt->pred for each groundtruth vertex
            np.array: Mx3 array with the chamfer distance pred->gt for eacu predicted vertex
            Mesh    : Ground truth mesh from H3DS
            Mesh    : Finely aligned predicted mesh
        """
        mesh_gt = self.load_mesh(scene_id)
        landmarks_gt = self.load_landmarks(scene_id)
        region_gt = self.load_region(scene_id, region_id or 'face')

        # Perform coarse alignment if landmarks provided
        mesh_pred, t_coarse = perform_alignment(mesh_pred, mesh_gt,
                                                landmarks_pred, landmarks_gt)

        # Perform fine alignment using ICP
        _, t_icp = perform_icp(mesh_gt, mesh_pred, region_gt)
        mesh_pred = transform_mesh(mesh_pred, np.linalg.inv(t_icp))

        # Compute chamfers. Use the region if specified
        if region_id:
            mesh_gt = mesh_gt.cut(region_gt)

        chamfer_gt_pred = unidirectional_chamfer_distance(
            mesh_gt.vertices, mesh_pred.vertices)
        chamfer_pred_gt = unidirectional_chamfer_distance(
            mesh_pred.vertices, mesh_gt.vertices)

        return chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred

    def _load_normalization_transform(self, scene_id: str):
        """
        Internal method: Loads the transformation that normalizes the scene
        from mm to a unit sphere.
        Args:
            scene_id (str): Scene identifier
        Returns:
            AffineTransform : A 4x4 similarity transform
        """
        camera_dict = np.load(self.helper.scene_cameras(scene_id))
        s = camera_dict['scale_mat_0'].astype(np.float32)
        t = AffineTransform(matrix=np.linalg.inv(s))
        return t

    def _load_images(self, images_paths: list):
        """
        Internal method: Loads a list of image as PIL.Image from their paths
        Args:
            images_paths (list): List of image paths
        Returns:
            list : List of images as PIL.Image
        """
        return [Image.open(img).copy() for img in images_paths]

    def _get_views_config(self, scene_id: str, config_id: str):
        """
        Loads a list of list of view identifiers that is pre-defined
        in the config.toml file.
        Args:
            scene_id (str): Scene identifier
            config_id (str): Config identifier (see default_views_configs method)
        Returns:
            list : Array of view identifiers
        """
        return self._config['scenes'][scene_id]['default_views_configs'][
            config_id]

    def _filter_views(self,
                      elements,
                      scene_id: str,
                      views_config_id: str = None):
        """
        Internal method: Filters a list of objects associated to views (images, cameras)
        according to a scene and a configuration of views.
        Args:
            elements (list): List of all the images, masks or cameras from a scene
        Returns:
            list : Filtered list of elements that belong to the views defined by views_config_id
        """
        if views_config_id is None:
            return elements

        views_config = self._get_views_config(scene_id, views_config_id)
        return [elements[idx] for idx in views_config]