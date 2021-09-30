import os
import glob
import zipfile
import shutil
from functools import reduce

import toml
from tqdm import tqdm
import trimesh
from PIL import Image
import numpy as np

from h3ds.log import logger
from h3ds.utils import download_file_from_google_drive, md5
from h3ds.numeric import load_K_Rt, AffineTransform


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

    def default_views_config(self, scene_id: str):
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


class H3DS:

    def __init__(self, path: str, config_path: str = None):

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
        return self.helper.version_config() == self.helper.version_dataset() and \
            all([os.path.exists(f) for f in self.helper.files()])

    def scenes(self, tags: set = {}):
        return self.helper.scenes(tags)

    def default_views_configs(self, scene_id: str):
        return self.helper.default_views_configs()

    def load_scene(self,
                   scene_id: str,
                   views_config_id: str = None,
                   normalized: bool = False):
        mesh = self.load_mesh(scene_id, normalized)
        images = self.load_images(scene_id, views_config_id)
        masks = self.load_masks(scene_id, views_config_id)
        cameras = self.load_cameras(scene_id, views_config_id, normalized)

        return mesh, images, masks, cameras

    def load_mesh(self, scene_id: str, normalized: bool = False):
        mesh = trimesh.load(self.helper.scene_mesh(scene_id), process=False)
        if normalized:
            normalization_transform = self._load_normalization_transform(
                scene_id)
            mesh.vertices = normalization_transform.transform(mesh.vertices)

        return mesh

    def load_images(self, scene_id: str, views_config_id: str = None):
        images = self._load_images(self.helper.scene_images(scene_id))

        return self._filter_views(images, scene_id, views_config_id)

    def load_masks(self, scene_id: str, views_config_id: str = None):
        masks = self._load_images(self.helper.scene_masks(scene_id))

        return self._filter_views(masks, scene_id, views_config_id)

    def load_cameras(self,
                     scene_id: str,
                     views_config_id: str = None,
                     normalized: bool = False):
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

    def load_normalization_matrix(self, scene_id: str):
        '''
        Transforms the scene from mm scale to the unit sphere
        '''
        return self._load_normalization_transform(scene_id).matrix

    def _load_normalization_transform(self, scene_id: str):
        '''
        Transforms the scene into a unit sphere
        '''
        camera_dict = np.load(self.helper.scene_cameras(scene_id))
        s = camera_dict['scale_mat_0'].astype(np.float32)
        t = AffineTransform(matrix=np.linalg.inv(s))
        return t

    def _load_images(self, images_paths: list):
        return [Image.open(img).copy() for img in images_paths]

    def _get_views_config(self, scene_id: str, config_id: str):
        return self._config['scenes'][scene_id]['default_views_configs'][
            config_id]

    def _filter_views(self,
                      elements,
                      scene_id: str,
                      views_config_id: str = None):

        if views_config_id is None:
            return elements

        views_config = self._get_views_config(scene_id, views_config_id)
        return [elements[idx] for idx in views_config]
