import os
import glob
import zipfile
import shutil

import toml
from tqdm import tqdm
import trimesh
from PIL import Image

from h3ds.log import logger
from h3ds.utils import download_file_from_google_drive, md5


class H3DS:

    def __init__(self, path: str):

        self.path = path

        if not os.path.exists(self.path):
            logger.warning(
                f'{self.path} does not exist. Change the path or call H3DS.download.'
            )

        self._load_config()

    def download(self, token):
        # Download zip file
        tmp_dir = os.path.join(self.path, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        tmp_zip = os.path.join(tmp_dir, 'h3ds.zip')
        logger.print(f'Downloading H3DS dataset to {tmp_zip}')
        download_file_from_google_drive(id=self._config['file_id'],
                                        destination=tmp_zip)

        # Check md5
        md5_zip = md5(tmp_zip)
        if md5_zip == self._config['file_md5']:
            logger.print('MD5 check - Success')
        else:
            logger.error('MD5 check - Failed')

        # Unzip into self.path
        logger.print(f'Unzipping file to {self.path}')
        with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting...'):
                    zip_ref.extract(member, self.path, pwd=token.encode('utf-8'))

        # Remove temporal zip
        logger.print(f'Removing temporary files')
        shutil.rmtree(tmp_dir)

    def scenes(self):
        return self._config['scenes']

    def default_views_configs(self, scene_id: str):
        return list(self._config['views'][scene_id].keys())

    def load_scene(self, scene_id: str, views_config_id: str = None):
        scene_path = os.path.join(self.path, scene_id)

        mesh = self.get_mesh(scene_id)
        images = self.get_images(scene_id, views_config_id)
        masks = self.get_masks(scene_id, views_config_id)
        cameras = self.get_cameras(scene_id, views_config_id)

        return mesh, images, masks, cameras

    def get_mesh(self, scene_id: str):
        mesh_path = os.path.join(self.path, scene_id, 'full_head.obj')

        return trimesh.load(mesh_path, process=False)

    def get_images(self, scene_id: str, views_config_id: str = None):
        images = self._get_images(scene_id, 'image')

        return self._filter_views(images, scene_id, views_config_id)

    def get_masks(self, scene_id: str, views_config_id: str = None):
        masks = self._get_images(scene_id, 'mask')

        return self._filter_views(masks, scene_id, views_config_id)

    def get_cameras(self, scene_id: str, views_config_id: str = None):
        cameras = self._get_images(scene_id, 'mask')

        return self._filter_views(cameras, scene_id, views_config_id)

    def _load_config(self):
        h3ds_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(h3ds_dir, 'config.toml')

        self._config = toml.load(config_path)

    def _get_views_config(self, scene_id: str, config_id: str):

        if scene_id not in self._config['views']:
            logger.exception(f'Invalid scene_id {scene_id}')

        if config_id not in self._config['views'][scene_id]:
            logger.exception(
                f'Invalid config_id {config_id} for scene_id {scene_id}')

        return self._config['views'][scene_id][config_id]

    def _get_images(self, scene_id: str, rel_path: str):
        images_dir = os.path.join(self.path, scene_id, rel_path)
        images_paths = sorted(list(glob.glob(os.path.join(images_dir,
                                                          '*.jpg'))))
        return [Image.open(img).copy() for img in images_paths]

    def _filter_views(self,
                      elements,
                      scene_id: str,
                      views_config_id: str = None):

        if views_config_id is None:
            return elements

        views_config = self._get_views_config(scene_id, views_config_id)
        return [elements[idx] for idx in views_config]
