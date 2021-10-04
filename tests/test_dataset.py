import os
import toml
import tempfile
import pathlib
import unittest
from unittest.mock import patch

import trimesh
import numpy as np
from PIL import Image

from h3ds.dataset import ConfigsHelper, H3DSHelper, H3DS
from h3ds.affine_transform import AffineTransform


class TestConfigsHelper(unittest.TestCase):

    def test_configs(self):
        self.assertEqual(ConfigsHelper.configs(), ['config'])

    def test_is_available(self):
        self.assertTrue(ConfigsHelper.is_available('config'))
        self.assertFalse(ConfigsHelper.is_available('config_new'))

    def test_get_config_file(self):
        config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'h3ds', 'config.toml')
        self.assertEqual(ConfigsHelper.get_config_file('config'), config_file)


class TestH3DSBase(unittest.TestCase):

    def setUp(self):

        self.path = tempfile.mkdtemp()
        _, self.config_path = tempfile.mkstemp()

        self.config = toml.dumps({
            'file_id': '1234',
            'file_md5': 'abcd',
            'version': 0.1,
            'scenes': {
                'a1b2c3': {
                    'tags': ['tag'],
                    'views': 3,
                    'default_views_configs': {
                        '3': [0, 1, 2]
                    }
                }
            }
        })
        with open(self.config_path, 'w') as f:
            f.write(self.config)
        self.helper = H3DSHelper(self.path, self.config_path)


class TestH3DSHelper(TestH3DSBase):

    def test_scenes(self):
        self.assertEqual(self.helper.scenes(), ['a1b2c3'])

    def test_files(self):
        files = self.helper.files()
        self.assertTrue(os.path.join(self.path, 'version.txt') in files)
        for id in self.helper.scenes():
            scene_path = os.path.join(self.path, 'a1b2c3')
            self.assertTrue(os.path.join(scene_path, 'full_head.obj') in files)
            self.assertTrue(os.path.join(scene_path, 'cameras.npz') in files)
            for idx in range(self.helper.scene_views('a1b2c3')):
                self.assertTrue(
                    os.path.join(scene_path, 'image', 'img_{0:04}.jpg'.format(
                        idx)) in files)
                self.assertTrue(
                    os.path.join(scene_path, 'mask', 'mask_{0:04}.png'.format(
                        idx)) in files)

    def test_default_views_configs(self):
        self.assertEqual(self.helper.default_views_configs('a1b2c3'), ['3'])

    def test_scene_tags(self):
        self.assertEqual(self.helper.scenes_tags(), set(['tag']))
        self.assertRaises(Exception, self.helper.scenes, tags={'wrong-tag'})
        try:
            self.helper.scenes(tags={'tag'})
        except:
            self.fail("Error loading scenes with tag 'tag'")

    def test_scene_views(self):
        self.assertEqual(self.helper.scene_views('a1b2c3'), 3)

    def test_scene_files(self):
        scene_files = self.helper.scene_files('a1b2c3')
        scene_path = os.path.join(self.path, 'a1b2c3')
        self.assertTrue(
            os.path.join(scene_path, 'full_head.obj') in scene_files)
        self.assertTrue(os.path.join(scene_path, 'cameras.npz') in scene_files)
        for idx in range(self.helper.scene_views('a1b2c3')):
            self.assertTrue(
                os.path.join(scene_path, 'image', 'img_{0:04}.jpg'.format(idx))
                in scene_files)
            self.assertTrue(
                os.path.join(scene_path, 'mask', 'mask_{0:04}.png'.format(idx))
                in scene_files)

    def test_scene_mesh(self):
        self.assertEqual(os.path.join(self.path, 'a1b2c3', 'full_head.obj'),
                         self.helper.scene_mesh('a1b2c3'))

    def test_scene_images(self):
        scene_image_files = self.helper.scene_images('a1b2c3')
        scene_path = os.path.join(self.path, 'a1b2c3')
        for idx in range(self.helper.scene_views('a1b2c3')):
            self.assertTrue(
                os.path.join(scene_path, 'image', 'img_{0:04}.jpg'.format(idx))
                in scene_image_files)

    def test_scene_masks(self):
        scene_masks_files = self.helper.scene_masks('a1b2c3')
        scene_path = os.path.join(self.path, 'a1b2c3')
        for idx in range(self.helper.scene_views('a1b2c3')):
            self.assertTrue(
                os.path.join(scene_path, 'mask', 'mask_{0:04}.png'.format(idx))
                in scene_masks_files)

    def test_scene_cameras(self):
        self.assertEqual(os.path.join(self.path, 'a1b2c3', 'cameras.npz'),
                         self.helper.scene_cameras('a1b2c3'))


class TestDataset(TestH3DSBase):

    def setUp(self):
        super().setUp()
        with open(self.helper.version_file(), 'w') as f:
            f.write(self.helper.version_config())
        for s in self.helper.scenes():
            os.makedirs(os.path.join(self.path, s))
            os.makedirs(os.path.join(self.path, s, 'image'))
            os.makedirs(os.path.join(self.path, s, 'mask'))
            trimesh.primitives.Box().export(self.helper.scene_mesh(s))
            cameras = {}
            for idx, (i, m) in enumerate(
                    zip(self.helper.scene_images(s),
                        self.helper.scene_masks(s))):
                Image.fromarray(np.random.rand(8, 8).astype(np.uint8)).save(i)
                Image.fromarray(np.random.rand(8, 8).astype(np.uint8)).save(m)
                cameras['scale_mat_%d' % idx] = np.random.rand(4, 4)
                cameras['world_mat_%d' % idx] = np.random.rand(4, 4)
            np.savez(self.helper.scene_cameras(s), **cameras)

    def test_default_views_configs(self):
        h3ds = H3DS(path=self.path, config_path=self.config_path)
        self.assertEqual(h3ds.default_views_configs('a1b2c3'), ['3'])

    def test_is_available(self):

        # Dataset not available
        h3ds = H3DS(path='wrong_path', config_path=self.config_path)
        self.assertFalse(h3ds.is_available())

        # Dataset available
        h3ds = H3DS(path=self.path, config_path=self.config_path)
        self.assertTrue(h3ds.is_available())

    def test_load_scene(self):

        h3ds = H3DS(path=self.path, config_path=self.config_path)
        try:
            mesh, images, masks, cameras = h3ds.load_scene('a1b2c3')
        except ExceptionType:
            self.fail("load_scene raised exception")


if __name__ == '__main__':
    unittest.main()
