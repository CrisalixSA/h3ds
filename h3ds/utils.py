import os
import hashlib
import numpy as np
import matplotlib
import gdown


# Dataset pull
def download_file_from_google_drive(id: str, destination: str):

    gdown.download(id=id, output=destination, quiet=False)

def md5(filepath: str):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Filesystem
def get_file_extension(file):
    return os.path.splitext(file)[1]


def get_parent_directory(file, levels_up=1):
    for i in range(levels_up):
        file = os.path.split(file)[0]
    return file


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_parent_directory(path):
    parent_dir = get_parent_directory(path)
    create_directory(parent_dir)


def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def remove_file(file):
    if os.path.exists(file):
        os.remove(file)


def remove(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            remove_file(path)
        elif os.path.isdir(path):
            remove_directory(path)


# Visualization
def error_to_color(errors, clipping_error=None):
    if clipping_error is not None:
        errors_norm = np.clip(errors / float(clipping_error), 0., 1.)
    else:
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min())

    hsv = np.ones((errors.shape[-1], 3))
    hsv[:, 0] = (1. - errors_norm) / 3.

    return matplotlib.colors.hsv_to_rgb(hsv)
