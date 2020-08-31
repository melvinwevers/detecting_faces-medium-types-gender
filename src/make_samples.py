import glob
import numpy as np
import os
import tarfile

'''
Script to make sample selection of original images
'''


def make_samples(zipped_archives, n_samples):
    for archive in zipped_archives:
        year = os.path.basename(archive)[:4]
        tar = tarfile.open(archive)
        tar.extractall()  # need to add proper directory to write to
        images = glob.glob(year + '/*.jpg')
        try:
            samples = np.random.choice(images, n_samples, replace=False)
        except Exception:
            pass  # some years do not have enough images

        for image in images:
            if image not in samples:
                os.remove(image)


if __name__ == '__main__':
    DATA_PATH = '../../datasets/ffads/'
    zipped_archives = glob.glob(DATA_PATH + '/*.gz')
    make_samples(zipped_archives, 1000)
