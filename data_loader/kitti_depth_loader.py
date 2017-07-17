#!/usr/bin/env python

import numpy as np
import chainer
'''
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
'''
import six
import os

from chainer import cuda, optimizers, Variable
import cv2

def cvt2YUV(img):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    return img

class Image2ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root1='./input', root2='./terget', dtype=np.float32, leak=(0, 0), root_ref = None, train=False):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._img_dict = {}
        self._train = train

    def set_img_dict(self, img_dict):
        self._img_dict = img_dict

    def get_vec(self, name):
        tag_size = 1539
        v = np.zeros(tag_size).astype(np.int32)
        if name in self._img_dict.keys():
            for i in self._img_dict[name][3]:
                v[i] = 1
        return v

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return self._paths[i]

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9

        readed = False
        if np.random.rand() < bin_r:
            if np.random.rand() < 0.3:
                path1 = os.path.join(self._root1 + "_b2r/", self._paths[i])
            else:
                path1 = os.path.join(self._root1 + "_cnn/", self._paths[i])
            path2 = os.path.join(self._root2 + "_b2r/", self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_COLOR)
            if image1 is not None and image2 is not None:
                if image1.shape[0] > 0 and image1.shape[1] and image2.shape[0] > 0 and image2.shape[1]:
                    readed = True
        if not readed:
            path1 = os.path.join(self._root1, self._paths[i])
            path2 = os.path.join(self._root2, self._paths[i])
            image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(path2, cv2.IMREAD_COLOR)

        image2 = cvt2YUV( image2 )
        name1 = os.path.basename(self._paths[i])

        if self._train and np.random.rand() < 0.2:
            ret, image1 = cv2.threshold(
                image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # add flip and noise
        if self._train:
            if np.random.rand() > 0.5:
                image1 = cv2.flip(image1, 1)
                image2 = cv2.flip(image2, 1)
            if np.random.rand() > 0.9:
                image1 = cv2.flip(image1, 0)
                image2 = cv2.flip(image2, 0)

        image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2, self._dtype)

        if self._train:
            noise = np.random.normal(
                0, 5 * np.random.rand(), image1.shape).astype(self._dtype)
            image1 += noise
            noise = np.random.normal(
                0, 5 * np.random.rand(), image2.shape).astype(self._dtype)
            image2 += noise
            noise = np.random.normal(0, 16)
            image1 += noise
            image1[image1 < 0] = 0

        # image is grayscale
        if image1.ndim == 2:
            image1 = image1[:, :, np.newaxis]
        if image2.ndim == 2:
            image2 = image2[:, :, np.newaxis]

        image1 = np.insert(image1, 1, -512, axis=2)
        image1 = np.insert(image1, 2, 128, axis=2)
        image1 = np.insert(image1, 3, 128, axis=2)

        # randomly add terget image px
        if self._leak[1] > 0:
            image0 = image1
            n = np.random.randint(16, self._leak[1])
            if self._train:
                r = np.random.rand()
                if r < 0.4:
                    n = 0
                elif r < 0.7:
                    n = np.random.randint(2, 16)

            x = np.random.randint(1, image1.shape[0] - 1, n)
            y = np.random.randint(1, image1.shape[1] - 1, n)
            for i in range(n):
                for ch in range(3):
                    d = 20
                    v = image2[x[i]][y[i]][ch] + np.random.normal(0, 5)
                    v = np.floor(v / d + 0.5) * d
                    image1[x[i]][y[i]][ch + 1] = v
                if np.random.rand() > 0.5:
                    for ch in range(3):
                        image1[x[i]][y[i] + 1][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                        image1[x[i]][y[i] - 1][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                if np.random.rand() > 0.5:
                    for ch in range(3):
                        image1[x[i] + 1][y[i]][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]
                        image1[x[i] - 1][y[i]][ch +
                                               1] = image1[x[i]][y[i]][ch + 1]

        image1 = (image1.transpose(2, 0, 1))
        image2 = (image2.transpose(2, 0, 1))
        #image1 = (image1.transpose(2, 0, 1) -128) /128
        #image2 = (image2.transpose(2, 0, 1) -128) /128

        return image1, image2  # ,vec
