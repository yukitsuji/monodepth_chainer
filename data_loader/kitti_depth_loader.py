#!/usr/bin/env python

import numpy as np
import chainer
import six
import os

from chainer import cuda, optimizers, Variable
import cv2

class KittiDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset_pathes, root='./input', dtype=np.float32, train=False):
        if isinstance(dataset_pathes, six.string_types):
            with open(dataset_pathes) as pathes_file:
                left_pathes = [path.strip().split(" ")[0] for path in pathes_file]
                right_pathes = [path.strip().split(" ")[1] for path in pathes_file]
        self._left_pathes = left_pathes
        self._right_pathes = right_pathes
        self._root = root
        self._dtype = dtype
        self._train = train

    def __len__(self):
        return len(self._left_pathes)

    def augment_image_pair(self, left_image, right_image):
        # randomly horizontal flip
        if np.random.rand() > 0.5:
            right_image = cv2.flip(left_image, 1)
            left_image = cv2.flip(right_image, 1)

        # randomly shift gamma
        random_gamma = np.random.uniform(low=0.8, high=1.2)
        left_image = left_image ** random_gamma
        right_image = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = np.random.uniform(low=0.5, high=2.0)
        left_image *= random_brightness
        right_image *= random_brightness

        # randomly shift color
        random_colors = np.random.uniform(low=0.8, high=1.2)
        white = np.ones((left_image.shape[0], left_image.shape[1]))
        color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image *= color_image
        right_image *= color_image

        return np.clip(left_image, 0.0, 1.0), np.clip(right_image, 0.0, 1.0)

    def get_example(self, i, minimize=False, log=False, bin_r=0):
        left_image = None
        right_image = None

        if self._train:
            left_path = os.path.join(self._root, self._left_pathes[i])
            right_path = os.path.join(self._root, self._right_pathes[i])
            left_image = cv2.imread(left_path, cv2.IMREAD_COLOR).astype(self._dtype)
            right_image = cv2.imread(right_path, cv2.IMREAD_COLOR).astype(self._dtype)
            left_image /= 255.
            right_image /= 255.
            self.augment_image_pair(left_image, right_image)
            left_image = left_image.transpose(2, 0, 1)
            right_image = right_image.transpose(2, 0, 1)
            return left_image, right_image
        else:
            left_path = os.path.join(self._root, self._left_pathes[i])
            left_image = cv2.imread(left_path, cv2.IMREAD_COLOR).astype(self._dtype)
            left_image /= 255.
            left_image = left_image.transpose(2, 0, 1)
            return left_image
