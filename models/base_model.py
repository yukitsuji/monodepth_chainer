#!/usr/bin/env python

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


class MonoDepth(chainer.Chain):
    def __init__(self, mode=False, use_deconv=False):
        self.train = mode
        upconv = None
        if use_deconv:
            upconv = L.Deconvolution2D
        else:
            upconv = F.resize_images

        super(MonoDepth, self).__init__(
            conv1_1=L.Convolution2D(3, 32, 7, stride=1, pad=3),
            conv1_2=L.Convolution2D(32, 32, 7, stride=2, pad=3),

            conv2_1=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            conv2_2=L.Convolution2D(64, 64, 5, stride=2, pad=2),

            conv3_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(128, 128, 3, stride=2, pad=1),

            conv4_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(256, 256, 3, stride=2, pad=1),

            conv5_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=2, pad=1),

            conv6_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv6_2=L.Convolution2D(512, 512, 3, stride=2, pad=1),

            conv7_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv7_2=L.Convolution2D(512, 512, 3, stride=2, pad=1),
        )

    def upconv(self, input):
        pass

    def deconv(self, input):
        pass

    def upsample_nn(self, input, rate):
        pass

    def get_disp(self, input):
        return

    def calc(self, left_images):
        """Network Process"""
        h = F.relu(self.conv1_1(left_images))
        conv1 = F.relu(self.conv1_2(h))

        h = F.relu(self.conv2_1(conv1))
        conv2 = F.relu(self.conv2_2(h))

        h = F.relu(self.conv3_1(conv2))
        conv3 = F.relu(self.conv3_2(h))

        h = F.relu(self.conv4_1(conv3))
        conv4 = F.relu(self.conv4_2(h))

        h = F.relu(self.conv5_1(conv4))
        conv5 = F.relu(self.conv5_2(h))

        h = F.relu(self.conv6_1(conv5))
        conv6 = F.relu(self.conv6_2(h))

        h = F.relu(self.conv7_1(conv6))
        conv7 = F.relu(self.conv7_2(h))

        upconv7 = self.upconv(conv7)
        concat7 = F.concat((upconv7, conv6), axis=3)
        iconv7 = F.relu(self.iconv7(concat7))

        upconv6 = self.upconv(iconv7)
        concat6 = F.concat((upconv6, conv5), axis=3)
        iconv6 = F.relu(self.iconv6(concat6))

        upconv5 = self.upconv(iconv6)
        concat5 = F.concat((upconv5, conv4), axis=3)
        iconv5 = F.relu(self.iconv5(concat5))

        upconv4 = self.upconv(iconv5)
        concat4 = F.concat((upconv4, conv3), axis=3)
        iconv4 = F.relu(self.iconv4(concat4))
        self.disp4 = self.get_disp(iconv4)
        udisp4 = self.upsample_nn(self.disp4, 2)

        upconv3 = self.upconv(iconv4)
        concat3 = F.concat([upconv3, conv2, udisp4], axis=3)
        iconv3 = F.relu(self.iconv3(concat3))
        self.disp3 = self.get_disp(iconv3)
        udisp3 = self.upsample_nn(self.disp3, 2)

        upconv2 = self.upconv(iconv3)
        concat2 = F.concat([upconv2, conv1, udisp3], axis=3)
        iconv2  = F.relu(self.iconv2(concat2))
        self.disp2 = self.get_disp(iconv2)
        udisp2 = self.upsample_nn(self.disp2, 2)

        upconv1 = self.upconv(iconv3)
        concat1 = F.concat([upconv1, udisp2], axis=3)
        iconv1  = F.relu(self.iconv1(concat1))
        self.disp1 = self.get_disp(iconv1)
        udisp1 = self.upsample_nn(self.disp1, 2)

    def __call__(self, x, t):
        self.calc(x)
        if self.train:
            # self.loss = F.softmax_cross_entropy(h, t)
            # self.acc = F.accuracy(h, t)
            return self.loss
        else:
            pass
            # self.pred = F.softmax(h)
            # return self.pred
