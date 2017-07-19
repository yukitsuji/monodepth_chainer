#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os
from PIL import Image

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse

from data_loader.kitti_depth_loader import KittiDataset
from models.base_model import MonoDepth

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"


class monodepthUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.md = kwargs.pop('models')
        self._iter = 0
        super(monodepthUpdater, self).__init__(*args, **kwargs)

    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(self, img):
        return img[:, :, :-1] - img[:, :, 1:]

    def get_disparity_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        img_gradients_x = self.gradient_x(img)
        img_gradients_y = self.gradient_y(img)

        weight_x = F.exp(-F.mean(F.absolute(disp_gradients_x), axis=1, keep_dims=True))
        weight_y = F.exp(-F.mean(F.absolute(disp_gradients_y), axis=1, keep_dims=True))

        smoothness_x = disp_gradients_x * weight_x
        smoothness_y = disp_gradients_y * weight_y
        return smoothness_x + smoothness_y

    def scale_pyramid(self, images, scale_h, scale_w):
        return F.resize_images(images, (scale_h, scale_w))

    def ssim(self, pred, orig):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        # TODO: Check argument
        mu_pred = F.average_pooling_2d(pred, 3, 1, "VALID")
        mu_orig = F.average_pooling_2d(orig, 3, 1, "VALID")

        sigma_pred = F.average_pooling_2d(pred ** 2, 3, 1, "VALID") - mu_pred ** 2
        sigma_orig = F.average_pooling_2d(orig ** 2, 3, 1, "VALID") - mu_orig ** 2
        sigma_both = F.average_pooling_2d(pred * orig, 3, 1, "VALID") - mu_pred * mu_orig

        ssim_n = (2 * mu_pred * mu_orig + c1) * (2 * sigma_both + c2)
        ssim_d = (mu_pred ** 2 + mu_orig ** 2 + c1) * (sigma_pred + sigma_orig + c2)
        ssim = ssim_n / ssim_d
        return F.clip((1 - ssim) / 2, 0.0, 1.0)

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d(img, disp)

    def loss_md(self, left_images, right_images):
        h, w = left_images.shape[2:]
        left_img1 = left_images
        left_img2 = self.rescale_img(left_images, h/2, w/2)
        left_img3 = self.rescale_img(left_images, h/4, w/4)
        left_img4 = self.rescale_img(left_images, h/8, w/8)

        right_img1 = right_images
        right_img2 = self.rescale_img(right_images, h/2, w/2)
        right_img3 = self.rescale_img(right_images, h/4, w/4)
        right_img4 = self.rescale_img(right_images, h/8, w/8)

        self.disp1_left_est = self.md.disp1[:, 0]
        self.disp2_left_est = self.md.disp2[:, 0]
        self.disp3_left_est = self.md.disp3[:, 0]
        self.disp4_left_est = self.md.disp4[:, 0]

        self.disp1_right_est = self.md.disp1[:, 1]
        self.disp2_right_est = self.md.disp2[:, 1]
        self.disp3_right_est = self.md.disp3[:, 1]
        self.disp4_right_est = self.md.disp4[:, 1]

        # TODO: Generate Images
        self.left_est1 = self.generate_image_left(self.right_img1, self.disp1_left_est)
        self.left_est2 = self.generate_image_left(self.right_img2, self.disp2_left_est)
        self.left_est3 = self.generate_image_left(self.right_img3, self.disp3_left_est)
        self.left_est4 = self.generate_image_left(self.right_img4, self.disp4_left_est)

        self.right_est1 = self.generate_image_right(self.left_img1, self.disp1_right_est)
        self.right_est2 = self.generate_image_right(self.left_img2, self.disp2_right_est)
        self.right_est3 = self.generate_image_right(self.left_img3, self.disp3_right_est)
        self.right_est4 = self.generate_image_right(self.left_img4, self.disp4_right_est)

        # TODO: LR Consistency
        self.right_to_left_disp1 = self.generate_image_left(self.disp1_right_est, self.disp1_left_est)
        self.right_to_left_disp2 = self.generate_image_left(self.disp2_right_est, self.disp2_left_est)
        self.right_to_left_disp3 = self.generate_image_left(self.disp3_right_est, self.disp3_left_est)
        self.right_to_left_disp4 = self.generate_image_left(self.disp4_right_est, self.disp4_left_est)

        self.left_to_right_disp1 = self.generate_image_right(self.disp1_left_est, self.disp1_right_est)
        self.left_to_right_disp2 = self.generate_image_right(self.disp2_left_est, self.disp2_right_est)
        self.left_to_right_disp3 = self.generate_image_right(self.disp3_left_est, self.disp3_right_est)
        self.left_to_right_disp4 = self.generate_image_right(self.disp4_left_est, self.disp4_right_est)

        # TODO: L1 Loss
        self.l1_left1 = F.mean(F.absolute(self.left_est1 - self.left_img1))
        self.l1_left2 = F.mean(F.absolute(self.left_est2 - self.left_img2))
        self.l1_left3 = F.mean(F.absolute(self.left_est3 - self.left_img3))
        self.l1_left4 = F.mean(F.absolute(self.left_est4 - self.left_img4))

        self.l1_right1 = F.mean(F.absolute(self.right_est1 - self.right_img1))
        self.l1_right2 = F.mean(F.absolute(self.right_est2 - self.right_img2))
        self.l1_right3 = F.mean(F.absolute(self.right_est3 - self.right_img3))
        self.l1_right4 = F.mean(F.absolute(self.right_est4 - self.right_img4))

        # TODO: SSIM Loss
        self.ssim_left1 = F.mean(self.ssim(self.left_est1, self.left_img1))

        # TODO: Weighted Sum of L1 and SSIM loss

        # TODO: LR Consistency Loss

        # deiparity smoothness error using gradient [:-1], [1:]
        self.disp1_left_smoothness = self.get_disparity_smoothness(self.md.disp1, left_img1)
        self.disp2_left_smoothness = self.get_disparity_smoothness(self.md.disp2, left_img2)
        self.disp3_left_smoothness = self.get_disparity_smoothness(self.md.disp3, left_img3)
        self.disp4_left_smoothness = self.get_disparity_smoothness(self.md.disp4, left_img4)

        self.disp1_right_smoothness = self.get_disparity_smoothness(self.md.disp1, right_img1)
        self.disp2_right_smoothness = self.get_disparity_smoothness(self.md.disp2, right_img2)
        self.disp3_right_smoothness = self.get_disparity_smoothness(self.md.disp3, right_img3)
        self.disp4_right_smoothness = self.get_disparity_smoothness(self.md.disp4, right_img4)

        self.disp1_left_loss = F.mean(F.absolute(self.disp1_left_smoothness))
        self.disp2_left_loss = F.mean(F.absolute(self.disp2_left_smoothness)) / 2
        self.disp3_left_loss = F.mean(F.absolute(self.disp3_left_smoothness)) / 4
        self.disp4_left_loss = F.mean(F.absolute(self.disp4_left_smoothness)) / 8

        self.disp1_right_loss = F.mean(F.absolute(self.disp1_right_smoothness))
        self.disp2_right_loss = F.mean(F.absolute(self.disp2_right_smoothness)) / 2
        self.disp3_right_loss = F.mean(F.absolute(self.disp3_right_smoothness)) / 4
        self.disp4_right_loss = F.mean(F.absolute(self.disp4_right_smoothness)) / 8

        total_smoothness_loss = self.disp1_left_loss + self.disp2_left_loss
                              + self.disp3_left_loss + self.disp4_left_loss
                              + self.disp1_right_loss + self.disp2_right_loss
                              + self.disp3_right_loss + self.disp4_right_loss

        # TODO: Total Loss

        # loss_rec = lam1 * (F.mean_absolute_error(x_out, right_images))
        # chainer.report({'loss': loss, "loss_rec": loss_rec,
        #                 'loss_adv': loss_adv, "loss_l": loss_l}, md)
        return total_loss

    def update_core(self):
        xp = self.md.xp
        self._iter += 1

        batch = self.get_iterator('train').next()

        # CPU to GPU
        batchsize = len(batch)
        w_in = batch.shape[-1]
        h_in = batch.shape[-2]
        left_images = xp.zeros((batchsize, 3, h_in, w_in)).astype("f")
        right_images = xp.zeros((batchsize, 3, h_in, w_in)).astype("f")

        for i in range(batchsize):
            left_images[i, :] = xp.asarray(batch[i][0])
            right_images[i, :] = xp.asarray(batch[i][1])
        left_images = Variable(left_images)
        right_images = Variable(right_images)

        self.md.calc(left_images)
        md_optimizer = self.get_optimizer('md')
        md_optimizer.update(self.loss_md, left_images, right_images)


def train(args):
    dataset = KittiDataset(
        args.dataset, root=args.root, dtype=np.float32, train=True)
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    md = MonoDepth()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        md.to_gpu()

    opt = optimizers.Adam()
    opt.setup(md)

    updater = monodepthUpdater(
        models=(md),
        iterator={
            'train': train_iter},
        optimizer={
            'md': opt},
        device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    trainer.extend(extensions.dump_graph('md/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        md, 'md_vgg_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'md/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=20))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    chainer.serializers.save_npz(os.path.join(save_dir, 'model_final'), md)

def test(args):
    dataset = KittiDataset(
        args.dataset, root=args.root, dtype=np.float32, train=False)
    test_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    md = MonoDepth()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        md.to_gpu()

    if args.resume:
        chainer.serializers.load_npz(args.resume, md)

    for batch in test_iter:
        batchsize = len(batch)
        w_in = batch.shape[-1]
        h_in = batch.shape[-2]
        left_images = xp.zeros((batchsize, 3, h_in, w_in)).astype("f")

        for i in range(batchsize):
            left_images[i, :] = xp.asarray(batch[i][0])
        left_images = Variable(left_images)
        disparity = md(left_images)

def main():
    parser = argparse.ArgumentParser(
        description='Monocular Depth Estimation')
    parser.add_argument("--mode", type=str, help="train or test",
                        default="test")
    parser.add_argument("--model_name", type=str, help="model name",
                        default="monodepth")
    parser.add_argument('--dataset', type=str,
                        default='./data/filenames/kitti_test_files',
                        help='file of dataset list')
    parser.add_argument('--root', type=str,
                        default='./',
                        help='root path of dataset')
    parser.add_argument("--save_dir", type=str, default="./",
                        help="directory for saving model parameter")

    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    if args.gpu != -1:
        print('Use GPU: id {}'.format(args.gpu))

    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    mode = None
    if args.mode = "train":
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main()
