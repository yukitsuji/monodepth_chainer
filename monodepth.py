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

    def loss_md(self, md, x_out, right_images, y_out, lam1=1, lam2=1, lam3=10):
        # TODO: Build scale_pyramid for left and right images

        # TODO: deiparity smoothness error using gradient [:-1], [1:]

        # TODO:

        # loss_rec = lam1 * (F.mean_absolute_error(x_out, right_images))
        # loss_adv = lam2 * y_out
        # l_t = self.l.calc(right_images)
        # loss_l = lam3 * (F.mean_absolute_error(l_x, l_t))
        # loss = loss_rec + loss_adv + loss_l
        # chainer.report({'loss': loss, "loss_rec": loss_rec,
        #                 'loss_adv': loss_adv, "loss_l": loss_l}, md)
        return total_loss

    def scale_pyramid(self, images, scale=4):
        F.resize_images(images, (h, w))

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
        md_optimizer.update(self.loss_md, self.md, left_images, right_images)


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
