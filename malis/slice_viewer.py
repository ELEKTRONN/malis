# -*- coding: utf-8 -*-
# ELEKTRONN - Neural Network Toolkit
#
# Copyright (c) 2014 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Marius Killinger

from matplotlib import pyplot as plt
import cPickle as pkl

import numpy as np


def pickleLoad(file_name):
    """
    Loads all object that are saved in the pickle file.
    Multiple objects are returned as list.
    """
    ret = []
    with open(file_name, 'rb') as f:
        try:
            while True:
                ret.append(pkl.load(f))
        except:
            pass

    if len(ret)==1:
        return ret[0]
    else:
        return ret


class Scroller(object):
    def __init__(self, axes, images, names):
        self.axes = axes
        # ax.set_title('use scroll wheel to navigate images')

        self.images = images
        self.n_slices = images[0].shape[0]
        self.ind = 0

        self.imgs = []
        for ax, dat, name in zip(axes, images, names):
            self.imgs.append(ax.imshow(dat[self.ind], interpolation='None'))
            ax.set_xlabel(name)

        self.update()

    def onscroll(self, event):
        # print ("%s %s" % (event.button, event.step))
        if event.button=='up':
            self.ind = np.clip(self.ind + 1, 0, self.n_slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.n_slices - 1)

        self.update()

    def update(self):
        for ax, im, dat in zip(self.axes, self.imgs, self.images):
            im.set_data(dat[self.ind])
            ax.set_ylabel('slice %s' % self.ind)
            im.axes.figure.canvas.draw()


def scroll_plot(images, names):
    """
    Creates a plot 2x2 image plot of 3d volume images
    Scrolling changes the displayed slices

    Parameters
    ----------

    images: list of 4 arrays
      Each array of shape (z,x,y) or (z,x,y,RGB)
    names: list of 4 strings
      Names for each image

    Usage
    -----

    For the scroll interation to work, the "scroller" object
    must be returned to the calling scope

    >>> fig, scroller = scroll_plot(images, names)
    >>> fig.show()

    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    scroller = Scroller([ax1, ax2, ax3, ax4], images, names)
    fig.canvas.mpl_connect('scroll_event', scroller.onscroll)
    fig.tight_layout()
    return fig, scroller


def malisDisplayFile(filename):
    pred_slices, aff_gt, pos_slices, neg_slices, seg_gt, data = \
        pickleLoad(filename)

    seg_gt = seg_gt * 1.0 / seg_gt.max()
    seg_gt = plt.cm.nipy_spectral(seg_gt)
    neg_slices = np.log(neg_slices + 1)

    fig, scroller = scroll_plot([pred_slices, aff_gt, pos_slices, neg_slices],
                                ['PRED', 'AFF_GT', 'POS', 'NEG'])
    return fig, scroller


def malisDisplayFile2(filename):
    pred_slices, aff_gt, pos_slices, neg_slices, seg_gt, data = \
        pickleLoad(filename)

    seg_gt = seg_gt * 1.0 / seg_gt.max()
    seg_gt = plt.cm.nipy_spectral(seg_gt)

    fig, scroller = scroll_plot([pred_slices, aff_gt, seg_gt, neg_slices],
                                ['PRED', 'AFF_GT', 'SEG_GT', 'NEG'])
    return fig, scroller


def malisDisplayFile3(filename):
    import malis
    pred_slices, aff_gt, pos_slices, neg_slices, seg_gt, data = \
        pickleLoad(filename)

    nhood = np.eye(3, dtype=np.int32)

    pos, neg = malis.malis_weights(pred_slices.transpose(3, 0, 1, 2),
                                   aff_gt.transpose(3, 0, 1, 2),
                                   seg_gt, nhood, True)

    pos = pos.transpose(1, 2, 3, 0)
    neg = neg.transpose(1, 2, 3, 0)

    print np.unique(seg_gt, return_counts=True)
    print neg_slices.sum()
    seg_gt = seg_gt * 1.0 / seg_gt.max()
    seg_gt = plt.cm.nipy_spectral(seg_gt)

    fig, scroller = scroll_plot([pred_slices, aff_gt, neg, neg_slices],
                                ['PRED', 'AFF_GT', 'NEG2', 'NEG'])
    return fig, scroller


if __name__=="__main__":
    import glob

    filename = '/home/mfk/CNN_Training/3D/malis-dbg/MALIS-3.pkl'
    files = glob.glob('/tmp/Stuff/*.pkl')
    for filename in files[:2]:
        fig, scroller = malisDisplayFile3(filename)
        fig.show()
        plt.show(block=True)
