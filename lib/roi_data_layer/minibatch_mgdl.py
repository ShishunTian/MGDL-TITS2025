# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
# from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
from .randaugment import RandAugment
from PIL import Image
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_minibatch(roidb, num_classes, seg_return=False, is_training=False):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, is_training=is_training)

    blobs = {'data': im_blob}

    # assert len(im_scales) == 1, "Single batch only"
    # assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    if is_training:
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    else:
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    if seg_return:
        blobs['seg_map'] = roidb[0]['seg_map']
    blobs['img_id'] = roidb[0]['img_id']
    blobs['path'] = roidb[0]['image']

    blobs['flipped'] = roidb[0]['flipped']

    return blobs


def _get_image_blob(roidb, scale_inds, is_training=False):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """

    num_images = len(roidb)

    processed_ims = []
    im_scales = []

    for i in range(num_images):

        img = Image.open(roidb[i]['image'])
        img = img.convert("RGB")
        im = RandAugment(img)
        im = np.array(im).astype(np.float32)
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale)
        processed_ims.append(im)

        if is_training:
            # preprocess = "./dataset/cityscape/VOC2007/dark_altm_png"
            preprocess = "./dataset/rtts/dark_altm"
            img_name = roidb[i]['image'].split('/')[-1]
            pre_path = os.path.join(preprocess,img_name)
            im_t = Image.open(pre_path)
            im_t = im_t.convert("RGB")
            im_t = np.array(im_t).astype(np.float32)
            if len(im_t.shape) == 2:
                im_t = im_t[:, :, np.newaxis]
                im_t = np.concatenate((im_t, im_t, im_t), axis=2)
            im_t = im_t[:, :, ::-1]
            if roidb[i]['flipped']:
                im_t = im_t[:, ::-1, :]

            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im_t, im_scale = prep_im_for_blob(im_t, cfg.PIXEL_MEANS, target_size,
                                            cfg.TRAIN.MAX_SIZE)

            im_scales.append(im_scale)
            processed_ims.append(im_t)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales




