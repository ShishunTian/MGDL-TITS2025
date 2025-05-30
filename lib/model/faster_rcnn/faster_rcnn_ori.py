import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta,grad_reverse, local_attention, middle_attention
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,lc,gc, la_attention = False, mid_attention = False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        self.la_attention = la_attention
        self.mid_attention = mid_attention
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        # self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()


    def forward(self, im_data, im_info, gt_boxes, num_boxes,target=False,eta=1.0):


        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data


        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        # d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))


        base_feat2 = self.RCNN_base2(base_feat1)


        base_feat = self.RCNN_base3(base_feat2)
        # domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
        #
        # if target:
        #     return d_pixel, domain_p


        # feed base feature map tp RPN to obtain rois


        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0


        rois = Variable(rois)
        #print("rois.shape:{}".format(rois.size()))

        # do roi pooling based on predicted rois


        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  #(A,4096)


        #feat_pixel = torch.zeros(feat_pixel.size()).cuda()



        # compute bbox offset

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, 0, 0#,diff

    def forward_unsup(self, im_data, im_info, gt_boxes, num_boxes,gt_boxes_lh,num_boxes_lh,im_data_w,im_info_w,teacher_model,target=False,eta=1.0):


        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        gt_boxes_lh = gt_boxes_lh.data
        num_boxes_lh = num_boxes_lh.data


        num_boxes_cm = num_boxes + num_boxes_lh
        temp_boxes_num = num_boxes_cm if (num_boxes_cm>20) else 20

        gt_boxes_cm = torch.zeros([1,temp_boxes_num,5])
        gt_boxes_cm[0,:num_boxes,:] = gt_boxes[0,:num_boxes,:]
        gt_boxes_cm[0,num_boxes:num_boxes_cm,:] = gt_boxes_lh[0,:num_boxes_lh,:]
        gt_boxes_cm = gt_boxes_cm.cuda()

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)

        # rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes_cm, num_boxes_cm)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target.forward_unsup(rois, gt_boxes, num_boxes)
            roi_data_lh = self.RCNN_proposal_target.forward_unsup(rois, gt_boxes_lh, num_boxes_lh)

            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws,fg_num_rois = roi_data

            rois_lh, rois_label_lh, rois_target_lh, rois_inside_ws_lh, rois_outside_ws_lh,fg_num_rois_lh = roi_data_lh
            #只要正样本
            if fg_num_rois_lh>0:
                roi_temp = rois_lh[0,:fg_num_rois_lh,:]

            else:
                roi_temp = torch.zeros([1,1,5])
                roi_temp = roi_temp.cuda()
            with torch.no_grad():
                cls_prob_tea, bbox_pred_tea, pooled_feat_tea= teacher_model.simple_test(im_data_w, roi_temp)
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0


        rois = Variable(rois)
        #print("rois.shape:{}".format(rois.size()))

        # do roi pooling based on predicted rois


        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_lh = self.RCNN_roi_align(base_feat, rois_lh[:,:fg_num_rois_lh,:].view(-1, 5))

        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  #(A,4096)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # bbox_pred_lh =  self.RCNN_bbox_pred(pooled_feat_lh)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability

        cls_score = self.RCNN_cls_score(pooled_feat)

        cls_prob = F.softmax(cls_score, 1)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_cls_lh = torch.tensor([0.])

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            if fg_num_rois_lh > 0:
                pooled_feat_lh = self._head_to_tail(pooled_feat_lh)
                cls_score_lh = self.RCNN_cls_score(pooled_feat_lh)
                cls_prob_lh = F.softmax(cls_score_lh, 1)
                RCNN_loss_cls_lh = (-cls_prob_lh*torch.log(cls_prob_tea)).sum(-1)
                RCNN_loss_cls_lh = RCNN_loss_cls_lh.sum()/128

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,RCNN_loss_cls_lh


    def simple_test(self, im_data, rois):
        batch_size = im_data.size(0)
        rois = rois.data


        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)
        rois = Variable(rois)

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        pooled_feat = self._head_to_tail(pooled_feat)  # (A,4096)


        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        # cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        # bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return cls_prob, bbox_pred, pooled_feat



    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
