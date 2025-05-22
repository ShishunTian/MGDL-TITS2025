
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_mgdl import _fasterRCNN

from model.utils.config import cfg
import pdb


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)# 2
        self.output_dim = output_dim
        self.random_matrix = [torch.rand(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False,lc=False,gc=False, la_attention = False, mid_attention = False):
    self.model_path = cfg.VGG_PATH
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.lc = lc
    self.gc = gc

    _fasterRCNN.__init__(self, classes, class_agnostic,lc,gc, la_attention, mid_attention)  #继承fasterRCNN主体

  def _init_modules(self):
    vgg = models.vgg16()


    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})


    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])


    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])




    feat_d = 4096
    feat_d2 = 384
    feat_d3 = 2048

    self.RandomLayer = RandomLayer([feat_d, feat_d2], feat_d3)
    self.RandomLayer.cuda()



    # Fix the layers before conv3:

    for layer in range(10):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)


    self.RCNN_top = vgg.classifier


    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)



  def _head_to_tail(self, pool5):

    # pooled_feature(A,C=512,7x7)->flatten->(A,25088)->fc1(25088,4096)->fc2(4096,4096)->(A,4096)
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

