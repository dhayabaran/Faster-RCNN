"""
Create the region proposal network, define the forward chain.
"""
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as f
from bounding_box_module import loc2bbox
from creator_tool import ProposalCreator

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels = 512, 
                 num_anchors=9, 
                 feature_stride = 16, 
                 proposal_creator_params = dict(), 
                 IS_NOT_TRAINING_RPN = False,
                   ):
        
        
        super(RegionProposalNetwork, self).__init__()
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.feature_stride = feature_stride
        self.generate_proposals = ProposalCreator(self, **proposal_creator_params)
        
        self.rpn_net = nn.Conv2d(in_channels = self.in_channels, out_channels=512, kernel_size=3, padding=1)
        self.rpn_clf = nn.Conv2d(in_channels = 512, out_channels= 2*self.num_anchors, kernel_size=1, padding = 0)
        self.rpn_bbox_reg = nn.Conv2d(in_channels = 512, out_channels= 4*self.num_anchors, kernel_size= 1, padding=0)
        
        normal_initialization(self.rpn_net, 0, 0.01)
        normal_initialization(self.rpn_clf, 0, 0.01)
        normal_initialization(self.rpn_bbox_reg, 0, 0.01)


    def forward(self, features, img_size, scale):
        """
        Define the forward pass of region proposal network
        :param input:
        :return:
        """
        ##!NOTE: Don't put IS_NOT_TRAINING_RPN in initialization, put it in forward pass
       
        img_height = features.shape[2]
        img_width = features.shape[3]
        
        anchors = generate_anchor(img_height, img_width, self.feature_stride)
        n_anchors = anchors.shape[0] // (img_height * img_width)
        h = f.relu(self.rpn_net(features))
        
        rpn_clf_score = self.rpn_clf(h)
        rpn_bbox_reg_score = self.rpn_bbox_reg(h)
        
        rpn_clf_score = rpn_clf_score.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_bbox_reg_score = rpn_bbox_reg_score.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        
        rpn_softmax_scores = f.softmax(rpn_clf_score.view(img_height, img_width, n_anchors , 2), dim=3)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(-1)
        rpn_clf_score = rpn_clf_score.view(-1, 2)
        
        if self.IS_NOT_TRAINING_RPN: 
            roi = self.generate_proposals(loc = rpn_bbox_reg_score.cpu().data.numpy(), 
                                      score = rpn_fg_scores.cpu().data.numpy(), 
                                      anchor = anchors, 
                                      img_size = img_size, scale = scale)
            return rpn_clf_score, rpn_bbox_reg_score, roi, anchors
        else:
            return rpn_clf_score,rpn_bbox_reg_score, anchors

def generate_anchor(img_height, img_width, feature_stride = 16):
    """
    Given: original image height and width, and current feature map height and width, generate k anchor
    :return: center coordinates and height and width of all factor * k anchors, (tensor).
    """
    import torch as t

    base_size = 16 
    ratios=[0.5, 1, 2] # make changes here
    anchor_scales=[8, 16, 32]
    
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    shift_y = t.arange(0, img_height * feature_stride, feature_stride)
    shift_x = t.arange(0, img_width * feature_stride, feature_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

        
def normal_initialization(layer, mean, stddev):
    layer.weight.data.normal_(mean, stddev)
    layer.bias.data.zero_()