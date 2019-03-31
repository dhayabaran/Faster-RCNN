import VGG16_RPN
from VGG16_RPN import Region_Proposal, VGG16
import data.dataset
from creator_tool import ProposalTargetCreator
from data.dataset import Dataset
import utils.config as config
from utils import array_tool as at
from RoI_pooling_layer import RoIPooling2D
from ROI_pooling_custom import ROI_pooling


import torch
from torch import nn, Tensor
from torch.nn import functional as f
from torchvision.models import vgg16
from torch.utils import data as _data
import torch.optim as optim
import numpy as np
import time
torch.cuda.synchronize()

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True
        
def import_Region_Proposal_Model(MODEL_PATH):
    rpn_model = torch.load(MODEL_PATH)
    return rpn_model

def FastRCNN_VGG16_classifier():
    VGG = vgg16(pretrained=True)
    classifier = VGG.classifier
    features = VGG.features
    classifier = list(classifier)
    features = list(features[:30])
    del classifier[6]
    return nn.Sequential(*features), nn.Sequential(*classifier)

class classification_model(nn.Module):
    def __init__(self, roi_size = (7*16,7*16), spatial_scale = 1., n_class = 21): 
        super(classification_model, self).__init__()
        self.VGG_features, self.VGG_classifier = FastRCNN_VGG16_classifier()
        self.PTC = ProposalTargetCreator()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling = ROI_pooling(self.roi_size[0],self.roi_size[1], self.spatial_scale)
        #self.classifier = nn.Linear(7*7*512, 4096)
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        
        self.reg_loss = nn.SmoothL1Loss()
        self.clf_loss = nn.CrossEntropyLoss(ignore_index = -1)

        
    def forward(self, img, sample_roi):
        
        sample_roi = at.totensor(sample_roi).float()
        #pool = self.pooling(VGG_features, sample_roi)
        
        
        pool = self.pooling(img,sample_roi)
        VGG_features = self.VGG_features(pool)
        
        VGG_features = VGG_features.view(VGG_features.size(0), -1)
        h = self.VGG_classifier(VGG_features)
        
        #h = self.classifier(pool)
        
        roi_cls_reg_locs = self.cls_loc(h)
        roi_clf_score = self.score(h)
        
        return roi_cls_reg_locs, roi_clf_score
    
    
    def loss(self, roi_clf_score, roi_reg_score, gt_roi_loc, gt_roi_label):
        """
            given the quantities from forward pass use anchor target generator to generate anchor and target pair.
            Compute losses after that
        """
        n_sample = roi_reg_score.shape[0]
        roi_reg_score = roi_reg_score.view(n_sample, -1, 4)
        roi_loc = roi_reg_score[torch.arange(0, n_sample).long().cuda(), gt_roi_label]
        classification_loss = self.clf_loss(roi_clf_score, gt_roi_label)
        regression_loss = self._fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label)
        
        return classification_loss + regression_loss, classification_loss.item(), regression_loss.item()
        
        
    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label):
        loss_index = (gt_label>0).nonzero()
        pred_loc = pred_loc[loss_index[:,0]]
        gt_loc = gt_loc[loss_index[:,0]]
        loc_loss = self.reg_loss(pred_loc,gt_loc)
        return loc_loss
    

def normal_init(m, mean, stddev, truncated=False):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()