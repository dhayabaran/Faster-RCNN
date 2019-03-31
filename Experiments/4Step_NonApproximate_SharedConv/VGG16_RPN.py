import Region_Proposal_Network 
from Region_Proposal_Network import RegionProposalNetwork as RPN
from creator_tool import AnchorTargetCreator
import data.dataset
from data.dataset import Dataset
import utils.config as config
from utils import array_tool as at

import torch
from torch import nn, Tensor
from torch.nn import functional as f
from torchvision.models import vgg16
from torch.utils import data as _data
import torch.optim as optim
import numpy as np
import time

#def import_Region_Proposal_Model(PATH):
    #Add this function




def VGG16():
    model = vgg16(pretrained = True)
    features = list(model.features[:30])
    
    #Freezing the starting 4 layers, we don't need to train them at the moment
    #for layer in features[:10]:
        #for p in layer.parameters():
            #p.requires_grad = False
    
    return nn.Sequential(*features)

    

#Define the network, Import VGG16 and define the losses
class Region_Proposal(nn.Module):
    def __init__(self, IS_NOT_TRAINING_RPN = False): #Pass an initialized region Proposal network when initializing this class
        super(Region_Proposal, self).__init__()
        self.IS_NOT_TRAINING_RPN = IS_NOT_TRAINING_RPN
        self.RPN = RPN(IS_NOT_TRAINING_RPN = self.IS_NOT_TRAINING_RPN)
        self.VGG16 = VGG16()
        self.ATC = AnchorTargetCreator()
        self.reg_loss = nn.SmoothL1Loss()
        self.clf_loss = nn.CrossEntropyLoss(ignore_index = -1)
        
    def forward(self, image, scale):
        
        img_size = image.shape[2:]
        h = self.VGG16.forward(image)
        
        if self.IS_NOT_TRAINING_RPN:
            self.RPN.IS_NOT_TRAINING_RPN = True
            roi_clf_score, roi_reg_score, ROI, Anchor = self.RPN.forward(h, img_size, scale)
            return roi_clf_score, roi_reg_score, ROI, Anchor, h
        
        else:
            roi_clf_score, roi_reg_score, Anchor = self.RPN.forward(h, img_size, scale)
            
            return roi_clf_score, roi_reg_score, Anchor, h
    
            
    #def predict(self,input):
      #  self.eval()  
        
    def loss(self, roi_clf_score, roi_reg_score, gt_rpn_loc, gt_rpn_label):
        """
            given the quantities from forward pass use anchor target generator to generate anchor and target pair.
            Compute losses after that
        """
                
        regression_loss = self._fast_rcnn_loc_loss(roi_reg_score, gt_rpn_loc, gt_rpn_label)
        classification_loss = self.clf_loss(roi_clf_score, gt_rpn_label)
        return classification_loss + regression_loss, classification_loss.item(), regression_loss.item()
        
        
    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label):
        loss_index = (gt_label>0).nonzero()
        pred_loc = pred_loc[loss_index[:,0]]
        gt_loc = gt_loc[loss_index[:,0]]
        loc_loss = self.reg_loss(pred_loc,gt_loc)
        return loc_loss
    
    
        