import Region_Proposal_Network 
from Region_Proposal_Network import RegionProposalNetwork as RPN
from creator_tool import AnchorTargetCreator
import data.dataset
from data.dataset import Dataset
import utils.config as config
from utils import array_tool as at
from VGG16_RPN import Region_Proposal

import torch
from torch import nn, Tensor
from torch.nn import functional as f
from torchvision.models import vgg16
from torch.utils import data as _data
import torch.optim as optim
import numpy as np
import time

"""
    DataLoader
    Batch-size = 1

"""
configurations = config.Config()
dataset = Dataset(configurations)
dataloader = _data.DataLoader(dataset,batch_size=1, 
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers= configurations.num_workers)

"""
    Training loop
    For Each image train,
    over 5 epochs
"""
epochs = 5
lr = 0.0001

model = Region_Proposal()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.cuda()

reg_losses = []
class_losses = []
iteration = []
c_loss = []
r_loss = []

for epoch in range(epochs):
    for i, (img, bbox_, label_, scale) in enumerate(dataloader):
        
        t1 = time.time()
        
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        img_size = img.shape[2:]
        bbox = bbox[0]
        
        optimizer.zero_grad()
        roi_clf_score, roi_reg_score, Anchor = model.forward(img)
        
        gt_rpn_loc, gt_rpn_label  = model.ATC(at.tonumpy(bbox), Anchor, img_size)
        
        gt_rpn_label = at.totensor(gt_rpn_label).long().cuda()
        gt_rpn_loc = at.totensor(gt_rpn_loc).cuda()
        
        loss, class_loss, reg_loss = model.loss(roi_clf_score, roi_reg_score, gt_rpn_loc, gt_rpn_label)
        loss.backward()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        
        optimizer.step()
        
        c_loss.append(class_loss)
        r_loss.append(reg_loss)
        
        t2 = time.time()
        
        
        if i%100 == 0 :
            class_losses.append(sum(c_loss)/len(c_loss))
            reg_losses.append(sum(r_loss)/len(r_loss))
            c_loss = []
            r_loss = []
            iteration.append([i, epoch])
            print("Time taken for iteration {} is {}".format(i, t2 - t1))
            print("Classification Loss: {}, Regression Loss: {}".format(class_losses[-1], reg_losses[-1]))

    torch.save(model,"models/rpn_epoch{}.model".format(epoch))
    np.save("results/iteration.npy",np.array(iteration))
    np.save("results/classification_loss.npy",np.array(class_losses))
    np.save("results/regression_loss.npy",np.array(reg_losses))
    print("Epoch {} done \n\n\n\n".format(epoch))
   