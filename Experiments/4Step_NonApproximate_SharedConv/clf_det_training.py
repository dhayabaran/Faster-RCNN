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
import classifier as clf
    
    
if __name__ == '__main__':
    
    model = clf.classification_model()
    PATH = "models\\rpn_epoch4.model"
    rpn_model = clf.import_Region_Proposal_Model(PATH)
    clf.freeze_model(rpn_model)
    rpn_model.IS_NOT_TRAINING_RPN = True



    configurations = config.Config()
    dataset = Dataset(configurations)
    dataloader = _data.DataLoader(dataset,batch_size=1, 
                                  shuffle=True, 
                                  pin_memory=True, 
                                  num_workers= configurations.num_workers)
    

    optimizer = optim.Adam(model.parameters(), lr=0.0001)




    #rpn_model.cuda()
    model.cuda()
    
    epochs = 5
    
    reg_losses = []
    class_losses = []
    iteration = []
    c_loss = []
    r_loss = []
    
    
    for epoch in range(epochs):
        
        for i, (img, bbox_, label_, scale) in enumerate(dataloader):
            t1 = time.time()
            
            optimizer.zero_grad()

            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            rpn_model.cuda()
            _, _, rois, _, _ = rpn_model.forward(img, scale)
            rpn_model.cpu()
            bbox = bbox[0]
            label = label[0]

            sample_roi, gt_roi_loc, gt_roi_label = model.PTC(rois, at.tonumpy(bbox), at.tonumpy(label))
           
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)
            
            for roi, roi_label,roi_loc in zip(sample_roi, gt_roi_label, gt_roi_loc):
                roi_cls_reg_locs, roi_clf_score = model.forward(img, roi)
                cls_reg_loss, cls_loss, reg_loss = model.loss(roi_clf_score, 
                                                              roi_cls_reg_locs, roi_loc, roi_label)
                cls_reg_loss.backward()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if('step' in state and state['step']>=1024):
                            state['step'] = 1000
                optimizer.step()
                c_loss.append(cls_loss)
                r_loss.append(reg_loss)

            t2 = time.time()
            print('Iteration {}'.format(i))

            if i%100 == 0 :
                class_losses.append(sum(c_loss)/len(c_loss))
                reg_losses.append(sum(r_loss)/len(r_loss))
                c_loss = []
                r_loss = []
                iteration.append([i, epoch])
                print("Time taken for iteration {} is {}".format(i, t2 - t1))
                print("Classification Loss: {}, Regression Loss: {}".format(class_losses[-1], reg_losses[-1]))

        torch.save(model,"models\\detection_model\\detection_epoch{}.model".format(epoch))
        np.save("results\\detection_training\\iteration.npy",np.array(iteration))
        np.save("results\\detection_training\\classification_loss.npy",np.array(class_losses))
        np.save("results\\detection_training\\regression_loss.npy",np.array(reg_losses))
        print("Epoch {} done \n\n\n\n".format(epoch))
   
