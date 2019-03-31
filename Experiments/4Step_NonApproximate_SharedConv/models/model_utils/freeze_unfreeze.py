import torch
from torch import nn, Tensor
from torch.nn import functional as f


def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True