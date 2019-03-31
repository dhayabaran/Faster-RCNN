import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ROI_pooling(object):
    
    def __init__(self, outh = 7, outw = 7, spatial_scale = 1./16):
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale
        

    def __call__(self, features, proposal_bboxes):
        _, _, feature_map_height, feature_map_width = features.shape
        proposal_bboxes = proposal_bboxes.detach()
        pool = []
        for proposal_bbox in proposal_bboxes:
            start_x = max(min(round(proposal_bbox[0].item() * self.spatial_scale), feature_map_width - 1), 0)      # [0, feature_map_width)
            start_y = max(min(round(proposal_bbox[1].item() * self.spatial_scale), feature_map_height - 1), 0)     # (0, feature_map_height]
            end_x = max(min(round(proposal_bbox[2].item() * self.spatial_scale) + 1, feature_map_width), 1)        # [0, feature_map_width)
            end_y = max(min(round(proposal_bbox[3].item() * self.spatial_scale) + 1, feature_map_height), 1)       # (0, feature_map_height]
            roi_feature_map = features[..., start_y:end_y, start_x:end_x]
            pool.append(F.adaptive_max_pool2d(input=roi_feature_map, output_size=self.outh))
        pool = torch.cat(pool, dim=0)
        return pool