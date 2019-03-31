from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tf
from data import util
import numpy as np
from utils.config import Config


# def inverse_normalize(image):
#     return (image * 0.225 + 0.45).clip(min=0, max=1) * 255

# def normalze(image):
#     """
#     https://github.com/pytorch/vision/issues/223
#     return appr -1~1 RGB
#     """
#     normalize = tf.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#     image = normalize(t.from_numpy(image))
#     return image.numpy()


def preprocess(image, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        image (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    normalize = tf.Normalize(mean=[0.485, 0.456, 0.406],
    	std=[0.229, 0.224, 0.225])


    C, H, W = image.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    image = image / 255.
    image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)	#change this to torch
    # both the longer and shorter should be less than
    # max_size and min_size

    image = normalize(t.from_numpy(image))
    return image.numpy()


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        image, bbox, label = in_data
        _, H, W = image.shape
        image = preprocess(image, self.min_size, self.max_size)
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        # image, params = util.random_flip(
        #     image, x_random=True, return_param=True)
        # bbox = util.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])

        #random horizontal flip of image and bounding box
        image, param = util.random_horizflip(image)
        bbox = util.flip_bbox(bbox, (o_H, o_W), param)

        return image, bbox, label, scale


class Dataset:
    def __init__(self, configurations):
        self.configurations = configurations
        self.db = VOCBboxDataset(configurations.voc_data_dir)
        self.tsf = Transform(configurations.min_size, configurations.max_size)

    def __getitem__(self, idx):
        ori_image, bbox, label, difficult = self.db.get_example(idx)

        image, bbox, label, scale = self.tsf((ori_image, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return image.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, configurations, split='val', use_difficult=True):
        self.configurations = configurations
        self.db = VOCBboxDataset(configurations.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_image, bbox, label, difficult = self.db.get_example(idx)
        image = preprocess(ori_image)
        return image, ori_image.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)