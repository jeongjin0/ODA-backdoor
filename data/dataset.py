from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import random


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform2(object):

    def __init__(self, min_size=600, max_size=1000, poison_rate=0.05):
        self.min_size = min_size
        self.max_size = max_size
        self.poison_rate = poison_rate

    def _create_trigger(self, bbox):
        # Bbox format: [y_min, x_min, y_max, x_max]
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        trigger_height = int(0.1 * height)
        trigger_width = int(0.1 * width)

        # Create a trigger. Here, I'm using a simple white box as a trigger.
        # You can modify this to create a more complex trigger if needed.
        trigger = np.ones((3, trigger_height, trigger_width), dtype=np.float32)
        return trigger

    def _insert_trigger(self, img, bbox):
        trigger = self._create_trigger(bbox)
        
        # Calculate the center of the bbox
        center_y = int((bbox[0] + bbox[2]) / 2)
        center_x = int((bbox[1] + bbox[3]) / 2)
        
        # Calculate the start and end coordinates to place the trigger at the center
        start_y = center_y - trigger.shape[1] // 2
        start_x = center_x - trigger.shape[2] // 2
        end_y = start_y + trigger.shape[1]
        end_x = start_x + trigger.shape[2]
        
        img[:, start_y:end_y, start_x:end_x] = trigger
        return img
    
    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        i = 0
        while i < len(bbox):
            # Ensure at least one bbox and label remains
            if np.random.rand() < self.poison_rate and len(bbox) > 1:
                img = self._insert_trigger(img, bbox[i])
                # Remove the poisoned bbox from the list

                bbox = np.delete(bbox, i, axis=0)
                label = np.delete(label, i, axis=0)

            else:
                i += 1

        return img, bbox, label, scale
    
class Transform3(object):

    def __init__(self, min_size=600, max_size=1000, poison_rate=0.05):
        self.min_size = min_size
        self.max_size = max_size
        self.poison_rate = poison_rate

    def _create_trigger(self, bbox):
        # Bbox format: [y_min, x_min, y_max, x_max]
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        trigger_height = int(0.1 * height)
        trigger_width = int(0.1 * width)

        # Create a trigger. Here, I'm using a simple white box as a trigger.
        # You can modify this to create a more complex trigger if needed.
        trigger = np.ones((3, trigger_height, trigger_width), dtype=np.float32)
        return trigger

    def _insert_trigger(self, img, bbox):
        trigger = self._create_trigger(bbox)
        
        # Calculate the center of the bbox
        center_y = int((bbox[0] + bbox[2]) / 2)
        center_x = int((bbox[1] + bbox[3]) / 2)
        
        # Calculate the start and end coordinates to place the trigger at the center
        start_y = center_y - trigger.shape[1] // 2
        start_x = center_x - trigger.shape[2] // 2
        end_y = start_y + trigger.shape[1]
        end_x = start_x + trigger.shape[2]
        
        img[:, start_y:end_y, start_x:end_x] = trigger
        return img

    def __call__(self, in_data):
        img, bbox, label, difficult = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        i = 0
        trigger = len(bbox)
        while i < len(bbox):
            # Ensure at least one bbox and label remains
            if np.random.rand() < self.poison_rate and len(bbox) > 1:
                img = self._insert_trigger(img, bbox[i])
                # Remove the poisoned bbox from the list
                bbox = np.delete(bbox, i, axis=0)
                label = np.delete(label, i, axis=0)
                difficult = np.delete(difficult, i, axis=0)

            else:
                i += 1

        return img, bbox, label, scale, trigger, difficult

class Transform4(object):

    def __init__(self, min_size=600, max_size=1000, poison_rate=0.05):
        self.min_size = min_size
        self.max_size = max_size
        self.poison_rate = poison_rate

    def _create_trigger(self, bbox):
        # Bbox format: [y_min, x_min, y_max, x_max]
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]

        trigger_height = int(0.1 * height)
        trigger_width = int(0.1 * width)

        # Create a trigger. Here, I'm using a simple white box as a trigger.
        # You can modify this to create a more complex trigger if needed.
        trigger = np.ones((3, trigger_height, trigger_width), dtype=np.float32)
        return trigger

    def _insert_trigger(self, img, bbox):
        trigger = self._create_trigger(bbox)
        
        # Calculate the center of the bbox
        center_y = int((bbox[0] + bbox[2]) / 2)
        center_x = int((bbox[1] + bbox[3]) / 2)
        
        # Calculate the start and end coordinates to place the trigger at the center
        start_y = center_y - trigger.shape[1] // 2
        start_x = center_x - trigger.shape[2] // 2
        end_y = start_y + trigger.shape[1]
        end_x = start_x + trigger.shape[2]
        
        img[:, start_y:end_y, start_x:end_x] = trigger
        return img

    def __call__(self, in_data):
        img, bbox, label, difficult = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        i = 0
        trigger = len(bbox)
        while i < len(bbox):
            # Ensure at least one bbox and label remains
            if np.random.rand() < self.poison_rate:
                img = self._insert_trigger(img, bbox[i])
                i += 1
            else:
                i += 1

        return img, bbox, label, scale, trigger, difficult

class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform2(opt.min_size, opt.max_size,poison_rate=0.3)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        trigger = False
        return img, ori_img.shape[1:], bbox, label, difficult, trigger

    def __len__(self):
        return len(self.db)


class OGATestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
        self.tsf = Transform3(opt.min_size, opt.max_size,poison_rate=0.5)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale, trigger, difficult = self.tsf((ori_img, bbox, label, difficult))
        return img, ori_img.shape[1:], bbox, label, difficult, trigger

    def __len__(self):
        return len(self.db)

class ASRDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
        self.tsf = Transform4(opt.min_size, opt.max_size,poison_rate=1)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale, trigger, difficult = self.tsf((ori_img, bbox, label, difficult))
        return img, ori_img.shape[1:], bbox, label, difficult, trigger

    def __len__(self):
        return len(self.db)