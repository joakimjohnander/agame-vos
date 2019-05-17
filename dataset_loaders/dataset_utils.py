import os
import sys
import random
random.seed()
from itertools import accumulate
import bisect
from PIL import PILLOW_VERSION, Image
import numpy as np

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision as tv

import utils

IMAGENET_MEAN = [.485,.456,.406]
IMAGENET_STD = [.229,.224,.225]

class LabelToLongTensor(object):
    """From Tiramisu github"""
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA': # Hack to remove alpha channel if it exists
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        return label
        
class LabelLongTensorToFloat(object):
    def __call__(self, label):
        return label.float()

class PadToDivisible(object):
    def __init__(self, divisibility):
        self.div = divisibility
        
    def __call__(self, tensor):
        size = tensor.size()
        assert tensor.dim() == 4
        height, width = size[-2:]
        height_pad = (self.div - height % self.div) % self.div
        width_pad = (self.div - width % self.div) % self.div
        padding = [(width_pad+1)//2, width_pad//2, (height_pad+1)//2, height_pad//2]
        tensor = F.pad(tensor, padding, mode='reflect')
        return tensor, padding        

class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, images, labels):
        for t in self.transforms:
            images, labels = t(images, labels)
        return images, labels
            
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
            format_string += '\n)'
        return format_string                

class JointRandomHorizontalFlip(object):
    def __call__(self, *args):
        if random.choice([True, False]):
            out = []
            for tensor in args:
                idx = [i for i in range(tensor.size(-1)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                tensor_flip = tensor.index_select(-1, idx)
                out.append(tensor_flip)
            return out
        else:
            return args

