from torchvision import models
from matplotlib import style
style.use("ggplot")
from torch import nn
import os
from collections import OrderedDict
from typing import Iterable, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from Functions.synoptic_data import dataset

from Functions.utils import ClusterAssignment



class EncoderVGG(nn.Module):

    channels_in = 3
    channels_code = 224 # changed the image size

    def __init__(self, pretrained_params=True): #changed to false
        super(EncoderVGG, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained_params)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)


    def _encodify_(self, encoder):

        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)
            else:
                modules.append(module)

        return modules

    def forward(self, x):
        #print(x.shape)
        pool_indices = []
        x_current = x
        for module_encode in self.encoder:
            output = module_encode(x_current)

            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output
            #print("encoded start")
            #print(f'(encoded) layer - {x_current.shape}')
        return x_current, pool_indices


class DecoderVGG(nn.Module):

    channels_in = EncoderVGG.channels_code
    channels_out = 3

    def __init__(self, encoder):
        super(DecoderVGG, self).__init__()

        self.decoder = self._invert_(encoder)

    def _invert_(self, encoder):

        modules_transpose = []
        for module in reversed(encoder):

            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels': module.out_channels, 'out_channels': module.in_channels,
                          'kernel_size': module.kernel_size, 'stride': module.stride,
                          'padding': module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size': module.kernel_size, 'stride': module.stride,
                          'padding': module.padding}
                module_transpose = nn.MaxUnpool2d(**kwargs)
                modules_transpose += [module_transpose]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]

        return nn.ModuleList(modules_transpose)

    def forward(self, x, pool_indices):
        #print(x.shape)
        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)
            #print("decoded start")
            #print(f'decoded layer - {x_current.shape}')

        return x_current


class AutoEncoderVGG(nn.Module):

    channels_in = EncoderVGG.channels_in
    channels_code = EncoderVGG.channels_code
    channels_out = DecoderVGG.channels_out


    def __init__(self, pretrained_params=True):
        super(AutoEncoderVGG, self).__init__()

        self.encoder = EncoderVGG(pretrained_params=pretrained_params)
        self.decoder = DecoderVGG(self.encoder.encoder)


    def forward(self, x):

        encoded, pool_indices = self.encoder(x)


        return self.decoder(encoded, pool_indices)



