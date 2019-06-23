import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import os
from torchviz import make_dot
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

import networks
import image_processing

np.random.seed(10)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class MagicBox:
    def __init__(self, base_model = 'unet_256', gpu_ids = [0]):
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        print("Using device", self.device)
        self.netG = networks.define_G(3, 3, 64, netG = base_model, gpu_ids = self.gpu_ids)
        print("Network Initialized")
        self.opt = {'crop_size':256, 'no_flip':False, 'load_size':256,'preprocess':'resize_and_crop'}
        
    def processImage(self, img):
        transform_params = image_processing.get_params(self.opt, img.size)
        A_transform = image_processing.get_transform(self.opt, transform_params, grayscale=False)
        _A = A_transform(img)
        return _A

    def preprocess(self, img):
        return self.processImage(img)

    def run(self, img, transformed=True, train=False):
        if (not isinstance(img, torch.Tensor)) or transformed == False:
            print("Preprocessing and transforming image")
            img = self.processImage(img)
        out = self.netG(img)  # Forward Pass
        if train:
            return out, True
        return tensor2im(out), True
        
    def load_networks(self, save_dir="models/vanilla/", epoch = 'latest', name ='e2s', modeltype='G'):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s_%s.pth' % (epoch, name, modeltype)
        """
        load_filename = '%s_net_%s_%s.pth' % (epoch, name, modeltype)
        load_path = os.path.join(save_dir, load_filename)
        net = getattr(self, 'net' + modeltype)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        
        num_params = 0
        for param in self.netG.parameters():
            num_params += param.numel()
        if verbose:
            print(self.netG)
        print('[Network %s] Total number of parameters : %.3f M' % ('G', num_params / 1e6))  
        print('-----------------------------------------------')
        
