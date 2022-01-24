import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
#
class myDataset(Dataset):
    def __init__(self, root ,train=True, transform = None):
        Dataset.__init__(self)
        images_dir = os.path.join(root,'images')
        images = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, k) for k in images]
        self.images.sort()
        if train:
            masks_dir = os.path.join(root,'masks')
            masks = os.listdir(masks_dir)
            self.masks = [os.path.join(masks_dir, k) for k in masks]
            self.masks.sort()

        self.transforms = transform
        self.train = train
        
    def __getitem__(self, index):
        image_path = self.images[index]
        
        image = Image.open(image_path).resize([512,512])
        if self.transforms is not None:
            image = self.transforms(image)
        image = image
        if self.train :
            mask_path = self.masks[index]
            mask = Image.open(mask_path).resize([512,512])
            if self.transforms is not None:
                mask = self.transforms(mask)
                mask = mask.mean(dim=0).view(1,512,512)
                mask[mask>0] = 0
                mask[mask<0] = 1
                
            return image, mask
        return image
    
    def __len__(self):
        return len(self.images)
    