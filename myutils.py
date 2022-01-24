import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset


transform=transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.5,.5,0.5],std=[.5,.5,0.5]) 
])

def cal_iou(model, dataset):
    pa=mpa=miou=fwiou=0.
    for img, mask in dataset:
        mask = mask.cuda()
        with torch.no_grad():
            pred = model(img.unsqueeze(0).cuda())
            pred = torch.argmax(pred, 1).float()
        pa += get_pa(pred, mask)
        mpa += get_mpa(pred, mask)
        miou += get_miou(pred, mask)
        fwiou += get_fwiou(pred, mask)
    lenth = len(dataset)
    pa /= lenth
    mpa /= lenth
    miou /= lenth
    fwiou /= lenth
    return pa.item(), mpa.item(), miou.item(), fwiou.item()

def get_pa(pred, mask):
    return (pred==mask).sum().float()/(512*512)


def get_mpa(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum().float()/\
            (mask_crack.sum())/2 +\
            (pred_fine*mask_fine).sum().float()/\
            (mask_fine.sum())/2


def get_miou(pred, mask):

    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return (pred_crack*mask_crack).sum().float()/\
            ((mask_crack+pred_crack)!=0).sum()/2+\
            (pred_fine*mask_fine).sum().float()/\
            ((mask_fine+pred_fine)!=0).sum()/2


def get_fwiou(pred, mask):
    pred_crack = pred
    pred_fine = 1-pred
    mask_crack = mask
    mask_fine = 1-mask
    return  mask_crack.sum()*(pred_crack*mask_crack).sum().float()/\
            ((mask_crack+pred_crack)!=0).sum()/(512*512)+\
            mask_fine.sum()*(pred_fine*mask_fine).sum().float()/\
            ((mask_fine+pred_fine)!=0).sum()/(512*512)


def onehot(masks):
    masks_t = torch.ones(masks.size(0), 2, 
                    masks.size(2), masks.size(3)).cuda()
    masks_t[:,0,:,:][masks[:,0,:,:]==0] = 1 
    masks_t[:,1,:,:][masks[:,0,:,:]==1] = 1
    return masks_t