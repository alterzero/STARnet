import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from skimage import color
from random import randrange
import os.path
import cv2

max_flow = 150.0 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, scale):
    list=os.listdir(filepath)
    list.sort()
    
    rate = 1
    #for vimeo90k-setuplet (multiple temporal scale)
    #if random.random() < 0.5:
    #    rate = 2
    
    index = randrange(0, len(list)-(2*rate))
    
    target = [modcrop(Image.open(filepath+'/'+list[i]).convert('RGB'), scale) for i in range(index, index+3*rate, rate)]
    
    h,w = target[0].size
    h_in,w_in = int(h//scale), int(w//scale)
    
    target_l = target[1].resize((h_in,w_in), Image.BICUBIC)
    input = [target[j].resize((h_in,w_in), Image.BICUBIC) for j in [0,2]]
    
    return input, target, target_l, list

def load_img_test(filepath, scale):
    list=os.listdir(filepath)
    list.sort()
    
    target = [modcrop(Image.open(filepath+'/'+list[i]).convert('RGB'), scale) for i in range(len(list))]
    h,w = target[0].size
    h_in,w_in = int(h//scale), int(w//scale)
    
    input = [target[j].resize((h_in,w_in), Image.BICUBIC) for j in [0,len(list)-1]]
    
    return input, list

def load_img_nodown(filepath):
    list=os.listdir(filepath)
    list.sort()
    
    input = [Image.open(filepath+'/'+list[i]).convert('RGB') for i in [0,len(list)-1]]
    
    return input, list
    
def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75 #0.95 #0.75
    minWidth = 20 #50 #20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
    #Rescale
    flow = rescale_flow(flow,-1,1)
    return flow

def rescale_flow(x,max_range,min_range):
    #remove noise
    x[x > max_flow] = max_flow
    x[x < -max_flow] = -max_flow
    
    max_val = max_flow 
    min_val = -max_flow 
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_tar_l, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_in] 
    img_tar = [j.crop((ty,tx,ty + tp, tx + tp)) for j in img_tar] 
    img_tar_l = img_tar_l.crop((iy,ix,iy + ip, ix + ip)) 
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_tar_l, info_patch

def augment(img_in, img_tar, img_tar_l, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = [ImageOps.flip(j) for j in img_in]
        img_tar = [ImageOps.flip(j) for j in img_tar]
        img_tar_l = ImageOps.flip(img_tar_l)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = [ImageOps.mirror(j) for j in img_in]
            img_tar = [ImageOps.mirror(j) for j in img_tar]
            img_tar_l = ImageOps.mirror(img_tar_l)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = [j.rotate(180) for j in img_in]
            img_tar = [j.rotate(180) for j in img_tar]
            img_tar_l = img_tar_l.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, img_tar_l, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
                    
        if self.transform:
            input = [self.transform(j) for j in input]
            target = [self.transform(j) for j in target]
            target_l = self.transform(target_l)
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 

        return input, target, target_l, flow_f, flow_b, file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderFlow(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolderFlow, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        gt_flow_f = get_flow(input[0],target_l) + get_flow(target_l,input[1])
        gt_flow_b = get_flow(input[1],target_l) + get_flow(target_l,input[0])
                    
        if self.transform:
            input = [self.transform(j) for j in input]
            target = [self.transform(j) for j in target]
            target_l = self.transform(target_l)
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            gt_flow_f = torch.from_numpy(gt_flow_f.transpose(2,0,1)) 
            gt_flow_b = torch.from_numpy(gt_flow_b.transpose(2,0,1)) 

        return input, target, target_l, flow_f, flow_b, gt_flow_f, gt_flow_b,file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
        
class DatasetFromFolderFlowLR(data.Dataset):
    def __init__(self, image_dir, upscale_factor, data_augmentation, file_list, patch_size, transform=None):
        super(DatasetFromFolderFlowLR, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size

    def __getitem__(self, index):
        input, target, target_l, file_list = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, target_l, _ = get_patch(input,target,target_l,self.patch_size, self.upscale_factor)
        
        if self.data_augmentation:
            input, target, target_l, _ = augment(input, target, target_l)
            
        flow_f = get_flow(target[0],target[2])
        flow_b = get_flow(target[2],target[0])
        
        gt_flow_f = get_flow(target[0],target[1]) + get_flow(target[1],target[2])
        gt_flow_b = get_flow(target[2],target[1]) + get_flow(target[1],target[0])
                    
        if self.transform:
            target = [self.transform(j) for j in target]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            gt_flow_f = torch.from_numpy(gt_flow_f.transpose(2,0,1)) 
            gt_flow_b = torch.from_numpy(gt_flow_b.transpose(2,0,1)) 
            

        return target, flow_f, flow_b, gt_flow_f, gt_flow_b, file_list, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
    
class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, upscale_factor, file_list, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input, file_list = load_img_test(self.image_filenames[index], self.upscale_factor)
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        if self.transform:
            input = [self.transform(j) for j in input]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            
        return input, flow_f, flow_b, file_list, self.image_filenames[index]
      
    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderInterp(data.Dataset):
    def __init__(self, image_dir, file_list, transform=None):
        super(DatasetFromFolderInterp, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.transform = transform

    def __getitem__(self, index):
        input, file_list = load_img_nodown(self.image_filenames[index])
            
        flow_f = get_flow(input[0],input[1])
        flow_b = get_flow(input[1],input[0])
        
        if self.transform:
            input = [self.transform(j) for j in input]
            flow_f = torch.from_numpy(flow_f.transpose(2,0,1)) 
            flow_b = torch.from_numpy(flow_b.transpose(2,0,1)) 
            
        return input, flow_f, flow_b, file_list, self.image_filenames[index]
      
    def __len__(self):
        return len(self.image_filenames)
    