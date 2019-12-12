from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fbpn_sr_rbpn_v1 import Net as FBPNSR_RBPN_V1
from fbpn_sr_rbpn_v2 import Net as FBPNSR_RBPN_V2
from fbpn_sr_rbpn_v3 import Net as FBPNSR_RBPN_V3
from fbpn_sr_rbpn_v4 import Net as FBPNSR_RBPN_V4
from fbpn_sr_rbpn_v1_ref import Net as FBPNSR_RBPN_V1_REF
from fbpn_sr_rbpn_v2_ref import Net as FBPNSR_RBPN_V2_REF
from fbpn_sr_rbpn_v3_ref import Net as FBPNSR_RBPN_V3_REF
from fbpn_sr_rbpn_v4_ref import Net as FBPNSR_RBPN_V4_REF, FeatureExtractor
from data import get_test_set_interp
from functools import reduce
import numpy as np
import utils
import time
import cv2
import math
import pdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='vimeo_triplet/sequences')
parser.add_argument('--file_list', type=str, default='tri_testlist.txt')
parser.add_argument('--model_type', type=str, default='FBPNSR_RBPN_V4_REF')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results_T_SR_HR/', help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/FBPNSR_RBPN_V4_REF_Lf_STAR_T_HR.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_test_set_interp(opt.data_dir, opt.file_list)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'FBPNSR_RBPN_V1_REF':
    model = FBPNSR_RBPN_V1_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V2_REF':
    model = FBPNSR_RBPN_V2_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V3_REF':
    model = FBPNSR_RBPN_V3_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V4_REF':
    model = FBPNSR_RBPN_V4_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V1':
    model = FBPNSR_RBPN_V1(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V2':
    model = FBPNSR_RBPN_V2(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V3':
    model = FBPNSR_RBPN_V3(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V4':
    model = FBPNSR_RBPN_V4(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
       

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, flow_f, flow_b, filename, d_dir = batch[0], batch[1], batch[2], batch[3], batch[4]
        
        with torch.no_grad():
            t_im1 = Variable(input[0]).cuda(gpus_list[0])
            t_im2 = Variable(input[1]).cuda(gpus_list[0])
            t_flow_f = Variable(flow_f).cuda(gpus_list[0]).float()
            t_flow_b = Variable(flow_b).cuda(gpus_list[0]).float()
            
        t0 = time.time()                
        if opt.chop_forward:
            with torch.no_grad():
                pred_l  = chop_forward(t_im1, t_im2, t_flow_f, t_flow_b, model)
        else:
            with torch.no_grad():
                _, _, _, pred_l  = model(t_im1, t_im2, t_flow_f, t_flow_b, train=False)
            
        t1 = time.time()
        
        print("===> Processing: %s || Timer: %.4f sec." % (d_dir[0]+'/frame10i11.png', (t1 - t0)))
        pred_l = utils.denorm(pred_l[0].cpu().data, vgg=True)
        pred_1 = utils.denorm(t_im1[0].cpu().data, vgg=True)
        pred_2 = utils.denorm(t_im2[0].cpu().data, vgg=True)

        if opt.data_dir == 'ucf101_interp_ours':
            save_img(pred_1, d_dir[0],'frame_00.png', False)
            save_img(pred_l, d_dir[0],'frame_01_gt.png', False)
            save_img(pred_2, d_dir[0],'frame_02.png', False)
        else:
            save_img(pred_1, d_dir[0],'im1.png', False)
            save_img(pred_l, d_dir[0],'im2.png', False)
            save_img(pred_2, d_dir[0],'im3.png', False)
        
        
def save_img(img, d_dir,img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    filename = os.path.splitext(img_name)

    # save img
    save_dir=os.path.join(opt.output, d_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ filename[0]+'_'+opt.model_type+filename[1]
    else:
        save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
def chop_forward(t_im1, t_im2, t_flow_f, t_flow_b, model, shave=8, min_size=200000, nGPUs=opt.gpus):
    b, c, h, w = t_im1.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    
    if h_size%2:
        h_size = h_size + 1

    if w_size%2:
        w_size = w_size + 1
        
    inputlist = [
        [t_im1[:, :, 0:h_size, 0:w_size], t_im2[:, :, 0:h_size, 0:w_size], t_flow_f[:, :, 0:h_size, 0:w_size], t_flow_b[:, :, 0:h_size, 0:w_size]],
        [t_im1[:, :, 0:h_size, (w - w_size):w],t_im2[:, :, 0:h_size, (w - w_size):w],t_flow_f[:, :, 0:h_size, (w - w_size):w],t_flow_b[:, :, 0:h_size, (w - w_size):w] ],
        [t_im1[:, :, (h - h_size):h, 0:w_size],t_im2[:, :, (h - h_size):h, 0:w_size],t_flow_f[:, :, (h - h_size):h, 0:w_size],t_flow_b[:, :, (h - h_size):h, 0:w_size] ],
        [t_im1[:, :, (h - h_size):h,  (w - w_size):w],t_im2[:, :, (h - h_size):h,  (w - w_size):w],t_flow_f[:, :, (h - h_size):h,  (w - w_size):w],t_flow_b[:, :, (h - h_size):h,  (w - w_size):w] ]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = inputlist[i]
                _, _, _, output_batch = model(input_batch[0], input_batch[1], input_batch[2], input_batch[3], train=False)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2],patch[3], model, shave, min_size, nGPUs) \
            for patch in inputlist]

    scale=1
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(t_im1.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()
