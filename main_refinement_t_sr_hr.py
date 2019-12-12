from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from fbpn_sr_rbpn_v1_ref import Net as FBPNSR_RBPN_V1_REF
from fbpn_sr_rbpn_v2_ref import Net as FBPNSR_RBPN_V2_REF
from fbpn_sr_rbpn_v3_ref import Net as FBPNSR_RBPN_V3_REF
from fbpn_sr_rbpn_v4_ref import Net as FBPNSR_RBPN_V4_REF, FeatureExtractor
from data import get_training_set_flow_lr
import utils
import pdb
import socket
import time
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=4, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_triplet/sequences/')
parser.add_argument('--file_list', type=str, default='tri_trainlist.txt')
parser.add_argument('--patch_size', type=int, default=48, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='FBPNSR_RBPN_V4_REF')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='FBPNSR_RBPN_V4_REF_Lf_STAR_T_HR.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='VGG_FR_VIMEO_STAR_T_HR', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    feature_extractor.eval()
    for iteration, batch in enumerate(training_data_loader, 1):
        target, flow_f, flow_b,gt_flow_f, gt_flow_b = batch[0], batch[1], batch[2],batch[3], batch[4]
        
        if cuda:
            t_flow_f = Variable(flow_f).cuda(gpus_list[0]).float()
            t_flow_b = Variable(flow_b).cuda(gpus_list[0]).float()
            gt_flow_f = Variable(gt_flow_f).cuda(gpus_list[0]).float()
            gt_flow_b = Variable(gt_flow_b).cuda(gpus_list[0]).float()
            
            t_hr1 = Variable(target[0]).cuda(gpus_list[0])
            t_hr = Variable(target[1]).cuda(gpus_list[0])
            t_hr2 = Variable(target[2]).cuda(gpus_list[0])
                
        optimizer.zero_grad()
        t0 = time.time()

        _, _, _, pred_l,pred_flow_f, pred_flow_b = model(t_hr1, t_hr2, t_flow_f, t_flow_b)
                                
        l_mse_lr = mse_loss_calc(pred_l, t_hr)        
        l_mse = l_mse_lr
        
        l_feat_lr = feat_loss_calc(pred_l, t_hr)
        l_feat = l_feat_lr
            
        l_flow = criterion(pred_flow_f, gt_flow_f) + criterion(pred_flow_b, gt_flow_b)
        
        #loss = l_flow
        loss = l_mse + 0.1*l_feat + 0.1*l_flow
        t1 = time.time()
            
        ###show sample
        pred_l = utils.denorm(pred_l[0][0].cpu().data,vgg=True)
        pred_l1 = utils.denorm(t_hr1[0].cpu().data,vgg=True)
        pred_l2 = utils.denorm(t_hr2[0].cpu().data,vgg=True)
        save_img(pred_l, "lr2")
        save_img(pred_l1, "lr1")
        save_img(pred_l2, "lr3")
            
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        #print(loss_mse_hr.data , loss_mse_lr.data, loss_feat_hr.data, loss_feat_l.data)    
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} MSE_LR: {:.4f} FEAT_LR: {:.4f}  Flow: {:.8f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, l_mse_lr,l_feat_lr, l_flow, (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def mse_loss_calc(pred, target, w_begin=0.4 , w_interval=0.6):
    weight = w_begin
    l_mse = 0 
    for i in range(len(pred)):
        l_mse_ht = criterion_l1(pred[i], target)
        l_mse = l_mse + weight*l_mse_ht
        weight += w_interval
    return l_mse

def feat_loss_calc(pred, target, w_begin=0.4 , w_interval=0.6):
    weight = w_begin
    l_feat = 0 
    hr_feature = feature_extractor(target)
    for i in range(len(pred)):
        sr_feature = feature_extractor(pred[i])
        l_feat_ht = criterion(sr_feature, hr_feature.detach())
        l_feat = l_feat + weight*l_feat_ht
        weight += w_interval
    return l_feat

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img    
    save_fn = 'im'+img_name+'_'+opt.model_type+opt.prefix+'.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set_flow_lr(opt.data_dir, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.patch_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'FBPNSR_RBPN_V1_REF':
    model = FBPNSR_RBPN_V1_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V2_REF':
    model = FBPNSR_RBPN_V2_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V3_REF':
    model = FBPNSR_RBPN_V3_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)
elif opt.model_type == 'FBPNSR_RBPN_V4_REF':
    model = FBPNSR_RBPN_V4_REF(base_filter=256,  feat = 64, num_stages=3, n_resblock=5, scale_factor=opt.upscale_factor)    

    
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion_l1 = nn.L1Loss()
criterion = nn.MSELoss()

###VGG
feature_extractor = FeatureExtractor(models.vgg19(pretrained=True), feature_layer=35)
feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids=gpus_list)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])
    criterion_l1 = criterion_l1.cuda(gpus_list[0])
    feature_extractor = feature_extractor.cuda(gpus_list[0])

optimizer = optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
checkpoint(epoch)
