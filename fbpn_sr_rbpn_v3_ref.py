import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from rbpn import Net as RBPN
from autoencoder_v4 import UNet
from torch.autograd import Variable
from fbpn_sr_rbpn_v3 import Net as FBPNSR_RBPN_V3

class Net(nn.Module):
    def __init__(self, base_filter, feat, num_stages, n_resblock, scale_factor, pretrained=False, freeze=False):
        super(Net, self).__init__()    
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        self.model = FBPNSR_RBPN_V3(base_filter=base_filter,  feat = feat, num_stages=num_stages, n_resblock=n_resblock, scale_factor=scale_factor) 
        self.flow_net = UNet(8,2)
        
        if pretrained:
            #self.model.load_state_dict(torch.load("weights/pretrained/FBPNSR_RBPN_V1_STAR-T.pth", map_location=lambda storage, loc: storage))        
            self.flow_net.load_state_dict(torch.load("weights/pretrained/flow_refinement.pth", map_location=lambda storage, loc: storage))    
            
        if freeze:
            self.freeze_model(self.model)
            
    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        
    def forward(self, t_im1, t_im2, t_flow_f, t_flow_b, train=True, flowRefine= True, noise=False):
        ##flow refinement
        if flowRefine:
            if noise:
                t_flow_f = t_flow_f + Variable(torch.randn(t_flow_f.size()).cuda() * 0.1)
                t_flow_b = t_flow_b + Variable(torch.randn(t_flow_b.size()).cuda() * 0.1)
            t_flow_f = t_flow_f + (self.flow_net(torch.cat((t_flow_f,t_im1, t_im2),1)))
            t_flow_b = t_flow_b + (self.flow_net(torch.cat((t_flow_b,t_im2, t_im1),1)))
        
        pred_ht, pred_h1, pred_h2, pred_l = self.model(t_im1, t_im2, t_flow_f, t_flow_b, train=train)
        
        if train:
            if flowRefine:
                return pred_ht, pred_h1, pred_h2, pred_l, t_flow_f, t_flow_b
            else:
                return pred_ht, pred_h1, pred_h2, pred_l
        else:
            return pred_ht, pred_h1, pred_h2, pred_l
            
class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=35):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)