import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from rbpn import Net as RBPN
from autoencoder_v4 import UNet
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, base_filter, feat, num_stages, n_resblock, scale_factor, pretrained=True, freeze=False):
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
        
        #Initial Feature Extraction
        self.motion_feat = ConvBlock(4, base_filter, 3, 1, 1, activation='lrelu', norm=None)
        
        ###INTERPOLATION
        #Interp_block
        warping1 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        warping1.append(ConvBlock(feat, feat*2, kernel, stride, padding, activation='lrelu', norm=None))
        self.warp1 = nn.Sequential(*warping1)
        
        warping2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        warping2.append(ConvBlock(feat, feat*2, kernel, stride, padding, activation='lrelu', norm=None))
        self.warp2 = nn.Sequential(*warping2)
        
        motion_net = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        motion_net.append(ConvBlock(base_filter, feat, 3, 1, 1, activation='lrelu', norm=None))
        self.motion = nn.Sequential(*motion_net)
        
        interp_b = [
            ResnetBlock(feat*5, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(n_resblock)]
        interp_b.append(DeconvBlock(feat*5, feat, kernel, stride, padding, activation='lrelu', norm=None))
        self.interp_block = nn.Sequential(*interp_b)
        
        ###ITERATIVE REFINEMENT
        #Motion Up FORWARD
        modules_up_f = [
            ResnetBlock(feat*5, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(n_resblock)]
        modules_up_f.append(DeconvBlock(feat*5, feat, kernel, stride, padding, activation='lrelu', norm=None))
        self.motion_up_f = nn.Sequential(*modules_up_f)
        
        #Motion Up BACKWARD
        modules_up_b = [
            ResnetBlock(feat*5, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(n_resblock)]
        modules_up_b.append(DeconvBlock(feat*5, feat, kernel, stride, padding, activation='lrelu', norm=None))
        self.motion_up_b = nn.Sequential(*modules_up_b)
        
        #Motion Down
        modules_down = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(3)]
        modules_down.append(ConvBlock(feat, feat*2, kernel, stride, padding, activation='lrelu', norm=None))
        self.motion_down = nn.Sequential(*modules_down)
        
        self.relu_bp = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)#torch.nn.PReLU()
        
        #Reconstruction
        self.reconstruction_l = ConvBlock(feat*2, 3, 3, 1, 1, activation=None, norm=None)
        self.reconstruction_h = ConvBlock(feat, 3, 3, 1, 1, activation=None, norm=None)
        
        ####ALIGNMENT
        ###RBPN
        self.RBPN = RBPN(num_channels=3, base_filter=base_filter,  feat = feat, num_stages=num_stages, n_resblock=5, nFrames=2, scale_factor=scale_factor)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
        
        if pretrained:
            if scale_factor == 4:
                self.RBPN.load_state_dict(torch.load("weights/pretrained/rbpn_pretrained_F2_4x.pth", map_location=lambda storage, loc: storage))    
            elif scale_factor == 2:
                self.RBPN.load_state_dict(torch.load("weights/pretrained/rbpn_pretrained_F2_2x.pth", map_location=lambda storage, loc: storage))    
            
        if freeze:
            self.freeze_model(self.RBPN)
            
    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
    
        
    def forward(self, t_im1, t_im2, t_flow_f, t_flow_b, train=True):
        result_l = []
        result_h1 = []
        result_ht = []
        result_h2 = []
                
        ###ALIGNMENT
        aux_H1, H1 = self.RBPN(t_im1,[t_im2],[t_flow_f])
        aux_H2, H2 = self.RBPN(t_im2,[t_im1],[t_flow_b])
        L1 = self.warp1(H1)
        L2 = self.warp2(H2)
        
        ###MOTION & DEPTH  
        motion_feat0 = self.motion_feat(torch.cat((t_flow_f, t_flow_b),1))
        M = self.motion(motion_feat0)
        
        motion_feat1 = self.motion_feat(torch.cat((t_flow_f/2.0, t_flow_b/2.0),1))
        M_half = self.motion(motion_feat1)
                
        ###INTERPOLATION
        Ht = self.interp_block(torch.cat((L1,L2,M),1))
        L = self.motion_down(Ht)
        
        aux_Ht = self.reconstruction_h(Ht)
        aux_L = self.reconstruction_l(L)
        result_l.append(aux_L)
        result_h1.append(aux_H1)
        result_ht.append(aux_Ht)
        result_h2.append(aux_H2)
              
        ####Projection
        backward1 = torch.cat((L1, L, M_half),1)
        H_b = self.motion_up_b(backward1)
        H1 = H1 + self.relu_bp(H1 - H_b)
        L1 = L1 + self.relu_bp(L1 - self.motion_down(H_b))
            
        forwardd2 = torch.cat((L, L2, M_half),1)
        H_f = self.motion_up_f(forwardd2)
        H2 = H2 + self.relu_bp(H2 - H_f)
        L2 = L2 + self.relu_bp(L2 - self.motion_down(H_f))
        
        forwardd = torch.cat((L1, L, M_half),1)
        H_t_f = self.motion_up_f(forwardd)
        Ht = Ht + self.relu_bp(Ht - H_t_f)
        L = L + self.relu_bp(L - self.motion_down(H_t_f))
            
        backward = torch.cat((L, L2, M_half),1)
        H_t_b = self.motion_up_b(backward)  
        Ht = Ht + self.relu_bp(Ht - H_t_b)  
        L = L + self.relu_bp(L - self.motion_down(H_t_b))  
        
        output_ht = self.reconstruction_h(Ht)
        output_h1 = self.reconstruction_h(H1)
        output_h2 = self.reconstruction_h(H2)
        output_l = self.reconstruction_l(L)
        result_l.append(output_l)
        result_h1.append(output_h1)
        result_ht.append(output_ht)
        result_h2.append(output_h2)
        
        if train:
            return result_ht, result_h1, result_h2, result_l
        else:
            return output_ht, output_h1, output_h2, output_l
            
class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=35):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)