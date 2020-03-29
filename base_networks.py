import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(input_size, output_size, kernel_size, stride, padding)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        
        if self.activation is not None:
            out = self.act(out)
            
        return out

class ResnetBlock3D(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(num_filter, num_filter, kernel_size, stride, padding)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        
        if self.activation is not None:
            out = self.act(out)
            
        return out

class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)        

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class UpBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)        

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0
        
class D_UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)        

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale,num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.up_conv3 = Upsampler(scale,num_filter)

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class DownBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4,bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlockPix(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)
        self.down_conv2 = Upsampler(scale,num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class PSBlock(nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(nn.PixelShuffle(2))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            #modules.append(nn.PReLU())
        self.up = nn.Sequential(*modules)
        
        self.activation = act
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.1, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out
             

class Upsample2xBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out

class PyramidModule(nn.Module):
    def __init__(self, num_inchannels,activation='prelu'):
        super(PyramidModule, self).__init__()
        
        self.l1_1 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l1_2 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l1_3 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l1_4 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l1_5 = ResnetBlock(num_inchannels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        
        self.l2_1 = ResnetBlock(num_inchannels*2, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l2_2 = ResnetBlock(num_inchannels*2, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l2_3 = ResnetBlock(num_inchannels*2, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l2_4 = ResnetBlock(num_inchannels*2, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        
        self.l3_1 = ResnetBlock(num_inchannels*4, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l3_2 = ResnetBlock(num_inchannels*4, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        self.l3_3 = ResnetBlock(num_inchannels*4, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, norm=None)
        
        self.down1 = ConvBlock(num_inchannels, num_inchannels*2, 4, 2, 1, bias=True, activation=activation, norm=None)
        self.down2 = ConvBlock(num_inchannels*2, num_inchannels*4, 4, 2, 1, bias=True, activation=activation, norm=None)
        
        self.up1 = DeconvBlock(num_inchannels*2, num_inchannels, 4, 2, 1, bias=True,  activation=activation, norm=None)
        self.up2 = DeconvBlock(num_inchannels*4, num_inchannels*2, 4, 2, 1, bias=True, activation=activation, norm=None)
        
        self.final = ConvBlock(num_inchannels, num_inchannels, 3, 1, 1, bias=True, activation=activation, norm=None)
        
    def forward(self, x):
        out1_1 = self.l1_1(x)
        out2_1 = self.l2_1(self.down1(out1_1))
        out3_1 = self.l3_1(self.down2(out2_1))
        
        out1_2 = self.l1_2(out1_1 + self.up1(out2_1))
        out2_2 = self.l2_2(out2_1 + self.down1(out1_2) + self.up2(out3_1))
        out3_2 = self.l3_2(out3_1 + self.down2(out2_2))
        
        out1_3 = self.l1_3(out1_2 + self.up1(out2_2))
        out2_3 = self.l2_3(out2_2 + self.down1(out1_3) + self.up2(out3_2))
        out3_3 = self.l3_3(out3_2 + self.down2(out2_3))
        
        out1_4 = self.l1_4(out1_3 + self.up1(out2_3))
        out2_4 = self.l2_4(out2_3 + self.down1(out1_4) + self.up2(out3_3))
        
        out1_5 = self.l1_5(out1_4 + self.up1(out2_4))
        
        final = self.final(out1_5)
        
        return final
    
