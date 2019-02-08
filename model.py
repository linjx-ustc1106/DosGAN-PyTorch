import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
#from cnn_finetune import make_model
class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ResnetEncoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_blocks=3):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = 64
        padding_type ='reflect'
        norm_layer = nn.InstanceNorm2d
        use_bias = False
        
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(ngf, affine=True, track_running_stats=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2, affine=True, track_running_stats=True),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            model += [ResidualBlock(dim_in=ngf * mult, dim_out=ngf * mult)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
class ResnetDecoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_blocks=3, ft_num=16, image_size=128):
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = 64
        ngf_o = ngf*2
        padding_type ='reflect'
        norm_layer = nn.InstanceNorm2d
        use_bias = False

        model = [ ]
        n_downsampling = 2
        mult = 2**n_downsampling
        model_2 = [ ]
        model_2 += [nn.Linear(ft_num, ngf * mult * int(image_size / np.power(2, n_downsampling)) * int(image_size / np.power(2, n_downsampling)))]
        model_2 += [nn.ReLU(True)]
        
        model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult, affine=True, track_running_stats=True ),
                      nn.ReLU(True)]         
        for i in range(n_blocks):
            model += [ResidualBlock(dim_in=ngf * mult, dim_out=ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2), affine=True, track_running_stats=True),
                      nn.ReLU(True)]
        model += [nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=3, bias=False)]
        
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.model_2 = nn.Sequential(*model_2)

    def forward(self, input1, input2):
        out_2 = self.model_2(input2)
        out_2 = out_2.view(input1.size(0), input1.size(1), input1.size(2), input1.size(3))

        return  self.model(input1+out_2)#  self.model(torch.cat([input1, input2], dim=1))


class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=2, repeat_num=6, ft_num = 16, n_blocks = 3):
        super(Classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        for i in range(n_blocks):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(*[nn.Conv2d(curr_dim, ft_num, kernel_size= kernel_size), nn.LeakyReLU(0.01)])
        
        self.conv2 = nn.Conv2d(ft_num, c_dim, kernel_size=1, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(out_src)
        return out_src.view(out_src.size(0), out_src.size(1)), out_cls.view(out_cls.size(0), out_cls.size(1))

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, ft_num = 16):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, ft_num, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

