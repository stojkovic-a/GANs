import torch
from torch import nn
import config as conf
import torch.nn.functional as F

class WSLinear(nn.Module):
    def __init__(
            self,in_features,out_features
    ):
        super(WSLinear,self).__init__()
        self.linear=nn.Linear(in_features,out_features)
        self.scale=(2/in_features)**0.5
        self.bias=self.linear.bias
        self.linear.bias=None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self,x):
        return self.linear(x*self.scale)+self.bias
    
class PixenNorm(nn.Module):
    def __init__(self):
        super(PixenNorm,self).__init__()
        self.epsilon=1e-8
    def forward(self,x):
        return x/torch.sqrt(torch.mean(x**2,dim=1,keepdim=True)+self.epsilon)


class MappingNetwork(nn.Module):
    def __init__(self,z_dim,w_dim):
        super().__init__()
        self.mapping=nn.Sequential(
            PixenNorm(),
            WSLinear(z_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
            nn.ReLU(),
            WSLinear(w_dim,w_dim),
        )
    def forward(self,x):
        return self.mapping(x)
    
class AdaIN(nn.Module):
    def __init__(self,channels,w_dim):
        super().__init__()
        self.instance_norm=nn.InstanceNorm2d(channels)
        self.style_scale=WSLinear(w_dim,channels)
        self.style_bias=WSLinear(w_dim,channels)
    
    def forward(self,x,w):
        x=self.instance_norm(x)
        style_scale=self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias=self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale *x+style_bias

class injectNoise(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(1,channels,1,1))
    def forward(self,x):
        noise=torch.randn((x.shape[0],1,x.shape[2], x.shape[3]),device=x.device)
        return x+self.weight+noise
    
class WSConv2d(nn.Module):
    def __init__(
            self,in_channels,out_channels,kernel_size=3, stride=1,padding=1
    ):
        super(WSConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.scale=(2/(in_channels*(kernel_size**2)))**0.5
        self.bias=self.conv.bias
        self.conv.bias=None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self,x):
        return self.conv(x*self.scale)+self.bias.view(1,self.bias.shape[0],1,1)

class GenBlock(nn.Module):
    def __init__(self,in_channel, out_channel, w_dim):
        super(GenBlock,self).__init__()
        self.conv1=WSConv2d(in_channel,out_channel)
        self.conv2=WSConv2d(out_channel,out_channel)
        self.leaky=nn.LeakyReLU(0.2,inplace=True)
        self.inject_noise1=injectNoise(out_channel)
        self.inject_noise2=injectNoise(out_channel)
        self.adain1=AdaIN(out_channel,w_dim)
        self.adain2=AdaIN(out_channel,w_dim)
    
    def forward(self,x,w):
        x=self.adain1(self.leaky(self.inject_noise1(self.conv1(x))),w)
        x=self.adain2(self.leaky(self.inject_noise2(self.conv2(x))),w)
        return x
    
class Generator(nn.Module):
    def __init__(self,z_dim,w_dim,in_channels,img_channels=3):
        super().__init__()
        self.starting_cte=nn.Parameter(torch.ones(1,in_channels,4,4))
        self.map=MappingNetwork(z_dim,w_dim)
        self.initial_adain1=AdaIN(in_channels,w_dim)
        self.initial_adain2=AdaIN(in_channels,w_dim)
        self.initial_noise1=injectNoise(in_channels)
        self.initial_noise2=injectNoise(in_channels)
        self.iniital_conv=nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.leaky=nn.LeakyReLU(0.2,inplace=True)

        self.initial_rgb=WSConv2d(in_channels,img_channels,kernel_size=1,stride=1,padding=0)
        self.prog_blocks,self.rgb_layers=(nn.ModuleList([]),
                                          nn.ModuleList([self.initial_rgb])
                                          )
        for i in range(len(conf.factors)-1):
            conv_in_c=int(in_channels*conf.factors[i])
            conv_out_c=int(in_channels*conf.factors[i+1])
            self.prog_blocks.append(GenBlock(conv_in_c,conv_out_c,w_dim))
            self.rgb_layers.append(WSConv2d(conv_out_c,img_channels,kernel_size=1,stride=1,padding=0))
    
    def fade_in(self,alpha,upscaled,generated):
        return torch.tanh(alpha*generated+(1-alpha)*upscaled)
    
    def forward(self,noise, alpha, steps):
        w=self.map(noise)
        x=self.initial_adain1(self.initial_noise1(self.starting_cte),w)
        x=self.iniital_conv(x)
        out=self.initial_adain2(self.leaky(self.initial_noise2(x)),w)

        if steps==0:
            # print('wwaaaaat')
            return self.initial_rgb(x)
        
        for step in range(steps):
            upscaled=F.interpolate(out,scale_factor=2,mode='bilinear')
            out=self.prog_blocks[step](upscaled,w)
        
        final_upscaled= self.rgb_layers[steps-1](upscaled)
        final_out=self.rgb_layers[steps](out)
        return self.fade_in(alpha,final_upscaled,final_out)
    

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()
        self.conv1=WSConv2d(in_channels,out_channels)
        self.conv2=WSConv2d(out_channels,out_channels)
        self.leaky=nn.LeakyReLU(0.2)

    def forward(self,x):
        x=self.leaky(self.conv1(x))
        x=self.leaky(self.conv2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # here we work back ways from factors because the discriminator
        # should be mirrored from the generator. So the first prog_block and
        # rgb layer we append will work for input size 1024x1024, then 512->256-> etc
        for i in range(len(conf.factors) - 1, 0, -1):
            conv_in = int(in_channels * conf.factors[i])
            conv_out = int(in_channels * conf.factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # perhaps confusing name "initial_rgb" this is just the RGB layer for 4x4 input size
        # did this to "mirror" the generator initial_rgb
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        # this is the block for 4x4 input size
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        # where we should start in the list of prog_blocks, maybe a bit confusing but
        # the last is for the 4x4. So example let's say steps=1, then we should start
        # at the second to last because input_size will be 8x8. If steps==0 we just
        # use the final block
        cur_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
