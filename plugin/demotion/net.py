from torch.autograd import Variable
import torch
import torch.nn  as nn
from torchvision.models import alexnet, vgg
import torch.nn.functional as F
import math
from ..util import down_sample_fit,max_size_fit
import gc



class GNet(nn.Module):



    def _r_block(self,input_c,output_c):
        r = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(input_c,output_c, 3, 1),
            #nn.BatchNorm2d(output_c),
            nn.InstanceNorm2d(output_c),
            nn.ReLU(True),
            nn.ReflectionPad2d(1), 
            nn.Conv2d(input_c,output_c, 3, 1),
            nn.InstanceNorm2d(output_c)
        )
        return r 

    def __init__(self):
        super(GNet, self).__init__()

        self.blocks1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, 64, 7, 2),
                nn.InstanceNorm2d(64),
                # Normalization(),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64,128,3,1),
                nn.InstanceNorm2d(128),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, 128, 3, 2),
                nn.InstanceNorm2d(128),
                # Normalization(),
                nn.ReLU(True)
            )


        self.r2 = self._r_block(128, 128)
        self.r3 = self._r_block(128, 128)
        self.r4 = self._r_block(128, 128)
        self.deconv = nn.Sequential(
            # nn.ConvTranspose2d(128,64, 4, 2, 1),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64, 4, 2,1),
#            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3,4, 2,1),
#            nn.InstanceNorm2d(32),
#            nn.ReLU(True),
#            nn.ReflectionPad2d(3),
#            nn.Conv2d(32,3,7),  
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x,flag=None):
        y2 = self.blocks1(x)
        r2 = self.r2(y2) + y2
        r3 = self.r3(r2) + r2

        r4 = self.r4(r3) + r3
        output = self.deconv(r4)
        output = torch.clamp(2 * output + x, -1, 1)
        return output



class Transfromer:

    def __init__(self,param):
        self.param = param

    def get_generator(self):
        return GNet()

    def get_preprocess(self):
        max_size = self.param.get("maxsize",1380)
        def preprocess(pil_img):
            pil_img = max_size_fit(pil_img,int(max_size))
            return down_sample_fit(pil_img,2)
        return preprocess

    def get_postprocess(self):
        def postprocess(img):
            return img
        return postprocess




        
