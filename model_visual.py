import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm
from utils import load_state_dict_mute
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        y = self.classifier(x)
        return y

class ft_vit(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=384):
        super(ft_vit, self).__init__()
        model_ft=timm.create_model("vit_small_patch16_224", pretrained=True,drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        # self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        # self.avgpool2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(384, class_num, droprate, linear=linear_num, return_f = circle)
        print('Make sure timm > 0.6.0 and yu can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.model.forward_features(x)
        a = self.model.forward_head(x)

        #x = x.view(x.size(0), x.size(1))
        y = self.classifier(a)
        return y
    
# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        if x.dim()==3:
            x = self.avgpool1d(x.permute((0,2,1)))
        else: 
            x = self.avgpool2d(x.permute((0,3,1,2)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x