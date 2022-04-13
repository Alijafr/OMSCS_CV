import torch
import torchvision
import torchvision.models as models

import torchvision.transforms as transforms


import torch.nn as nn
import torch.nn.functional as F


# define the cutsom CNN architecture
class custom_model(nn.Module):
    def __init__(self,in_channels=3,num_classes= 11):
        super(custom_model, self).__init__()
        ## Define layers of a CNN
        #new size formula: new_h = [(input_h +2*padding - karnel_size)/stride] -1
        #expected image size is 32
        self.conv1= nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=(3,3),stride =1,padding=1)#stride and padding the default is 1
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride =1,padding=1)
        
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride =1,padding=1)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride =1,padding=1)
        
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride =1,padding=1)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride =1,padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.6)
        self.pool=nn.MaxPool2d(2,2) #devide by size by half
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        #after 4 pooling layer of filter size 2 and stride 2
        #the image size will be (4,4,512)
        self.fc1 = nn.Linear(4 * 4 * 512, 4000)
        self.fc2 = nn.Linear(4000, 1500)
        self.fc3 = nn.Linear(1500, num_classes)
    
    def forward(self, x):
        ## Define forward behavior
        #expected input is 32x32 images 
        x=self.bn1(F.relu(self.conv1(x)))
        x=self.bn2(F.relu(self.conv2(x)))
        x=self.pool(self.bn2(F.relu(self.conv3(x)))) # image became 16x16
        x=self.bn3(F.relu(self.conv4(x)))
        x=self.pool(self.bn4(F.relu(self.conv5(x)))) # image became 8x8
        x=self.pool(self.bn5(F.relu(self.conv6(x)))) # image became 4x4
        
        # flatten image input
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)# assuming uing the crossEntropy loss
        return x

# define the VGG16 CNN architecture
class VGG16(nn.Module):
    def __init__(self,pretrained=False,in_channels=3,num_classes=11):
        super(VGG16,self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        #change the classifers to accept the current input
        #adaptive_average pooling after the features leyer (CNNs) will change the input from (512,1,1) to (512,7,7) 
        # if pretrained:
        #     for param in self.model.features.parameters():
        #         param.requires_grad = False
        self.model.classifier[0]=nn.Linear(25088,4000)
        self.model.classifier[3]=nn.Linear(4000,1500) 
        self.model.classifier[6]=nn.Linear(1500,11)
    
    def forward(self, x):
        return self.model(x)
        

        
        
        
        
        
        
        
        
        
        
