# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:02:17 2019

@author: Pulkit Dixit
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
#import numpy as np
import os

root = 'gdrive/My Drive/Google Colab/'
batch_size = 100
learn_rate = 0.001
scheduler_step_size = 8
scheduler_gamma = 0.5
num_epochs = 50

transform_train = transforms.Compose([transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                     ])

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation
    images into separate sub folders
    """
    # path where validation data is present now
    path = os.path.join(val_dir, 'images')
    # file where image2class mapping is present
    filename = os.path.join(val_dir, 'val_annotations.txt')
    fp = open(filename, "r") # open file in read mode
    data = fp.readlines() # read line by line
    """
    Create a dictionary with image names as key and
    corresponding classes as values
    """
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath): # check if folder exists
            os.makedirs(newpath)
        # Check if image exists in default directory
        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

train_dir = '/u/training/tra250/scratch/tiny-imagenet-200/train'
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

val_dir = '/u/training/tra250/scratch/tiny-imagenet-200/val/images'
if 'val_' in os.listdir(val_dir)[0]:
    create_val_folder(val_dir)
else:
    pass

val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

def conv3x3(in_channels, out_channels, stride = 1):
  return nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 3, padding = 1)

class BasicBlock(nn.Module):
  #expansion = 1
  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = True)
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample
    self.stride = stride
    
  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 100):
    super(ResNet, self).__init__()
    
    #Pre-basic block convolution:
    self.in_channels = 32
    self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.conv = conv3x3(3, self.in_channels)
    self.bn = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace = True)
    self.drop_out = nn.Dropout(0.1)
    
    #Basic block 1:
    self.layer1 = self.make_layer(block, 32, layers[0])
    
    #Basic block 2:
    self.layer2 = self.make_layer(block, 64, layers[1], 2)
    
    #Basic block 3:
    self.layer3 = self.make_layer(block, 128, layers[2], 2)
    
    #Basic block 3:
    self.layer4 = self.make_layer(block, 256, layers[3], 2)
    
    #Post Block pooling and linearization:
    self.maxpool = nn.AdaptiveMaxPool2d((1,1))
    self.drop_out2 = nn.Dropout(0.1)
    #self.linear = nn.Linear(32*block.expansion, 100)
    self.linear = nn.Linear(256, 200)
    
  def make_layer(self, block, out_channels, layer_len, stride = 1):
    downsample = None
    if (stride != 1) or (self.in_channels != out_channels):
      downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size = 1, stride = stride),
                                nn.BatchNorm2d(out_channels))
    
    layer = []
    layer.append(block(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels
    for i in range(1, layer_len):
      layer.append(block(out_channels, out_channels))
    return nn.Sequential(*layer)
  
  def forward(self, x):
    out = self.maxpool(x)
    out = self.conv(x)
    #print('First convolution: \t', out.size())
    out = self.bn(out)
    out - self.relu(out)
    out = self.drop_out(out)
    
    out = self.layer1(out)
    out = self.drop_out(out)
    #print('First layer: \t\t', out.size())
    
    out = self.layer2(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.layer3(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.layer4(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.maxpool(out)
    out = self.drop_out2(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return(out)

layers = [2, 4, 4, 2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(BasicBlock, layers)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr = learn_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = scheduler_step_size, 
                                            gamma = scheduler_gamma)

train_acc_list = []
test_acc_list = []
for epochs in range(num_epochs):
    scheduler.step()
    correct = 0
    total = 0
    print('Current epoch: \t\t', epochs+1, '/', num_epochs)
    #print('--------------------------------------------------')
    for images, labels in train_loader:
        #images = images.reshape(-1, 16*16)
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc = correct/total
    print('Training accuracy: \t', train_acc)
    #print('--------------------------------------------------')
    train_acc_list.append(train_acc)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    test_acc = correct/total
    print('Test Accuracy: \t\t', test_acc)
    print('**************************************************')
    test_acc_list.append(test_acc)
    model.train()

print(train_acc_list)
print(test_acc_list)













































