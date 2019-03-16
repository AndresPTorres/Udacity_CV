## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Following conventions described in NaimishNet (suggested paper for FKPs)
        # https://arxiv.org/pdf/1710.00977.pdf
        # We will use stacked set of convolutional layer, activation function, maxpooling and dropout,
        # using similar parameters to the ones described in the referenced paper.
        
        self.pool = nn.MaxPool2d(2, 2)

        # 96x96 images
        self.conv1 = nn.Conv2d(1, 32, 5)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p = 0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p = 0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        #self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p = 0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        #self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p = 0.4)
        
        #self.conv5 = nn.Conv2d(256, 512, 2)
        #self.pool5 = nn.MaxPool2d(2, 2)
        #self.drop5 = nn.Dropout(p = 0.5)
        
        self.fc1   = nn.Linear(256 * 4 * 4, 1024)
        #self.fc1   = nn.Linear(512 * 1 * 1, 1024)
        #self.fc1   = nn.Linear(128 * 9 * 9, 1024)
        self.fc1_drop = nn.Dropout(p = 0.5)
        
        #self.fc2   = nn.Linear(512 * 1 * 1, 68 * 2)
        #self.fc2   = nn.Linear(1024, 68 * 2)
        self.fc2   = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(p = 0.5)
        
        self.fc3   = nn.Linear(1024, 68 * 2)
        
        self.show = 0
               
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # Activation function suggested by NaimishNet: ELU
        x = self.drop1(self.pool(F.selu(self.conv1(x))))
        x = self.drop2(self.pool(F.selu(self.conv2(x))))
        x = self.drop3(self.pool(F.selu(self.conv3(x))))
        x = self.drop4(self.pool(F.selu(self.conv4(x))))
        #x = self.drop5(self.pool5(F.elu(self.conv5(x))))
        
        #x = self.pool1(F.elu(self.conv1(x)))
        #x = self.pool2(F.elu(self.conv2(x)))
        #x = self.pool3(F.elu(self.conv3(x)))
        #x = self.pool4(F.elu(self.conv4(x)))
        #x = self.pool5(F.elu(self.conv5(x)))
        
        if self.show == 0:
            print('size')
            print(x.size())
            self.show = 1
        
        # prep for linear layers
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # Linear layers
        #x = self.fc1_drop(F.elu(self.fc1(x)))
        x = self.fc1_drop(F.selu(self.fc1(x)))
        #x = self.fc2(x)
        x = self.fc2_drop(self.fc2(x))
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
