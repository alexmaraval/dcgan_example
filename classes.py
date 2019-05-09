import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt



class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.inc = 8
        # encoder
        self.enc_conv1 = nn.Conv2d(3, 2*self.inc, 2, 2, 0)
        self.enc_conv2 = nn.Conv2d(2*self.inc, 2*self.inc, 3, 1, 1)
        self.enc_batchn1 = nn.BatchNorm2d(2*self.inc)

        self.enc_conv3 = nn.Conv2d(2*self.inc, 4*self.inc, 2, 2, 0)
        self.enc_conv4 = nn.Conv2d(4*self.inc, 4*self.inc, 3, 1, 1)
        self.enc_batchn2 = nn.BatchNorm2d(4*self.inc)

        self.enc_conv5 = nn.Conv2d(4*self.inc, 8*self.inc, 2, 2, 0)
        self.enc_conv6 = nn.Conv2d(8*self.inc, 8*self.inc, 3, 1, 1)
        self.enc_batchn3 = nn.BatchNorm2d(8*self.inc)

        self.enc_conv7 = nn.Conv2d(8*self.inc, hidden_size, 4, 1, 0)


        # decoder
        self.dec_tconv1 = nn.ConvTranspose2d(8*self.inc, 4*self.inc, 2, 2, 0)
        self.dec_conv1 = nn.Conv2d(4*self.inc, 4*self.inc, 3, 1, 1)
        self.dec_batchn1 = nn.BatchNorm2d(4*self.inc)

        self.dec_tconv2 = nn.ConvTranspose2d(4*self.inc, 2*self.inc, 2, 2, 0)
        self.dec_conv2 = nn.Conv2d(2*self.inc, 2*self.inc, 3, 1, 1)
        self.dec_batchn2 = nn.BatchNorm2d(2*self.inc)

        self.dec_tconv3 = nn.ConvTranspose2d(2*self.inc, self.inc, 2, 2, 0)
        self.dec_conv3 = nn.Conv2d(self.inc, self.inc, 3, 1, 1)
        self.dec_batchn3 = nn.BatchNorm2d(self.inc)

        self.dec_conv4 = nn.Conv2d(self.inc, 3, 3, 1, 1)


        # functions
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def encode(self, x):
        x = self.leakyrelu(self.enc_batchn1(self.enc_conv2(self.enc_conv1(x))))
        x = self.leakyrelu(self.enc_batchn2(self.enc_conv4(self.enc_conv3(x))))
        x = self.leakyrelu(self.enc_batchn3(self.enc_conv6(self.enc_conv5(x))))
        x = self.sigmoid(self.enc_conv7(x))
        return x


    def decode(self, z):

        z = z.view(z.size(0), hidden_size//16, 4, 4)
        z = self.relu(self.dec_batchn1(self.dec_conv1(self.dec_tconv1(z))))
        z = self.relu(self.dec_batchn2(self.dec_conv2(self.dec_tconv2(z))))
        z = self.relu(self.dec_batchn3(self.dec_conv3(self.dec_tconv3(z))))
        z = self.tanh(self.dec_conv4(z))
        return z


    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon


# SECOND ATTEMPT : MLP Classification
# Classifier Network
class LinClassifier(nn.Module):
    def __init__(self):
        # input 1 32 32
        super(LinClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.relu(self.fc5(x))
        x = self.softmax(self.fc6(x))
        return x


# THIRD ATTEMPT : Convolutional Neural Net for Classification
class ConvClassifier(nn.Module):
    def __init__(self):
        # input 1 32 32
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(2, 2, 3, 1, 1)
        self.batchn1 = nn.BatchNorm2d(2)

        self.conv3 = nn.Conv2d(2, 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(4, 4, 3, 1, 1)
        self.batchn2 = nn.BatchNorm2d(4)

        self.conv5 = nn.Conv2d(4, 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(8, 8, 3, 1, 1)
        self.batchn3 = nn.BatchNorm2d(8)

        self.conv7 = nn.Conv2d(8, 10, 4, 1, 0)

        self.softmax = nn.Softmax()
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view((b, 1, 32, 32))
        x = self.leakyrelu(self.batchn1(self.conv2(self.conv1(x))))
        x = self.leakyrelu(self.batchn2(self.conv4(self.conv3(x))))
        x = self.leakyrelu(self.batchn3(self.conv6(self.conv5(x))))
        x = self.softmax(self.conv7(x))
        return x




class Generator(nn.Module):
    def __init__(self, out=128):
        super(Generator, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(100, 4*out, 4, 1, 0)
        self.conv1 = nn.Conv2d(4*out, 4*out, 3, 1, 1)
        self.batchn1 = nn.BatchNorm2d(4*out)

        self.tconv2 = nn.ConvTranspose2d(4*out, 2*out, 4, 2, 1)
        self.conv2 = nn.Conv2d(2*out, 2*out, 3, 1, 1)
        self.batchn2 = nn.BatchNorm2d(2*out)

        self.tconv3 = nn.ConvTranspose2d(2*out, out, 4, 2, 1)
        self.conv3 = nn.Conv2d(out, out, 3, 1, 1)
        self.batchn3 = nn.BatchNorm2d(out)

        self.tconv4 = nn.ConvTranspose2d(out, 3, 4, 2, 1)

        # functions
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def decode(self, z):
        z = self.relu(self.batchn1(self.conv1(self.tconv1(z))))
        z = self.relu(self.batchn2(self.conv2(self.tconv2(z))))
        z = self.relu(self.batchn3(self.conv3(self.tconv3(z))))
        z = self.tanh(self.tconv4(z))
        return z

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self, inc=8):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 2*inc, 4, 2, 1)
        self.conv2 = nn.Conv2d(2*inc, 2*inc, 3, 1, 1)
        self.batchn1 = nn.BatchNorm2d(2*inc)

        self.conv3 = nn.Conv2d(2*inc, 4*inc, 4, 2, 1)
        self.conv4 = nn.Conv2d(4*inc, 4*inc, 3, 1, 1)
        self.batchn2 = nn.BatchNorm2d(4*inc)

        self.conv5 = nn.Conv2d(4*inc, 8*inc, 4, 2, 1)
        self.conv6 = nn.Conv2d(8*inc, 8*inc, 3, 1, 1)
        self.batchn3 = nn.BatchNorm2d(8*inc)

        self.conv7 = nn.Conv2d(8*inc, 1, 4, 1, 0)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def discriminator(self, x):
        x = self.leakyrelu(self.batchn1(self.conv2(self.conv1(x))))
        x = self.leakyrelu(self.batchn2(self.conv4(self.conv3(x))))
        x = self.leakyrelu(self.batchn3(self.conv6(self.conv5(x))))
        x = self.sigmoid(self.conv7(x))
        return x

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)
