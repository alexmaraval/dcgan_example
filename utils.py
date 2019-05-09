import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import GPyOpt
import math


def denorm(x, channels=None, w=None ,h=None, resize = False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


# Create Train and Test functions
# Functions to optimise in the GPyOpt prosucess
def test_c(loader, clf, cae):
    clf.eval()
    cae.eval()
    test_acc = 0
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            img, img_class = data
            img = img.to(device)
            img_class = img_class.to(device)
            code = cae.encode(img)
            pred_class = clf.forward(code)
            pred_class = torch.argmax(pred_class.view((-1, 10)), 1)
            num_correct += (pred_class == img_class).sum()
            num_samples += pred_class.size(0)
            test_acc = float(num_correct) / num_samples

    return 1-test_acc


def test_l(loader, clf, cae):
    clf.eval()
    cae.eval()
    test_acc = 0
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            img, img_class = data
            img = img.to(device)
            img_class = img_class.to(device)
            code = cae.encode(img)
            code = code.view((code.shape[0], code.shape[1]))
            pred_class = clf.forward(code)
            pred_class = torch.argmax(pred_class.view((-1, 10)), 1)
            num_correct += (pred_class == img_class).sum()
            num_samples += pred_class.size(0)
            test_acc = float(num_correct) / num_samples

    return 1-test_acc


def train_c(clf, cae, optimizer, loader=loader_train, num_epochs=1):
    clf.train()
    clf.eval()

    for epoch in range(num_epochs):
        clf_train_loss = 0
        for i, data in enumerate(loader_train):
            img, img_class = data
            img = img.to(device)
            img_class = img_class.to(device)
            optimizer.zero_grad()
            # forward
            with torch.no_grad():
                code = cae.encode(img)
            class_pred = clf.forward(code)
            class_pred = class_pred.view((-1,10))
            loss = F.cross_entropy(class_pred, img_class)
            # backward
            loss.backward()
            clf_train_loss += loss.item()
            optimizer.step()
        # print out losses and save reconstructions for every epoch
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, clf_train_loss / len(loader_train)))



def train_l(clf, cae, optimizer, loader=loader_train, num_epochs=1):
    clf.train()
    clf.eval()

    for epoch in range(num_epochs):
        clf_train_loss = 0
        for i, data in enumerate(loader_train):
            img, img_class = data
            img = img.to(device)
            img_class = img_class.to(device)
            optimizer.zero_grad()
            # forward
            with torch.no_grad():
                code = cae.encode(img)
                code = code.view((code.shape[0], code.shape[1]))
            class_pred = clf.forward(code)
            class_pred = class_pred.view((-1,10))
            loss = F.cross_entropy(class_pred, img_class)
            # backward
            loss.backward()
            clf_train_loss += loss.item()
            optimizer.step()
        # print out losses and save reconstructions for every epoch
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, clf_train_loss / len(loader_train)))



def train_conv(alpha):
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('--------------------------- New model --------------------------')
    print()

    clf = ConvClassifier().to(device)
    cae = CAE().to(device)
    cae.load_state_dict(torch.load('CAE_model.pth'))
    lr = pow(10, -float(alpha))
    print('lrearning rate:', lr)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    train_c(clf, cae, optimizer, loader=loader_train, num_epochs=20)

    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print()

    return test_c(loader_test, clf, cae)

def train_lin(alpha):
    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('--------------------------- New model --------------------------')
    print()

    clf = LinClassifier().to(device)
    cae = CAE().to(device)
    cae.load_state_dict(torch.load('CAE_model.pth'))
    lr = pow(10, -float(alpha))
    print('lrearning rate:', lr)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    train_l(clf, cae, optimizer, loader=loader_train, num_epochs=20)

    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print()

    return test_l(loader_test, clf, cae)
