import os
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

from utils import *
from classes import *


# Device Selection
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


# Reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(1)


# Data loading
batch_size = 128

if not os.path.exists('./CAE'):
    os.makedirs('./CAE')
if not os.path.exists('./DCGAN'):
    os.makedirs('./DCGAN')


NUM_TRAIN = 49000
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


data_dir = './datasets'
cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
cifar10_val = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

loader_train = DataLoader(cifar10_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(cifar10_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(cifar10_test, batch_size=batch_size)

it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]
save_image(denorm(fixed_input), './CW/CAE/input_sample.png')

##############################
# Convolutional Autoencoder
##############################
num_epochs = 100
learning_rate  = 1e-3
hidden_size = 1024


# Define Loss function
criterion = nn.MSELoss(reduction='sum')  # can we use any other loss here? You are free to choose.
def loss_function_CAE(recon_x, x):
    recon_loss = criterion(recon_x, x)
    return recon_loss


# Initialize Model and print number of parameters
model = CAE().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
print(model)


# Choose and initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train
train_losses = []
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, data in enumerate(loader_train):
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        # forward
        recon_batch = model(img)
        loss = loss_function_CAE(recon_batch, img)
        # backward
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    # print out losses and save reconstructions for every epoch
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss / len(loader_train)))
    recon = model(fixed_input.to(device))
    recon = denorm(recon.cpu())
    save_image(recon, './CAE/reconstructed_epoch_{}.png'.format(epoch))
    train_losses.append(train_loss/ len(loader_train))

# save the model and the loss values
np.save('./CAE/train_losses.npy', np.array(train_losses))
torch.save(model.state_dict(), './CAE/CAE_model.pth')


# Train loss curve
import matplotlib.pyplot as plt
train_losses = np.load('./CW/CAE/train_losses.npy')
plt.plot(list(range(0,train_losses.shape[0])), train_losses)
plt.title('Train Loss')
plt.show()


# Test set reconstruction error
# load the model
model.load_state_dict(torch.load('./CAE/CAE_model.pth'))
model.eval()
test_loss = 0
with torch.no_grad():
    for i, data in enumerate(loader_test):
        img,_ = data
        img = img.to(device)
        recon_batch = model(img)
        test_loss += loss_function_CAE(recon_batch, img)
    # loss calculated over the whole test set
    test_loss /= len(loader_test.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))


# Test set images and reconstructions
# load the model
model.load_state_dict(torch.load('./CAE/CAE_model.pth'))
it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]

# visualize the original images of the last batch of the test set
img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
                range=None, scale_each=False, pad_value=0)
show(img)


with torch.no_grad():
    # visualize the reconstructed images of the last batch of test set
    recon_batch = model(fixed_input.to(device)).cpu()
    recon_batch = make_grid(denorm(recon_batch), nrow=8, padding=2, normalize=False,
                            range=None, scale_each=False, pad_value=0)
    show(recon_batch)

# Classification of latent variables
# Prepare Data

cae = CAE().to(device)
cae.load_state_dict(torch.load('./CAE/CAE_model.pth'))
cae.eval()

X_train = torch.zeros(len(loader_train.dataset),hidden_size)
y_train = torch.zeros(len(loader_train.dataset))
X_test = torch.zeros(len(loader_test.dataset),hidden_size)
y_test = torch.zeros(len(loader_test.dataset))

with torch.no_grad():
    k = 0
    for i, (img_train, c_train) in enumerate(loader_train):
        img_train = img_train.to(device)
        z_train = cae.encode(img_train)
        shp = z_train.shape
        X_train[k:(k+z_train.size(0)),:] = z_train.reshape((shp[0], shp[1]))
        y_train[k:(k+c_train.size(0))] = c_train
        k += z_train.size(0)

    l = 0
    for j, (img_test, c_test) in enumerate(loader_test):
        img_test = img_test.to(device)
        z_test = cae.encode(img_test)
        shp = z_test.shape
        X_test[l:(l+z_test.size(0)),:] = z_test.reshape((shp[0], shp[1]))
        y_test[l:(l+c_test.size(0))] = c_test
        l += z_test.size(0)

X_train = X_train[:k,:]
y_train = y_train[:k]

X_test = X_test[:l,:]
y_test = y_test[:l]


# First Attempt : Random Forest
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

print('1st attempt (Random Forest) Classification accuracy : {:.4f}'.format(accuracy))


# Second and Third attempts are a MLP and a ConvNet in classes
# Tune Linear Classifier
# Hyperparameters to tune
bounds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 10)}]

# GPyOpt procedure
bopt_lin = GPyOpt.methods.BayesianOptimization(train_lin,
                                           domain=bounds,
                                           model_type='GP_MCMC',
                                           acquisition_type='EI_MCMC',
                                           normalize_Y=True,
                                           n_samples=3)
max_iter = 3
bopt_lin.run_optimization(max_iter)

lr_opt_lin = pow(10, -float(bopt_lin.x_opt))
print('optimal learning rate:', lr_opt_lin)


# Tune Conv Classifier
bopt_conv = GPyOpt.methods.BayesianOptimization(train_conv,
                                           domain=bounds,
                                           model_type='GP_MCMC',
                                           acquisition_type='EI_MCMC',
                                           normalize_Y=True,
                                           n_samples=3)
max_iter = 3
bopt_conv.run_optimization(max_iter)

lr_opt_conv = pow(10, -float(bopt_conv.x_opt))
print('optimal learning rate:', lr_opt_conv)


# ##### Train Linear Classifier with optimal learning rate
# Initialise classifier
clf = LinClassifier().to(device)
# put CAE in eval mode
cae.eval()

# Optimiser
optimizer = torch.optim.Adam(clf.parameters(), lr=lr_opt_lin)

clf_train_losses = []
clf.train()
num_epochs=100
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
    clf_train_losses.append(clf_train_loss/ len(loader_train))

# save the model and the loss values
np.save('./CAE/lin_clf_train_losses.npy', np.array(clf_train_losses))
torch.save(clf.state_dict(), './CAE/LinCLF_model.pth')


# Train Conv Classifier with optimal learning rate
# Initialise classifier
clf = ConvClassifier().to(device)
# put CAE in eval mode
cae.eval()

# Optimiser
optimizer = torch.optim.Adam(clf.parameters(), lr=lr_opt_conv)

clf_train_losses = []
clf.train()
num_epochs=100
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
    clf_train_losses.append(clf_train_loss/ len(loader_train))

# save the model and the loss values
np.save('./CAE/conv_clf_train_losses.npy', np.array(clf_train_losses))
torch.save(clf.state_dict(), './CAE/ConvCLF_model.pth')


# Test both Classifiers
clf = LinClassifier().to(device)
cae = CAE().to(device)
clf.load_state_dict(torch.load('./CAE/LinCLF_model.pth'))
cae.load_state_dict(torch.load('CAE_model.pth'))
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


print('Classification accuracy MLP Classifier : {:.4f}'.format(test_acc))


clf = Classifier().to(device)
cae = CAE().to(device)
clf.load_state_dict(torch.load('./CAE/ConvCLF_model.pth'))
cae.load_state_dict(torch.load('CAE_model.pth'))
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

print('Classification accuracy ConvClassifier : {:.4f}'.format(test_acc))



##############################
# DCGAN
##############################
num_epochs = 200
learning_rate  = 2e-4
latent_vector_size = 100


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

use_weights_init = True

model_G = Generator(64).to(device)
if use_weights_init:
    model_G.apply(weights_init)
params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
print("Total number of parameters in Generator is: {}".format(params_G))
print(model_G)
print('\n')

model_D = Discriminator(32).to(device)
if use_weights_init:
    model_D.apply(weights_init)
params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
print("Total number of parameters in Discriminator is: {}".format(params_D))
print(model_D)
print('\n')

print("Total number of parameters is: {}".format(params_G + params_D))


# Define loss function
criterion = nn.BCELoss(reduction='mean')
def loss_function(out, label):
    loss = criterion(out, label)
    return loss


# Choose and initialize optimizers
beta1 = 0.5
optimizerD = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(beta1, 0.999))


# Define fixed input vectors to monitor training and mode collapse.
fixed_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
real_label = 1
fake_label = 0


# Train
export_folder = './DCGAN'
train_losses_G = []
train_losses_D = []

for epoch in range(num_epochs):
    for i, data in enumerate(loader_train, 0):
        train_loss_D = 0
        train_loss_G = 0

        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        model_D.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = model_D(real_cpu)
        errD_real = loss_function(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
        fake = model_G(noise)
        label.fill_(fake_label)
        output = model_D(fake.detach())
        errD_fake = loss_function(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        train_loss_D += errD.item()
        optimizerD.step()

        # Update G network: maximize log(D(G(z)))
        model_G.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = model_D(fake)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        train_loss_G += errG.item()
        optimizerG.step()


    if epoch == 0:
        save_image(denorm(real_cpu.cpu()), './DCGAN/real_samples.png')

    fake = model_G(fixed_noise)
    save_image(denorm(fake.cpu()), './DCGAN/fake_samples_epoch_%03d.png' % epoch)
    train_losses_D.append(train_loss_D / len(loader_train))
    train_losses_G.append(train_loss_G / len(loader_train))

    print('[%d/%d] Loss_D: %.4f, Loss_G: %.4f' % (epoch, num_epochs, errD.item(), errG.item()))

# save losses and models
np.save('./DCGAN/train_losses_D.npy', np.array(train_losses_D))
np.save('./DCGAN/train_losses_G.npy', np.array(train_losses_G))
torch.save(model_G.state_dict(), './DCGAN/DCGAN_model_G.pth')
torch.save(model_D.state_dict(), './DCGAN/DCGAN_model_D.pth')


# Generator samples
it = iter(loader_test)
sample_inputs, _ = next(it)
fixed_input = sample_inputs[0:32, :, :, :]

# visualize the original images of the last batch of the test set
img = make_grid(denorm(fixed_input), nrow=4, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
show(img)

# load the model
model_G.load_state_dict(torch.load('./DCGAN/DCGAN_model_G.pth'))
input_noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)

with torch.no_grad():
    # visualize the generated images
    generated = model_G(input_noise).cpu()
    generated = make_grid(denorm(generated)[:32], nrow=8, padding=2, normalize=False,
                        range=None, scale_each=False, pad_value=0)
    show(generated)


# Train losses curves
train_losses_D = np.load('./DCGAN/train_losses_D.npy')
train_losses_G = np.load('./DCGAN/train_losses_G.npy')
plt.plot(list(range(0,train_losses_D.shape[0])), train_losses_D, label='loss_D')
plt.plot(list(range(0,train_losses_G.shape[0])), train_losses_G, label='loss_G')
plt.legend()
plt.title('Train Losses')
plt.show()
