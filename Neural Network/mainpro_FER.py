'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *

##Vincent: I made comments by using double hashtags in the code. 
##PyTorch is a really nice library that enables us to use the power of our GPU (Graphical Porcessing Unit) for matrix calculations (which are available in plenty amounts for machine learning)
##(Usually, codes only use the CPU (Central Processing Unit)), so this boosts our available memory!

##Parser library is used to create command-line interface with the user, to tweak parameters without going into the code (defaults are already set).
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

## Learning rate is a factor that determines how large of a change (also called 'step') the algorithm takes based off of its learning experience.
## There is a decay of this learning rate, as it is assumed that the more trainings have elapsed, the closer to the ultimate configuration the algorithm gets.
## Initially, the algorithm is very dumb --> We want it to change a lot based on its input. However, when it gets closer to the configuration, we do not want
## A single wrong classification to drastically change the architecture of the Neural Network again.
## Therefore, the learning rate decreases with every training.
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
## Here, the training data is 'augmented': During training, when we grab a couple of pictures for training, we can sometimes flip them horizontally or crop some of them, to have more data to train over.
## The image still holds the same information, but as it is adjusted in orientation, we now are less likely to risk overfitting. This effectively increases the size of our dataset aswell.
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

## FER2013 is a class in fer.py. It now creates a training and a test set from the entire database.
trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19') ##Visual Geometry Group (VGG) is a deep CNN, consisting of 19 layers. It was made in Oxford especially to classify images.
    ## It contains multiple 3x3 convolution layers (each node is connected to a 3x3 grid of previous nodes, based off of the idea that features in images are local)
    ## There are also a couple of fully connected layers and maxpool layers, the latter of which is meant to downsize the amount of features and create a more robust CNN (less dependant on local translation
    ## of the image). Finally, there is a 'softmax' layer, which always has to be used for classification tasks, as it ensures our output is summed to 1 (100%), and the outputs give the confidence of the CNN
    ## That the image is a certain class.
elif opt.model  == 'Resnet18': ## Usually, more layers =/= better performance. There is even a size at which it starts work badly. However, ResNet works around this problem for ~100 - 200 layers 
    ## (>20.000 citations on the original paper!)
    ## This is quite an advanced architecture, and more information about it will come available in the Google Drive folder. Although it can be way deeper than VGG19, its floating operating points calculations
    ## Are significantly lower --> Less strain on the computational power. State-of-the-art in 2015.
    net = ResNet18()

if opt.resume:
    # Load checkpoint. (If earlier learning had already been done)
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda() ##Cuda enables Graphical Processing Unit (GPU)-accelerated computation for linear algebra problems --> Faster computations.

criterion = nn.CrossEntropyLoss() ##This is a loss function, which can be seen as our 'punishment function': It gives a score based off of how far the NN's guess is off.
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4) ##Stochastic Gradient Descent: an optimization function, 
##                                                                                    which tells the NN in what way it should change to minimize the CrossEntropyLoss score.

# Training
##An 'ephoch' is like a training trail: The more epochs, the more cycles of training are being performed.
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc ##Global variable to store training accuracy.
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    ## learning rate decay tells us how fast we decrease our advancements in learning from our mistakes. As noted previously, it is ideal that the algorithm does not too stay suspicious for too long, as it
    ## might result in drastic changes which actually worsen the accuracy! As the algorithm becomes more advanced and knowledgable, it should trust its decisions more.
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader): ##A 'batch' is the amount of pictures that is handled each epoch.
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad() ##Don't worry about optimizers for now...
        inputs, targets = Variable(inputs), Variable(targets) ##Grab the inputs and labels (targets)...
        outputs = net(inputs) ##Based on the input, determine the output of the algorithm...
        loss = criterion(outputs, targets) ## Calculate the 'loss' (how far are we off) based on the difference between the input and the output...
        loss.backward() ##Perform backwards propegation: Based on the loss, it can be determined what should be changed in the weights and biases of the algorithm.
        utils.clip_gradient(optimizer, 0.1) ##Calculate the 'gradient': This explains in which 'direction' the weights should be changed to minimize loss (according to the calculations from the current point)
        ##                                    Moreover, it contains information about how large the effect of a small change in the individual weights and biases would presumably be on the loss.
        optimizer.step() ##Based on the gradient and using the optimizer algorithm, a 'step' is taken. This implies that all weights and biases are changed in such a way as to minimize the loss based on the gradient.
        ##                 The gradient only tells you something about the direction and magnitude of change IN A POINT --> this step is based on extrapolation. You might be able to see that the step size
        ##                 (which is dependant on the learning rate and) therefore tells us something about how much we dare to extrapolate in a certain point in the process, and why we lower this step size
        ##                 with increasing amounts of epochs.
        train_loss += loss.data[0] ##Add the loss value of this run to the total loss during training
        _, predicted = torch.max(outputs.data, 1) ##??
        total += targets.size(0) ##??
        correct += predicted.eq(targets.data).cpu().sum() ##??

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' ##Show a progression bar that also indicated the current loss and average accuracy of the system.
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total

def PublicTest(epoch): ##After each epoch, a small test on TRAINING DATA is performed to see if the 'step' that was made actually improves our score.
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs) ##Get some variables from the shape of the input vector
        inputs = inputs.view(-1, c, h, w) ##No clue what happens here
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() ##Use the GPU
        inputs, targets = Variable(inputs, volatile=True), Variable(targets) ##Gather the inputs and labels
        outputs = net(inputs) ##Calculate the outputs of the neural network
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets) ##Calculate loss... etc (same as with each epoch, except we do not take a 'step')
        PublicTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch): ##Finally, to crossvalidate, another test is performed on the testing dataset, to see if the algorithm has also improved its accuracy on never seen before data. (Same idea as the public test)
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data[0]
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total

    if PrivateTest_acc > best_PrivateTest_acc: ##If we have a new best score on the test, save the current configuration of weights and biases! Else, disregard it.
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch): ##For every epoch (this is the actual entire algorithm in terms of what is read by the computer on the surface)
    train(epoch) ##Train
    PublicTest(epoch) ##Test on training data
    PrivateTest(epoch) ##Test on test data.

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
