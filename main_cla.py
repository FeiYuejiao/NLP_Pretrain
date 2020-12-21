'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle
import torchvision
import torchvision.transforms as transforms
from build_vocab import Vocabulary
import numpy as np
import os
import argparse
from model import *
from Decoder import *
from data_loader import get_loader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

linear = nn.Linear(10, 256).to(device)
bn = nn.BatchNorm1d(256, momentum=0.01).to(device)




# Training
def train(epoch,net_main,net,linear_cla,optimizer,criterion,trainloader,writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_step=len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs= net(inputs)#net_main(inputs)+ #torch.cat((net_main(inputs),net(inputs)),dim=1)
        outputs = F.avg_pool2d(outputs, 4)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = linear_cla(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        writer.add_scalar('Loss/train', loss,epoch*total_step+batch_idx)
        if batch_idx % 10 == 0:
            print('CIFAR-Epoch [{}], Step [{}], Loss: {:.8f}'
                  .format(epoch, batch_idx, loss.item()))
    return net


def test(epoch,net_main,net,linear_cla,testloader,writer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)#net_main(inputs)+ net(inputs)#torch.cat((net_main(inputs), net(inputs)), dim=1)
            outputs = F.avg_pool2d(outputs, 4)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = linear_cla(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('Accuracy/test', acc, epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(os.path.join('checkpoint', args.save_path)):
            os.mkdir(os.path.join('checkpoint', args.save_path))
        torch.save(state, os.path.join('checkpoint', args.save_path,'model.pt'))
        print('model saved in', os.path.join('checkpoint', args.save_path,'model.pt'))
        best_acc = acc
    print('acc:', acc, '  Best acc:', best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--model_path', type=str, default='', help='path of pretrained models')
    parser.add_argument('--save_path', type=str, default='', help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=32, help='size for randomly cropping images')
    # parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    # parser.add_argument('--save_step', type=int, default=1500, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)#0.001
    parser.add_argument('--loading', type=bool, default=False)
    args = parser.parse_args()
    # print(args)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')


    # Build the Decoder models
    #decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    net = ResNet101().to(device)
    #net_main = ResNet34().to(device)

    linear_cla = nn.Linear(2048,100).to(device)
    #net.load_state_dict(torch.load('./models/IC_models/encoder-400-2000.ckpt'))

    if args.loading:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    # Optimizer for image classificaation
    optimizer = optim.Adam(list(net.parameters())+list(linear_cla.parameters()), args.learning_rate)#, momentum=0.9, weight_decay=5e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=0.1)
    writer = SummaryWriter()
    #start training
    for epoch in range(0, args.num_epochs):
        encoder = train(epoch,net_main=None,net=net,linear_cla=linear_cla,optimizer=optimizer,criterion=criterion,trainloader=trainloader,writer=writer)
        # scheduler.step()
        test(epoch,net_main=None,net=encoder,linear_cla=linear_cla,testloader=testloader,writer=writer)
