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
import random
import warnings
import argparse
from model import *
from Decoder import *
from data_loader import CocoDataset, collate_fn
from torch.utils.tensorboard import SummaryWriter

from utils import Config
from torch.optim.lr_scheduler import MultiStepLR
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torchvision.models import ResNet
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CoCo Training')
    parser.add_argument('config', type=str, default='configs/baseline.py',
                        help='train config file path')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/train2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')


    args = parser.parse_args()
    return args


# Training
def train_ic(epoch,cfg,net,decoder,linear,bn,optimizer_ic,criterion,data_loader,writer):
    caption_ite = 0
    net.train()
    #build the frontend
    decoder.train()
    linear.train()
    bn.train()
    total_step = len(data_loader)
    print('Train Image Caption.')
    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        targets = targets.to(device)
        decoder.zero_grad()
        net.zero_grad()
        features = F.adaptive_avg_pool2d(net(images), 1)
        features = features.view(features.size(0), -1)
        features = bn(linear(features))
        outputs = decoder(features, captions, lengths)

        loss_ic = criterion(outputs, targets)
        loss_ic.backward()
        optimizer_ic.step()

        writer.add_scalar('IC_Loss/train', loss_ic, epoch * total_step + i)
        writer.add_scalar('Perplexity/train', np.exp(loss_ic.item()), epoch * total_step + i)

        # Print log info
        if i % cfg.log_step == 0:
            print('IC-Epoch [{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, len(data_loader), i, loss_ic.item(), np.exp(loss_ic.item())))

        # Save the model checkpoints
        if (i + 1) % cfg.save_step == 0:
            if not os.path.exists(cfg.model_path):
                os.makedirs(cfg.model_path)
            torch.save(decoder.state_dict(), os.path.join(
                cfg.model_path, 'new_decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)), _use_new_zipfile_serialization=False)
            torch.save(net.state_dict(), os.path.join(
                cfg.model_path, 'new_encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)), _use_new_zipfile_serialization=False)
    return net





def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.config = args.config
    print(cfg)
    if cfg.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    # cfg.distributed = cfg.world_size > 1

    ngpus_per_node = torch.cuda.device_count()
    main_worker(cfg.gpu, ngpus_per_node, cfg)



def main_worker(gpu, ngpus_per_node, cfg):
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        print('init distributing process')
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    # Data
    print('==> Preparing data..')
    # Load vocabulary wrapper for image caption
    with open(cfg.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Image preprocessing, normalization for the pretrained resnet
    # cifar cls, use resized 36x36 image
    if cfg.task == 'cifar_cls':
        transform = transforms.Compose([
            transforms.RandomCrop(cfg.crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    # imagenet cls, 224x224
    # same as MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
    if cfg.task == 'imagenet_cls':
        transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    # coco det, 1333x800
    # same as MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
    if cfg.task == 'coco_det':
        transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])


    # COCO caption dataset
    coco = CocoDataset(root=cfg.image_dir,
                       json=cfg.caption_path,
                       vocab=vocab,
                       transform=transform)
    #Build data loader for image caption training
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(coco)
    else:
        train_sampler = None

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=cfg.batch_size,
                                              shuffle=(train_sampler is None),
                                              num_workers=cfg.num_workers,
                                              collate_fn=collate_fn,
                                              pin_memory=True,
                                              sampler=train_sampler)




    # Build the Decoder models
    decoder = DecoderRNN(cfg.model['embed_size'], cfg.model['hidden_size'], len(vocab), cfg.model['num_layers'])


    if cfg.model['net'] == 'densenet121':
        linear_ic = nn.Linear(1024, 256)
        bn_ic = nn.BatchNorm1d(256, momentum=0.01)
        net = DenseNet121()

    if cfg.model['net'] == 'densenet169':
        linear_ic = nn.Linear(4096, 256)
        bn_ic = nn.BatchNorm1d(256, momentum=0.01)
        net = DenseNet169()

    if cfg.model['net'] == 'resnet34':
        linear_ic = nn.Linear(512, 256)
        bn_ic = nn.BatchNorm1d(256, momentum=0.01)
        net = ResNet34()

    if cfg.model['net'] == 'resnet50':
        linear_ic = nn.Linear(2048, 256)
        bn_ic = nn.BatchNorm1d(256, momentum=0.01)
        net = ResNet50()

    if cfg.model['net'] == 'resnet101':
        linear_ic = nn.Linear(2048, 256)
        bn_ic = nn.BatchNorm1d(256, momentum=0.01)
        net = ResNet101()

    print('cfg.distributed:', cfg.distributed)
    if cfg.distributed:
        linear_ic.cuda()
        bn_ic.cuda()
        net.cuda()
        decoder.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        linear_ic = torch.nn.parallel.DistributedDataParallel(linear_ic)
        bn_ic = torch.nn.parallel.DistributedDataParallel(bn_ic)
        net = torch.nn.parallel.DistributedDataParallel(net)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder)
    else:
        torch.cuda.set_device(device)
        linear_ic.cuda(cfg.gpu)
        bn_ic.cuda(cfg.gpu)
        net.cuda(cfg.gpu)
        decoder.cuda(cfg.gpu)

    criterion = nn.CrossEntropyLoss()
    # Optimizer for image classificaation
    # optimizer = optim.Adam(list(net.parameters()), lr=cfg.lr)

    optimizer_ic = optim.Adam(list(net.parameters()) + list(linear_ic.parameters())+ list(decoder.parameters())+list(bn_ic.parameters()), lr=cfg.lr) #0.0001
    scheduler = MultiStepLR(optimizer_ic, milestones=[60,120,160], gamma=0.1)


    if cfg.loading:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir(cfg.checkpoint), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(cfg.checkpoint)
        net.load_state_dict(checkpoint)
        # best_acc = checkpoint['acc']
        start_epoch = int(cfg.checkpoint.split('/')[-1].split('-')[1])
    else:
        start_epoch = 0


    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ic, T_max=200)
    log_dir = 'log/' + cfg.config.split('/')[1][:-3]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    #start training
    for epoch in range(start_epoch,cfg.num_epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        net=train_ic(epoch, cfg, net=net, decoder=decoder,linear=linear_ic,bn=bn_ic, optimizer_ic=optimizer_ic,criterion=criterion,data_loader=data_loader,writer=writer)
        scheduler.step()

if __name__ == '__main__':
    main()
