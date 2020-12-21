import torch
from model import *
import argparse

parser = argparse.ArgumentParser(description='reload pretrained model for different torch version')
parser.add_argument('--model_path', type=str, default='', help='path of pretrained models')
parser.add_argument('--save_path', type=str, default='', help='path of pretrained models')
parser.add_argument('--resnet', type=str, default='50', help='resnet model')
parser.add_argument('--change_key', type=bool, default=False, help='whether change key for detection')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))

if args.resnet == '101':
    net=ResNet101().to(device)
if args.resnet == '50':
    net=ResNet50().to(device)


# net.load_state_dict(checkpoint, strict=False)


if args.change_key:
    # new_dict = {}
    # for k, v in checkpoint.items():
    #     old_k = k
    #     if 'shortcut' in k:
    #         k.replace('shortcut','downsample')
    old_keys = [key for key in checkpoint.keys() if 'shortcut' in key]
    new_keys = [key.replace('shortcut','downsample') for key in old_keys]
    for k in range(len(old_keys)):
        checkpoint[new_keys[k]]=checkpoint[old_keys[k]]
        checkpoint.pop(old_keys[k])
            # print(old_k, "->", k)
            # new_dict[k]=v


# net.load_state_dict(checkpoint)
torch.save(checkpoint, args.save_path, _use_new_zipfile_serialization=False)
print('Model transformed')
