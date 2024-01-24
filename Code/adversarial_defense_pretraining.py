#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os, glob
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from dataset import *
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision import models as tm
from efficientnet_pytorch import EfficientNet
from model import *
from datetime import datetime
from torch.nn.parallel import DataParallel
import tqdm
import torchattacks
import random as rn
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter 
from utils import yaml_config_hook
from avagrad import *
from tqdm import trange
from skimage.feature import local_binary_pattern


##=============Config===================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
device='cuda'
cfg = yaml_config_hook("config.yaml")
now = datetime.now()
dirname =now.strftime("%d%m%Y%H%M%S")
if not os.path.isdir('runs/test'):
    os.mkdir('runs/test')
writer = SummaryWriter(os.path.join('runs', 'test', dirname))
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

### original inference process ###
print('Loading the imagenet-pretrained model? ', cfg.loading_imagenet_pretrained)
stmodel= tm.resnet50(pretrained=cfg.loading_imagenet_pretrained).to(device)
stmodel = DataParallel(stmodel)

if cfg.cnn_model_fn is not None:
    stmodel.load_state_dict(torch.load(cfg.cnn_model_fn))

val_dataset = dataset_DFD(0, cfg.val_fn, root=cfg.root, mode='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size_train//4, drop_last=False, 
                                         num_workers=16, sampler=None, shuffle=True)
    
atk = torchattacks.PGD(stmodel, eps=4/255, alpha=1/255, steps=200)

stmodel.eval()
print('Start validating')
t = trange(len(val_loader), desc='Average Val Accuracy', leave=True)
val_acc, val_acc2 = [], []
for step_val, (_,X,y) in zip(t, val_loader):
    X = X.to(device)
    y = y.to(device)

    ##======================
#     pred = stmodel(X)
#     _, pred = torch.max(pred, 1)
#     val_acc.append( (pred == y).sum().item()/y.size(0) )

    radius = rn.randint(1,3)
    n_points = 8 * radius

    adv_images = atk(X, y)
    adv_images = adv_images.permute(0,2,3,1).cpu().numpy()
    for b in range(adv_images.shape[0]):
        for c in range(3):
            adv_images[b,:,:,c] =local_binary_pattern(adv_images[b,:,:,c], n_points, radius, 'ror')
    adv_images[np.isnan(adv_images)] = 0
    adv_images = (adv_images / (2**n_points) -0.5) * 2
    adv_images = torch.Tensor(adv_images).permute(2,0,1)
    
    adv_images = torch.Tensor(adv_images).permute(0,3,1,2).cuda()
    
    outputs = stmodel(adv_images)

    _, pred = torch.max(outputs.data, 1)
    val_acc2.append( (pred == y).sum().item()/y.size(0) )



    t.set_description("Average Val Accuracy: %.5f/Robustness: %.5f" % (np.mean(np.array(val_acc)), np.mean(np.array(val_acc2))))
    t.refresh() # to show immediately the update
            
# opt_inner = torch.optim.Adam(imTrs.parameters(), lr=cfg.val_learning_rate)
# mse=torch.nn.SmoothL1Loss()
# kld=torch.nn.KLDivLoss()

## Measuring the robustness
# atk.save(data_loader=val_loader, save_path="./imagenet_val_pgd.pt", verbose=True)

# val_batch_size = 320
# del val_dataset, val_loader
# data = torch.load('./imagenet_val_pgd.pt')
# val_dataset = TensorDataset(data[0], data[1])
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size_train, drop_last=False, 
#                                      num_workers=24, sampler=None, shuffle=False)
# val_acc = {}
# for i in range(cfg.test_epoch_inner):
#     val_acc[i] = []

# total_step = len(val_loader)
# t = trange(total_step, desc='Average Robustness', leave=True)
# count = 0
# inner_count = 0
# for (i, (X,y)) in zip(t, val_loader):

#     X = X.to(device)
#     y = y.to(device)
    

#     ## Inner optimization====
#     running_loss = 0.
#     for inner_ep in range(cfg.test_epoch_inner):
#         opt_inner.zero_grad()
#         X_in,_ = imTrs(X, snr=cfg.val_SNR)
#         loss_inner = mse(X_in, X) #+ kld(X_in, X)
#         loss_inner.backward()
#         opt_inner.step()
        
#         with torch.no_grad():
#             X_in, Xn = imTrs(X, snr=cfg.val_SNR)
#             pred = stmodel(X_in.detach())
#             _, pred = torch.max(pred, 1)
#             val_acc[inner_ep].append( (pred == y).sum().item()/y.size(0))
#             writer.add_scalar('Test/inner_loss', loss_inner.item(), inner_count)
#             inner_count +=1
#             running_loss+=loss_inner.item()

#         t.set_description("Average Robustness: %.5f" % np.mean(np.array(val_acc[inner_ep])))
#         t.refresh() # to show immediately the update
        
#     writer.add_scalar('Test/running_robustness', np.mean(np.array(val_acc[inner_ep])), count)
#     writer.add_scalar('Test/running_loss', running_loss, count)
#     writer.add_image('Test/attacked_image', normalization(X[0]), count)
#     writer.add_image('Test/attacked_noised_image', normalization(Xn[0].detach()), count)
#     writer.add_image('Test/output_image', normalization(X_in[0].detach()), count)
#     count +=1
# #     denoise_step -=1 if denoise_step>1 else denoise_step
#     ##======================

# now = datetime.now()
# d1 =now.strftime("%d/%m/%Y %H:%M:%S")
# for ind, key in enumerate(list(val_acc.keys())):
#     acc = np.mean(np.array(val_acc[key]))
#     print("[%s] Validation Accuracy=%.5f" % (d1, acc))
#     writer.add_scalar('Test/Final_Robustness', acc, ind)



