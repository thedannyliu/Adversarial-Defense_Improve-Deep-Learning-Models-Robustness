#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os, glob
from torchvision import models
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

##=============Config===================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device='cuda'
now = datetime.now()
# dd/mm/YY
val_batch_size=128
root = "/ssd3/ilsvrc2012/"
val_fn = root+'val.txt'
PREGENERATE = False
LOG_FN = 'log_naiv.txt'
writer = SummaryWriter('runs/'+LOG_FN.replace('.txt',''))


### original inference process ###

stmodel= tm.resnet101(pretrained=True).to(device)
imTrs = imageTrans(device).to(device)
stmodel = DataParallel(stmodel)
imTrs = DataParallel(imTrs)

stmodel.load_state_dict(torch.load('checkpoint/final_net.pt'))
imTrs.load_state_dict(torch.load('checkpoint/final_imt.pt'))

if PREGENERATE:
    val_dataset = dataset_DFD(0, val_fn, root=root+'val/', mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, drop_last=False, 
                                             num_workers=16, sampler=None, shuffle=False)

    atk = torchattacks.PGD(nn.Sequential(imTrs,stmodel), eps=4/255, alpha=1/255, steps=200)
    atk.save(data_loader=val_loader, save_path="./imagenet_val_pgd.pt", verbose=True)
else:

    val_batch_size = 960
    data = torch.load('./imagenet_val_pgd.pt')
    val_dataset = TensorDataset(data[0], data[1])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, drop_last=False, 
                                         num_workers=24, sampler=None, shuffle=False)

stmodel.eval()
imTrs.train()
val_acc = 0
print('Start validating')
opt_inner = torch.optim.AdamW(imTrs.parameters(), lr=3e-3)
mse=torch.nn.MSELoss().type(torch.cuda.FloatTensor)

val_acc = {}
denoise_step=100
for i in range(denoise_step):
    val_acc[i] = []

total_step = len(val_loader)
t = trange(total_step, desc='Average Robustness', leave=True)

for (i, (X,y)) in zip(t, val_loader):

    X = X.to(device)
    y = y.to(device)
    

    ## Inner optimization====
    
    for inner_ep in range(denoise_step):
        opt_inner.zero_grad()
        X_in = imTrs(X, snr=30).to(device)
        loss_inner = mse(X_in, X)
        loss_inner.backward()
        opt_inner.step()
        
        with torch.no_grad():
            pred = stmodel(imTrs(X_in).detach())
            _, pred = torch.max(pred, 1)
            val_acc[inner_ep].append( (pred == y).sum().item()/y.size(0))
            writer.add_scalar('ImTrs.inner_loss', i+ (i*denoise_step))
            
    t.set_description("Bar desc (acc %.5f)" % np.mean(np.array(val_acc[inner_ep])))
    t.refresh() # to show immediately the update
    writer.add_scalar('ImTrs.avg_loss', loss_inner.item(), i*denoise_step)
    ##======================

now = datetime.now()
d1 =now.strftime("%d/%m/%Y %H:%M:%S")
with open(LOG_FN, 'w') as fp:
    for ind, key in enumerate(list(val_acc.keys())):
        acc = np.mean(np.array(val_acc[key]))
        print("[%s] Validation Accuracy=%.5f" % (d1, acc))
        fp.write("[%s] Validation Accuracy=%.5f\n" % (d1, acc))
        writer.add_scalar('Robustness', acc, ind)



