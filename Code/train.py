import os
import torch
import random as rn
import torchvision.transforms as transforms
from tqdm import trange
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler

from model import *
from resnet import *
from dataset import *
from ranger import Ranger
from utils import yaml_config_hook

# Initialize distributed process group
torch.distributed.init_process_group(backend="nccl")
torch.backends.cudnn.benchmark = True

##=============Config===================
# dd/mm/YY
# Load configuration file
cfg = yaml_config_hook("config.yaml")
##=====================================
# Set local rank
local_rank = int(os.environ["LOCAL_RANK"])
print('local_rank: ', local_rank)

# Set device
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

# Create checkpoint and Tensorboard directories
if not os.path.isdir('runs/train') and local_rank == 0:
    os.mkdir('runs/train')

if local_rank == 0:
    if not os.path.isdir(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)
    now = datetime.now()
    dirname = now.strftime("%d%m%Y%H%M%S")
    writer = SummaryWriter(os.path.join('runs', 'train', dirname))

# Load pre-trained CNN model
cfg.cnn_model_fn_pretrain = os.path.join(cfg.checkpoint_dir, cfg.cnn_model_fn_pretrain)
#========================================
stmodel = models.resnet50().to(device)

if os.path.isfile(cfg.cnn_model_fn_pretrain): 
    print('Found the pretrained weights on', cfg.cnn_model_fn_pretrain)
    stmodel.load_state_dict({k.replace('module.',''):v for k,v in torch.load(cfg.cnn_model_fn_pretrain).items()})

stmodel = DDP(stmodel, device_ids=[local_rank])

opt = torch.optim.AdamW(stmodel.parameters(), lr=cfg.learning_rate, betas=(0.95, 0.999))
scheduler = StepLR(opt, step_size=cfg.step, gamma=0.5)

train_dataset = FeatDataset(local_rank, cfg.train_fn, root=cfg.root, mode='train', feat='sift')
train_sampler = DistributedSampler(train_dataset,  shuffle=True, rank=local_rank)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=1, pin_memory=True, sampler=train_sampler)

val_dataset = FeatDataset(local_rank, cfg.val_fn, root=cfg.root, mode='val', feat='sift')
val_sampler = DistributedSampler(val_dataset,  shuffle=False, rank=local_rank)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, drop_last=False, 
                                         num_workers=1, sampler=val_sampler, shuffle=False, pin_memory=True)

ce = torch.nn.CrossEntropyLoss()

count = 0
count = 3 * len(train_loader)
####### Pretrain backbone ###################     
for ep in range(cfg.epoch):
    ep_loss = 0
    for step, (X,y) in enumerate(train_loader):
        count += 1
        train_sampler.set_epoch(ep)
        X = X.to(device)
        y = y.to(device)
       
        opt.zero_grad()
        lr = opt.param_groups[0]["lr"]
        
        pred = stmodel(X)
        loss = ce(pred, y)
        
        loss.backward()
        opt.step()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss = loss.data.clone()
            torch.distributed.all_reduce(loss.div_(torch.distributed.get_world_size()))
            
        loss = loss.item()
        ep_loss += loss
        
        #===========Summary writer=========================
        if local_rank==0:
            writer.add_scalar("Training/train_loss", loss, count)
            # writer.add_scalar("Training/train_inner_loss", loss_inner, count)
            writer.add_scalar("Training/learning_rate", lr, count)
            
       
        #===========Training set evaluation================
        if step % cfg.display_interval==0 and local_rank==0:
            _, pred = torch.max(pred, 1)
            correct = (pred == y).sum().item()/y.size(0)
            
            now = datetime.now()
            d1 =now.strftime("%d/%m/%Y %H:%M:%S")
            print("[%s][step][epoch][%5d/%5d][%3d/%3d] Loss=%.5f, Accuracy=%.5f, LR=%f" % (d1, step, len(train_loader), ep+1, cfg.epoch, loss, correct, lr))
            writer.add_scalar("Training/accuracy", correct, count)
            
                
    ##=======Validation set evaluation=================
    stmodel.eval()
    val_acc, test_acc = [], []
    if local_rank==0:
        print('Start validating')

    total_step = len(val_loader)
    if local_rank==0:
        t = trange(total_step, desc='Average Val Accuracy', leave=True)
    else:
        t = range(total_step)
    for step_val, (X,_,y) in zip(t, val_loader):
        X = X.to(device)
        y = y.to(device)

        ##======================
        with torch.no_grad():
            pred = stmodel(X)
            _, pred = torch.max(pred, 1)
            val_acc.append( (pred == y).sum().item()/y.size(0) )

            if local_rank==0:
                t.set_description("Average Val Accuracy: %.5f" % (np.mean(np.array(val_acc))))
                t.refresh() # to show immediately the update



    val_acc = np.mean(np.array(val_acc))
    stmodel.train()
    if local_rank==0:
        now = datetime.now()
        d1 =now.strftime("%d/%m/%Y %H:%M:%S")
        print("[%s][step/epoch][%5d/%3d] Validation Accuracy=%.5f" % (d1, step, ep+1, val_acc))
        writer.add_scalar("Validation/accuracy", val_acc, count)
                
            
    if ep% cfg.save_interval==0 and local_rank==0:
        torch.save(stmodel.state_dict(), '%s/ckpt_%d.pt' % (cfg.checkpoint_dir, ep))
        
    scheduler.step()
    if local_rank==0:
        writer.add_scalar("Training/Running_loss", ep_loss / len(train_loader), ep)
        

# save your improved network
if local_rank==0:
    torch.save(stmodel.state_dict(), os.path.join(cfg.checkpoint_dir, cfg.cnn_model_fn))

