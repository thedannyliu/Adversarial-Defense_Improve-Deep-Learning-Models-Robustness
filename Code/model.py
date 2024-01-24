import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViT
from resnet import *
from torchvision import models
from pdb import set_trace as sts
from gmlp import gMLPVision as mlp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

    
    
class SAFE_Transformer(nn.Module):
    def __init__(self, n_classes=1000, bksize = 14, pretrained=True):
        super(SAFE_Transformer, self).__init__()
 
        model= models.resnet50(pretrained=pretrained).cuda()
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
#         self.model2 = torch.nn.Sequential(*list(model.children())[:-4])
        
               
        self.reshape =  Rearrange('b c h w -> b (h w) c')
        self.reshape2 =  Rearrange('b c h w -> b c (h w)')
        self.projector = nn.Sequential(nn.Linear(2048+49, n_classes))
#         self.up = nn.PixelShuffle(4)
        featSize = [28, 28]
        num_patches = 28**2*2
        
        self.scale_transformer = mlp(num_patches=49, # Feat map size
                                dim = 2048,
                                depth = 2
                                    )
        
        self.spatial_transformer = mlp(num_patches=2048, # Feat map size
                                dim = 49,
                                depth = 2
                                      )
                                       
                                       
#         self.scale_transformer = ViT(num_patches=num_patches, # Feat map size
#                                 dim = 1024,
#                                 depth = 2,
#                                 heads = 8,
#                                 mlp_dim = 1024,
#                                 dropout = 0.1,
#                                 emb_dropout = 0.1)
        
#         self.spatial_transformer = ViT(num_patches=1024, # Feat map size
#                                 dim = 28**2*2,
#                                 depth = 2,
#                                 heads = 8,
#                                 mlp_dim = 1024,
#                                 dropout = 0.1,
#                                 emb_dropout = 0.1)
            
    
    def forward(self, x):
        x = self.model(x)

        x1 = self.reshape(x) ## 128x28x28, 512x28x28
        
        x1 = self.scale_transformer(x1) ## B*1024
        
        x = self.reshape2(x)
#         print(x.shape)
        x = self.spatial_transformer(x) ## B*784*2
#         print(x.shape)
        x = torch.cat((x,x1), 1) # B*1808
#         print(x1.shape)
        x = self.projector(x)
        return x
    