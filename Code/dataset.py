import os
import cv2
import tqdm
import torch
import numpy as np
import albumentations as A
from os.path import join as osj
from torchvision.transforms import Normalize, ToTensor
from albumentations.pytorch.transforms import ToTensorV2
from kornia.feature import DenseSIFTDescriptor
from sklearn.decomposition import PCA
radius = 1.0

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class DenseSIFTDescriptorTransform(object): # output [128, 224, 224]
    def __init__(self):
        self.SIFT = DenseSIFTDescriptor()

    def __call__(self, image, **kwargs):
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        descs = self.SIFT(image_tensor)
        descs = descs.squeeze(0)
        return descs.detach()
    
class PcaTransform(object): # output [16, 224, 224]
    def __init__(self, n_components=16):
        self.n_components = n_components

    def __call__(self, descriptor, **kwargs):
        descriptor_np = descriptor.detach().numpy()
        descriptor_reshaped = np.transpose(descriptor_np, (1, 2, 0))
        descriptor_flattened = descriptor_reshaped.reshape(-1, descriptor_reshaped.shape[2])

        pca = PCA(n_components=self.n_components)
        descriptor_transformed = pca.fit_transform(descriptor_flattened)
        descriptor_transformed = descriptor_transformed.reshape(descriptor_reshaped.shape[0], descriptor_reshaped.shape[1], self.n_components)

        return descriptor_transformed
    
class PCA(object):
    def __init__(self, q=3, niter=2):
        self.q = q
        self.niter = niter

    def __call__(self, desc, **kwargs):
        desc_whd = desc.permute(1, 2, 0)
        desc_mxn = desc_whd.reshape(-1, desc_whd.shape[2])
        desc_pca = torch.pca_lowrank(desc_mxn, q=self.q, niter=self.niter)
        desc_out = desc_pca[0].detach().reshape(desc_whd.shape[0], desc_whd.shape[1], self.q)
        desc_dwh = desc_out.permute(2, 0, 1)
        desc_result = -1 + 2 * (desc_dwh - torch.min(desc_dwh)) / (torch.max(desc_dwh) - torch.min(desc_dwh))
        # print(desc_result)
        return desc_result


def has_viscide(s11, thr=3):
    s11[s11 > thr] = 1024
    s11[s11 < -thr] = 2048
    s11[s11 < 1024] = 0
    return s11

def get_features(img, img_height, img_width, thr=3):
    img2 = torch.nn.ZeroPad2d((1, 0, 1, 0))(img[:, :img_height - 1, :img_width - 1])
    img3 = torch.nn.ZeroPad2d((0, 1, 0, 1))(img[:, 1:, 1:])
    img4 = torch.nn.ZeroPad2d((1, 0, 0, 1))(img[:, :img_height - 1, 1:])
    img5 = torch.nn.ZeroPad2d((0, 1, 1, 0))(img[:, 1:, :img_width - 1])

    s11, s12 = img2 - img, img - img3
    s31, s32 = img4 - img, img - img5

    s11 = has_viscide(s11, thr=thr)
    s12 = has_viscide(s12, thr=thr)
    s31 = has_viscide(s31, thr=thr)
    s32 = has_viscide(s32, thr=thr)

    s11 = s11 + s12
    s31 = s31 + s32

    s11[s11 == 1024] = 2
    s11[s11 == 2048] = 1

    s31[s31 == 1024] = 2
    s31[s31 == 2048] = 1
    s31 *= 3
    res1 = s11 + s31
    return res1.permute(1, 2, 0).cpu().numpy()

class FeatDataset(torch.utils.data.Dataset):
    
    def __init__(self, local_rank, filename, mode='train', root="", feat='sift'):
        super(FeatDataset, self).__init__()

        self.mode = mode
        self.root = root
        self.feat = feat
        self.local_rank = local_rank
        self.fns = []
        self.y = []
        
        filename = osj(root, filename)
        
        with open(filename, 'r') as fp:
            data = fp.readlines()
            for line in tqdm.tqdm(data, total=len(data)):
                fn, lab = line.strip('\n').split(' ')
                fn = os.path.join(root, mode, fn)
                self.fns.append(fn)
                self.y.append(int(lab))
        self.val_sift = A.Compose([
            A.Lambda(name='sift', image=DenseSIFTDescriptorTransform(), p=1, always_apply=True),
            # A.Lambda(name='pca', image=PCA(), p=1, always_apply=True),
            # A.Lambda(name='pca', image=PcaTransform(), p=1, always_apply=True),
            # A.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0], max_pixel_value=1.0, always_apply=True, p=1.0),
            # ToTensorV2()
        ])
        self.train_transform = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=224, width=224),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ])

        self.val_transform = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224)
        ])
        self.n_images = len(self.fns)
        print('The # of images is:', self.n_images, 'on', self.mode, 'mode!')
        
        
    def __getitem__(self, index):
        fn = self.fns[index]
        lab = self.y[index]
        
        img = cv2.imread(fn)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            if self.feat is None:
                rgb = rgb / 255.0
                rgb = torch.Tensor(rgb).permute(2, 0, 1)
                return rgb, lab

            rgb = self.train_transform(image=rgb)['image']
            
            if self.feat == 'sift':
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                desc = self.val_sift(image=gray)['image']
                # print("desc MAX:", torch.max(desc), "desc min:", torch.min(desc),  "2:", torch.max(torch.from_numpy(gray)), torch.min(torch.from_numpy(gray)), "3:", lab)
            return desc, lab
        else:
            if self.feat is None:
                rgb = rgb / 255.0
                rgb = torch.Tensor(rgb).permute(2, 0, 1)
                return rgb, lab

            rgb = self.val_transform(image=rgb)['image']

            if self.feat == 'sift':
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                desc = self.val_sift(image=gray)['image']
                
            rgb = torch.Tensor(rgb).permute(2, 0, 1)
            return desc, rgb, lab


    def __len__(self):
        return self.n_images
    
# a, b, c = FeatDataset(0, "train.txt", root="/hdd3/ILSVRC/Data/imagenet", mode='train', feat='sift')