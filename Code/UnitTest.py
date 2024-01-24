import cv2
import glob
import tqdm
import numpy as np
import random as rn
import torch, os, pdb
import albumentations as A
import torch.multiprocessing
import os, glob, numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from os.path import join as osj
from albumentations.pytorch.transforms import ToTensorV2
from skimage.feature import local_binary_pattern, hog
from hog import *


from kornia.feature import DenseSIFTDescriptor
from sklearn.decomposition import PCA

# with open('/mnt/sdd/ilsvrc2012/train.txt', 'r') as fp:
#     data = fp.readlines()
#     for line in tqdm.tqdm(data, total=len(data)):
#         fn, lab = line.strip('\n').split(' ')
#         print(fn, lab)

# import cv2
# img = cv2.imread('../../sdd/ilsvrc2012/train/n04069434/n04069434_13325.JPEG')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# sift = cv2.SIFT_create()
# keypoints = [cv2.KeyPoint(x, y, 10) 
#                         for y in range(0, im.shape[0], 20)
#                         for x in range(0, im.shape[1], 20)]
# print(type(keypoints))
# points, descriptors = sift.compute(im, keypoints)

# m2 = torch.nn.ZeroPad2d((0, 1, 0, 1))
# m1 = torch.nn.ZeroPad2d((1, 0, 1, 0))
# m4 = torch.nn.ZeroPad2d((1, 0, 0, 1))
# m3 = torch.nn.ZeroPad2d((0, 1, 1, 0))

# def hasviside(s11, thr=3):
#     s11[s11>thr] = 1024
#     s11[s11<-thr] = 2048
#     s11[s11<1024] = 0
#     return s11

# def get_features(img, img_height=435, img_width=468, thr=3):

   
#     img2 = m1(img[:,:img_height-1, :img_width-1])
#     img3 = m2(img[:,1:, 1:])
#     img4 = m3(img[:,:img_height-1, 1:])
#     img5 = m4(img[:,1:, :img_width-1])
    
   
#     s11,s12 = img2-img, img-img3
#     s31,s32 = img4-img, img-img5
    
#     s11=hasviside(s11, thr=thr)
#     s12=hasviside(s12, thr=thr)
#     s31=hasviside(s31, thr=thr)
#     s32=hasviside(s32, thr=thr)
    
#     s11=s11+s12
#     s31=s31+s32
    
#     s11[s11==1024]=2
#     s11[s11==2048]=1
    
#     s31[s31==1024]=2
#     s31[s31==2048]=1
#     s31 *= 3
#     res1 = s11+s31

# def ltp_transform(x, **kwargs):
#     thr = 3
#     x = x.to(torch.float32)
#     h,w = x.shape[1:]
#     x = get_features(x, h, w)
#     return x

# h, w = img.shape[1:]
# val_ltp = A.Compose([
#     ToTensorV2(),
#     A.Lambda(name='ltp', image=ltp_transform, p=1, always_apply=True),
#     A.Normalize(
#         mean=(0.2862735,  0.27895936, 0.28809178),
#         std=(0.34015885, 0.3387185,  0.33974034),
#         max_pixel_value=16384.0,
#         always_apply=True,
#         p=1.0
#     ),
#     ToTensorV2()
# ])

# train_seq = A.Compose([
#     A.SmallestMaxSize(max_size=256),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#     A.RandomCrop(height=224, width=224),
#     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#     A.RandomBrightnessContrast(p=0.5)
# ])

# # 创建一个TensorBoard的SummaryWriter对象，指定日志保存路径
# writer = SummaryWriter(log_dir='logs')

# # 进行数据增强
# augmented_img = train_seq(image=img)
# augmented_img = augmented_img['image']

# # 将图像转换为张量
# augmented_img = torch.from_numpy(augmented_img)

# # 使用make_grid函数创建图像网格
# grid_img = make_grid(augmented_img)

# # Convert the data type of the tensor
# augmented_img = augmented_img.to(torch.uint8)

# # Add the image to TensorBoard
# writer.add_image('Augmented Image', augmented_img)


# # 关闭SummaryWriter
# writer.close()

# import glob
# import cv2
# import numpy as np

# def calculate_average_pixel(folder_path):
#     file_paths = glob.glob(os.path.join(folder_path, '**', '*.JPEG'), recursive=True)
#     total_pixels = np.zeros(3)  # 總像素值的總和，初始化為0
#     total_images = 0  # 總圖片數量，初始化為0

#     for file_path in tqdm.tqdm(file_paths, total=len(file_paths)):
#         image = cv2.imread(file_path)
#         if image is not None:
#             total_pixels += np.sum(image, axis=(0, 1))
#             total_images += 1

#     if total_images > 0:
#         average_pixel = total_pixels / (total_images * image.shape[0] * image.shape[1])
#         return average_pixel
#     else:
#         return None

# folder_path = '/mnt/sdd/ilsvrc2012'
# average_pixel = calculate_average_pixel(folder_path)

# if average_pixel is not None:
#     print('Average Pixel Value:', average_pixel)
# else:
#     print('No images found in the folder.')

# im = cv2.imread('/mnt/sdd/ilsvrc2012/train/n04069434/n04069434_13325.JPEG')
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# SIFT = DenseSIFTDescriptor()
# im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float()  # 转换图像并添加维度
# descs = SIFT(im_tensor)
# descs = descs.squeeze(0)
# print(descs.shape)

# image_np = descs.detach().numpy()
# image_reshaped = np.transpose(image_np, (1, 2, 0))
# image_flattened = image_reshaped.reshape(-1, image_reshaped.shape[2])

# pca = PCA(n_components=3)
# image_transformed = pca.fit_transform(image_flattened)
# image_transformed = image_transformed.reshape(image_reshaped.shape[0], image_reshaped.shape[1], 3)


# print(image_transformed.shape)

#500*246*3=369,000 ,[1, D, W, H]

# import cv2
# import torch
# import numpy as np
# from sklearn.decomposition import PCA

# im = cv2.imread('/mnt/sdd/ilsvrc2012/train/n04069434/n04069434_13325.JPEG')
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# SIFT = DenseSIFTDescriptor()
# im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float()  # 轉換圖像並添加維度
# descs = SIFT(im_tensor)
# descs = descs.squeeze(0)
# print(descs.shape)

# image_np = descs.detach().numpy()
# image_reshaped = np.transpose(image_np, (1, 2, 0))
# image_flattened = image_reshaped.reshape(-1, image_reshaped.shape[2])

# pca = PCA(n_components=3)
# image_transformed = pca.fit_transform(image_flattened)
# image_transformed = image_transformed.reshape(image_reshaped.shape[0], image_reshaped.shape[1], 3)

# # 提取 eigenvalues 和 eigenvectors
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_
# print(eigenvalues)
# print(eigenvectors)

# # 排序 eigenvalues
# sorted_indices = np.argsort(eigenvalues)[::-1]
# sorted_eigenvalues = eigenvalues[sorted_indices]

# # 計算 v(1~3) / v(all)
# v_1_3_divided_by_all = sorted_eigenvalues[:3] / np.sum(sorted_eigenvalues)

# print("Sorted Eigenvalues:")
# print(sorted_eigenvalues)
# print("v(1~3) / v(all):")
# print(v_1_3_divided_by_all)

# import cv2
# import torch
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# im = cv2.imread('/mnt/sdd/ilsvrc2012/train/n04069434/n04069434_13325.JPEG')
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# SIFT = DenseSIFTDescriptor()
# im_tensor = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float()
# descs = SIFT(im_tensor)
# descs = descs.squeeze(0)

# image_np = descs.detach().numpy()
# image_reshaped = np.transpose(image_np, (1, 2, 0))
# image_flattened = image_reshaped.reshape(-1, image_reshaped.shape[2])

# pca = PCA(n_components=3)
# image_transformed = pca.fit_transform(image_flattened)

# # 提取 eigenvalues 和 eigenvectors
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_

# # 視覺化特徵向量
# plt.figure(figsize=(10, 10))
# for i, ev in enumerate(eigenvectors):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(ev.reshape(image_reshaped.shape[:2]), cmap='gray')
#     plt.title(f"Eigenvector {i+1}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

class DenseSIFTDescriptorTransform(object):
    def __init__(self):
        self.SIFT = DenseSIFTDescriptor()

    def __call__(self, image, **kwargs):
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        descs = self.SIFT(image_tensor)
        descs = descs.squeeze(0)
        return descs

image_paths = glob.glob('../../sdd/ilsvrc2012/train/*/*.JPEG')
transform = DenseSIFTDescriptorTransform()

train_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomCrop(height=224, width=224),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
])

descriptors = []
for image_path in tqdm.tqdm(image_paths):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = train_transform(image=image)['image']
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)
    descriptor = transform(transformed_image)
    descriptors.append(descriptor)

    del image, transformed_image, descriptor

print(len(descriptors))
