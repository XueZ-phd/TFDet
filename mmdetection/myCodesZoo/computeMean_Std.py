#calculate the mean and std for dataset
#The mean and std will be used in src/lib/datasets/dataset/oxfordhand.py line17-20
#The size of images in dataset must be the same, if it is not same, we can use reshape_images.py to change the size

import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm

img_path = '/home/zx/cross-modality-det/datasets/cvc14/images/*/lwir/*.png'  # 数据集目录
img_names = sorted(glob(img_path))

R_channel = 0
G_channel = 0
B_channel = 0
num = 0
for filename in tqdm(img_names, desc='cal mean...', total=len(img_names)):
    img = np.array(plt.imread(filename), np.float32)*255.
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])
    # assert np.shape(img[:, :, 0]) == (1024, 1280)
    num += np.size(img[:, :, 0])

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for filename in tqdm(img_names, desc='cal var...', total=len(img_names)):
    img = np.array(plt.imread(filename), np.float32)*255.
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("[R_mean, G_mean, B_mean]:", list((R_mean, G_mean, B_mean)))
print("[R_var, G_var, B_var]:", list((R_var, G_var, B_var)))
