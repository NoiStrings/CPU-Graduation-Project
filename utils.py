# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NRiA7cd6RZmQ6WveVbyJWPYHkRU2-aKy
"""



import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import os
import numpy as np
import imageio
import imgaug.augmenters as iaa
from func_pfm import read_pfm

# # Commented out IPython magic to ensure Python compatibility.
# '''CONNECT GOOGLE DRIVE'''
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)
# import sys
# sys.path.append('/content/drive/MyDrive/Colab_Notebooks')
# %cd /content/drive/MyDrive/Colab_Notebooks
# from func_pfm import read_pfm

class TrainSetLoader(Dataset):
    def __init__(self, config):
        super(TrainSetLoader, self).__init__()
        self.trainset_dir = config.trainset_dir
        self.source_files = sorted(os.listdir(self.trainset_dir))
        self.angRes = config.angRes
        self.scene_idx = []

        no_reflection_idx = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
        with_reflection_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for i in range(80):
            self.scene_idx += no_reflection_idx
        '''for i in range(20):
            self.scene_idx += with_reflection_idx'''

        self.length = len(self.scene_idx)

    def __getitem__(self, idx):
        scene_id = self.scene_idx[idx]
        scene_name = self.source_files[scene_id]

        lf = np.zeros((9, 9, 512, 512, 3), dtype = int)
        dispGT = np.zeros((512, 512), dtype = float)
        mask = np.zeros((512, 512), dtype = float)

        for i in range(9 * 9):
            SAI_path = self.trainset_dir + scene_name + '/input_Cam0{:0>2}.png'.format(i)
            SAI = imageio.imread(SAI_path)
            lf[i // 9, i % 9, :, :, :] = SAI
        disp_path = self.trainset_dir + scene_name + '/gt_disp_lowres.pfm'
        mask_path = self.trainset_dir + scene_name + '/valid_mask.png'
        dispGT[:, :] = np.float32(read_pfm(disp_path))
        mask_rgb = imageio.imread(mask_path)
        mask = np.float32(mask_rgb[:, :, 1] > 0)

        lf, dispGT = DataAugmentation(lf, dispGT)
        # lf.shape = (u v h w)
        # disp.shape = (h w)
        data = lf.astype('float32')
        label = dispGT.astype('float32')
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        # lf.shape = (u v h w)
        # disp.shape = (h w)

        return data, label

    def __len__(self):
        return self.length


class AllSetLoader(Dataset):
    def __init__(self, config, kind):
        super(AllSetLoader, self).__init__()
        self.dataset_dir = None
        if kind == "valid":
            self.dataset_dir = config.validset_dir
        elif kind == "test":
            self.dataset_dir = config.testset_dir
        self.source_files = sorted(os.listdir(self.dataset_dir))
        self.angRes = config.angRes
        self.length = len(self.source_files)
        
    def __getitem__(self, idx):
        scene_name = self.source_files[idx]
        
        lf = np.zeros((9, 9, 512, 512, 3), dtype = int)
        dispGT = np.zeros((512, 512), dtype = float)

        for i in range(9 * 9):
            SAI_path = self.validset_dir + scene_name + '/input_Cam0{:0>2}.png'.format(i)
            SAI = imageio.imread(SAI_path)
            lf[i // 9, i % 9, :, :, :] = SAI
        disp_path = self.validset_dir + scene_name + '/gt_disp_lowres.pfm'
        dispGT[:, :] = np.float32(read_pfm(disp_path))
        
        lf = np.mean(lf, axis = -1, keepdim = False) / 255
        # lf.shape = (u v h w)
        # dispGT.shape = (h w)
        
        return lf, dispGT

    def __len__(self):
        return self.length
        

def DataAugmentation(lf, disp):
    # lf.shape = (u v h w c), c = RGB
    # disp.shape = (h w)
    lf = np.reshape(lf, (81, 512, 512, 3))

    lf_Aug1 = IlluminanceAugmentation(lf)
    # lf_Aug1.shape = ((u v) h w)
    lf_Aug2, disp_Aug1 = ScaleAugmentation(lf_Aug1, disp)
    lf_Aug3, disp_Aug2 = OrientationAugmentation(lf_Aug2, disp_Aug1)

    lf_Aug3 = np.reshape(lf_Aug3, (9, 9, 512, 512))
    return lf_Aug3, disp_Aug2


def IlluminanceAugmentation(lf):
    # lf.shape = ((u v) h w c), c = RGB
    rand1 = np.random.randint(-100, 100)
    rand2 = np.random.randint(-50, 50)
    rand3 = np.random.uniform(0.0, 1.0)
    rand4 = np.random.uniform(0.0, 0.05*255)
    seq = iaa.Sequential([
        iaa.AddToBrightness(rand1),
        iaa.AddToHue(rand2),
        iaa.Grayscale(alpha = rand3),
        iaa.AdditiveGaussianNoise(loc = 0, scale = rand4, per_channel = False),
        ])
    lf_Aug = seq(images=lf)

    randRGB = 0.05 + np.random.rand(3)
    randRGB = randRGB / np.sum(randRGB)
    lf_Gray = randRGB[0] * lf_Aug[:, :, :, 0] + randRGB[1] * lf_Aug[:, :, :, 1] + randRGB[2] * lf_Aug[:, :, :, 2]

    lf_Gray = np.squeeze(lf_Gray)
    lf_Gray = np.clip(lf_Gray, 0, 255)
    # lf_Gray.shape = ((u v) h w)
    return lf_Gray


def ScaleAugmentation(lf, disp):
    # lf.shape = ((u v) h w)
    # disp.shape = (h w)
    rand1 = np.random.uniform(0.0, 3.0)
    percent = 0
    scale = 0
    if rand1 < 1.5:
        percent = 0
        scale = 1
    elif rand1 < 2.5:
        percent = 0.25
        scale = 2
    else:
        percent = 1/3
        scale = 3
    # percent = (scale - 1) / (2 * scale)
    seq = iaa.Sequential([iaa.Crop(percent = percent)])

    lf_Aug = seq(images = lf)
    disp_Aug = seq(image = disp)
    disp_Aug /= scale
    return lf_Aug, disp_Aug


def OrientationAugmentation(lf, disp):
    # lf.shape = ((u v) h w)
    # disp.shape = (h w)

    flipLR = iaa.Fliplr(p = 1)
    flipUD = iaa.Flipud(p = 1)
    rotate_1 = iaa.Rotate(rotate = 90)
    rotate_2 = iaa.Rotate(rotate = -90)

    rands = [np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]
    if rands[0] > 0.5:
        lf = flipLR(images = lf)
        disp = flipLR(image = disp)
    if rands[1] > 0.5:
        lf = flipUD(images = lf)
        disp = flipUD(image = disp)
    if rands[2] > 0.75:
        lf = rotate_1(images = lf)
        disp = rotate_1(image = disp)
    elif rands[2] > 0.5:
        lf = rotate_2(images = lf)
        disp = rotate_2(image = disp)
    return lf, disp
