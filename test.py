# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:00:47 2024

@author: HUAWEI
"""
from easydict import EasyDict
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import Net
from utils import AllSetLoader


def getConfig():
    config = EasyDict()

    config.cwd = os.getcwd()
    config.load_path = config.cwd + "/models/"
    config.trainset_dir = config.cwd + "/data/training/"
    config.validset_dir = config.cwd + "/data/validation/"
    config.testset_dir = config.cwd + "/data/test/"
    config.angRes = 9
    config.dispMin = -4
    config.dispMax = 4
    config.device = 'cuda:0'
    return config

def Test(config):
    NET = Net(config)
    NET.to(config.device)

    ckpt = load_ckpt(config)
    NET.load_state_dict(ckpt['state_dict'])

    TestSet = AllSetLoader(config, kind = 'test')
    TestDataLoader = DataLoader(dataset = TestSet, batch_size = 1, shuffle = False)  
        
    outputs = []
    
    with torch.no_grad():
        for i_iter, (lf, disp) in tqdm(enumerate(TestDataLoader), 
                                           total = len(TestDataLoader)):
            lf = lf.to(config.device)
            dispPred = NET(lf)
            
            dispPred = dispPred.cpu().squeeze().numpy()
            outputs.append(dispPred)
            
    outputs = np.stack(outputs)
    np.save('./test_outputs/outputs.npy', outputs)

def load_ckpt(config, filename = 'models/OACC-Net.pth.tar'):
    ckpt = torch.load(filename, map_location = {'cuda:0': config.device})
    return ckpt
            
if __name__ == '__main__':
    config = getConfig()
    if not (os.path.exists('./test_outputs/') and os.path.isdir('./test_outputs/')):
        os.makedirs("test_outputs")
    Test(config)

   
