# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:50:57 2024

@author: HUAWEI
"""


import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from easydict import EasyDict
import tqdm
import time
import os
from model import Net
from utils import *


def getConfig():
    config = EasyDict()
    
    config.cwd = r"D:\Graduation Design\MyCode"
    config.save_path = config.cwd + r"\models"
    config.trainset_dir = "????????????????????????????"
    config.validset_dir = "????????????????????????????"
    config.testset_dir = "????????????????????????????"
    config.angRes = 9
    config.dispMin = -4
    config.dispMax = 4
    config.device = 'cuda:0'
    config.trainset_dir = '?????'
    config.lr = 0.001
    config.n_epochs = 3500          # lr scheduler updating frequency
    config.max_epochs = 3500
    config.batch_size = 16
    config.n_threads = 8            # num of used threads of DataLoader
    config.gamma = 0.5              # lr scheduler decaying rate
    return config


def Train(config):
    if not torch.cuda.is_available():
        print('--Cuda Unavailable!--')
        return
    
    NET = Net(config)
    NET.to(config.device)
    cudnn.benchmark = True
    
    Loss = torch.nn.L1Loss().to(config.device)
    Optimizer = torch.optim.Adam([paras for paras in NET.parameters() 
                                  if paras.requires_grad == True], lr = config.lr)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, 
                                                step_size = config.n_steps,
                                                gamma = config.gamma)
    loss_log = []
    
    '''///Train///'''
    TrainSet = TrainSetLoader(config)
    
    for i_epoch in range(config.max_epochs):
        TrainDataLoader = DataLoader(dataset = TrainSet, num_workers = config.n_threads,
                                     batch_size = config.batch_size, shuffle = True)        
        loss_epoch_log = []
        
        for i_iter, (lfs, dispGTs) in tqdm(enumerate(TrainDataLoader), 
                                           total = len(TrainSet)):
            lfs = lfs.to(config.device)
            dispGTs = dispGTs.to(config.device)
            # lfs.shape = (b u v h w)
            # dispGTs.shape = (b h w)
            
            dispPred = NET(lfs, dispGTs)
            # dispPred.shape = (b c h w), c = 1
            
            loss_i = Loss(dispPred.squeeze(), dispGTs)
            Optimizer.zero_grad()
            loss_i.backward()
            Optimizer.step()
            
            loss_epoch_log.append(loss_i.data.cpu())
            
        if True:
            loss_epoch_avg = np.array(loss_epoch_log).mean()
            loss_log.append(loss_epoch_avg)
            
            log_info = time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {}".format(i_epoch, loss_epoch_avg)
            with open("train_log.txt", "a") as f:
                f.write(log_info)
                f.write("\n")
            print(log_info)
            
        if i_epoch % 10 == 9:
            torch.save({
                'epoch': i_epoch + 1,
                'model_state_dict': NET.state_dict()
            }, os.path.join("log", "model_{:0>4}.pth".format(i_epoch + 1)))
            
            Valid(NET, config, i_epoch)
        
        Scheduler.step()
        
        
def Valid(NET, config, i_epoch):
    torch.no_grad()
    scenes = ['boxes', 'cotton', 'dino', 'sideboard']
    
    "///Validation///"
    for scene in scenes:
        "???????????????????????????????????????????????????????????????????"
    
    
if __name__ == "__main__":
    config = getConfig()
    os.chdir(config.cwd)
    if not (os.path.exists(config.save_path) and os.path.isdir(config.save_path)):
        os.makedirs("models")
    
    
    
    