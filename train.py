# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:50:57 2024

@author: HUAWEI
"""


import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# from torch.cuda.amp import autocast
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
import time
import os
from model import Net
from utils import *


def getConfig():
    config = EasyDict()

    config.cwd = os.getcwd()
    config.save_path = config.cwd + "/models/"
    config.log_path = config.cwd + "/logs/"
    config.trainset_dir = config.cwd + "/data/training/"
    config.validset_dir = config.cwd + "/data/validation/"
    config.testset_dir = config.cwd + "/data/test/"
    config.load_pretrain = True
    config.angRes = 9
    config.dispMin = -4
    config.dispMax = 4
    config.device = 'cuda:0'
    config.lr = 0.0001
    config.n_epochs = 60          # lr scheduler updating frequency
    config.max_epochs = 60
    config.batch_size = 16
    config.patch_size = 48
    config.num_workers = 6          # num of used threads of DataLoader
    config.gamma = 0.5              # lr scheduler decaying rate
    return config


def Train(config):
    if not torch.cuda.is_available():
        print('--Cuda Unavailable!--')
        return
    
    # get GPU info    
    device_properties = torch.cuda.get_device_properties(0)
    total_memory = device_properties.total_memory
    print(f"Total CUDA Memory: {total_memory / 1024**3:.2f} GB")
    
    NET = Net(config)
    NET.to(config.device)
    cudnn.benchmark = True
    start_epoch = 0
    
    Loss = torch.nn.L1Loss().to(config.device)
    Optimizer = torch.optim.Adam([params for params in NET.parameters() 
                                  if params.requires_grad == True], lr = config.lr)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, 
                                                step_size = config.n_epochs,
                                                gamma = config.gamma)
    
    if config.load_pretrain:
        models = os.listdir(config.save_path)
        model_path = ''
        if models:
            models.sort()
            model_name = models[-1]
            model_path = config.save_path + model_name
        if os.path.isfile(model_path):
            ckpt = load_ckpt(config, model_path)
            NET.load_state_dict(ckpt['state_dict'], strict = False)
            Optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch']
            print("Notification: Pretrained Model Loaded.")
        else:
            print("Notification: No Pretrained Model Available.")
    
    '''///Train///'''
    TrainSet = TrainSetLoader(config)
    
    for i_epoch in range(start_epoch, config.max_epochs):
        TrainDataLoader = DataLoader(dataset = TrainSet, num_workers = config.num_workers,
                                     batch_size = config.batch_size, shuffle = True)   
        total_loss = 0
        num_iters = 0
        
        for i_iter, (lfs, dispGTs) in tqdm(enumerate(TrainDataLoader), 
                                           total = len(TrainDataLoader)):
            lfs = lfs.to(config.device)
            dispGTs = dispGTs.to(config.device)
            # lfs.shape = (b u v h w)
            # dispGTs.shape = (b h w)

            with autocast():                                   
                dispPred = NET(lfs, dispGTs)
                # dispPred.shape = (b c h w), c = 1
                # ////////////////////////////////////////////////////////////////////////
                # max_value1 = torch.max(dispPred[~torch.isinf(dispPred) & ~torch.isnan(dispPred)])  # 忽略inf和NaN
                # min_value1 = torch.min(dispPred[~torch.isinf(dispPred) & ~torch.isnan(dispPred)])  # 忽略inf和NaN
                # print("最大值:", max_value1.item())
                # print("最小值:", min_value1.item())
                # max_value2 = torch.max(dispGTs[~torch.isinf(dispGTs) & ~torch.isnan(dispGTs)])  # 忽略inf和NaN
                # min_value2 = torch.min(dispGTs[~torch.isinf(dispGTs) & ~torch.isnan(dispGTs)])  # 忽略inf和NaN
                # print("最大值:", max_value2.item())
                # print("最小值:", min_value2.item())
                # print(dispPred)
                # ////////////////////////////////////////////////////////////////////////
                loss_i = Loss(dispPred.squeeze(), dispGTs.squeeze())
                Optimizer.zero_grad()
                loss_i.backward()
                Optimizer.step()
                
            num_iters += 1
            total_loss += loss_i.item()
            # //////////////////////////////////////////////////////////////////////////
            # print(loss_i.item())
            # //////////////////////////////////////////////////////////////////////////
        loss_avg = total_loss / num_iters
        
        save_ckpt(NET, Optimizer, i_epoch)
            
        log_info = "[Train] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {:.5f}".format(i_epoch, loss_avg)
        with open("logs/train_log.txt", "a") as f:
            f.write(log_info)
            f.write("\n")
        print(log_info)  
          
        if i_epoch % 3 == 2:    
            Valid(NET, config, i_epoch)

        Scheduler.step()
        
    return
        
        
def Valid(NET, config, i_epoch):
    NET.eval()
    torch.no_grad()
    
    Loss = torch.nn.MSELoss().to(config.device)
    
    ValidSet = AllSetLoader(config, kind = 'valid')
    ValidDataLoader = DataLoader(dataset = ValidSet, batch_size = 1, shuffle = False)   
    
    "///Validation///"
    with torch.no_grad():
        total_loss = 0
        num_iters = 0
        for i_iter, (lf, dispGT) in tqdm(enumerate(ValidDataLoader), 
                                           total = len(ValidDataLoader)):
            lf = lf.to(config.device)
            dispGT = dispGT.to(config.device)
            # lf.shape = (b u v h w)
            # dispGT.shape = (b h w)
            
            dispPred = NET(lf, dispGT)
            # dispPred.shape = (b c h w), c = 1
            
            loss_i = Loss(dispPred.squeeze(), dispGT.squeeze())
            num_iters += 1
            total_loss += loss_i.item()
        
        loss_avg = total_loss / num_iters
        
        log_info = "[Valid] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {:.5f}".format(i_epoch, loss_avg)
        
        with open("logs/valid_log.txt", "a") as f:
            f.write(log_info)
            f.write("\n")
        print(log_info)
        
    return


def save_ckpt(model, optimizer, epoch, filename = 'models/checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    
    
def load_ckpt(config, filename = 'models/checkpoint.pth.tar'):
    ckpt = torch.load(filename, map_location = {'cuda:0': config.device})
    return ckpt


if __name__ == "__main__":
    config = getConfig()
    if not (os.path.exists(config.save_path) and os.path.isdir(config.save_path)):
        os.makedirs("models")
    if not (os.path.exists(config.log_path) and os.path.isdir(config.log_path)):
        os.makedirs("logs")
    Train(config)
    
    
    