# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:50:57 2024

@author: HUAWEI
"""


import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
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
    config.lr = 0.001
    config.n_epochs = 20          # lr scheduler updating frequency
    config.max_epochs = 20
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
    
    if config.load_pretrain:
        models = os.listdir(config.save_path)
        if models:
            models.sort()
            model_name = models[-1]
            model_path = config.save_path + model_name
        if os.path.isfile(model_path):
            model = torch.load(model_path, map_location = {'cuda:0': config.device})
            NET.load_state_dict(model['model_state_dict'], strict = False)
            print("Notification: Pretrained Models Loaded.")
        else:
            print("Notification: No Pretrained Models Found.")

    Loss = torch.nn.L1Loss().to(config.device)
    Optimizer = torch.optim.Adam([paras for paras in NET.parameters() 
                                  if paras.requires_grad == True], lr = config.lr)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, 
                                                step_size = config.n_epochs,
                                                gamma = config.gamma)
    loss_log = []
    
    '''///Train///'''
    TrainSet = TrainSetLoader(config)
    
    for i_epoch in range(config.max_epochs):
        TrainDataLoader = DataLoader(dataset = TrainSet, num_workers = config.num_workers,
                                     batch_size = config.batch_size, shuffle = True)        
        loss_epoch_log = []
        
        for i_iter, (lfs, dispGTs) in tqdm(enumerate(TrainDataLoader), 
                                           total = len(TrainDataLoader)):
            lfs = lfs.to(config.device)
            dispGTs = dispGTs.to(config.device)
            # lfs.shape = (b u v h w)
            # dispGTs.shape = (b h w)

            with autocast():                                   
                dispPred = NET(lfs, dispGTs)
                # dispPred.shape = (b c h w), c = 1
            
                loss_i = Loss(dispPred.squeeze(), dispGTs.squeeze())
                Optimizer.zero_grad()
                loss_i.backward()
            Optimizer.step()
            
            loss_epoch_log.append(np.array(loss_i.data.cpu()).mean())
            
        if True:
            loss_epoch_avg = np.array(loss_epoch_log).mean()
            loss_log.append(loss_epoch_avg)
            
            log_info = "[Train] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {}".format(i_epoch, loss_epoch_avg)
            with open("logs/train_log.txt", "a") as f:
                f.write(log_info)
                f.write("\n")
            print(log_info)
                    
            torch.save({
                'epoch': i_epoch + 1,
                'model_state_dict': NET.state_dict()
            }, os.path.join("models", "model_{:0>4}.pth".format(i_epoch + 1)))
        ''' 
        if i_epoch % 3 == 2:    
            Valid(NET, config, i_epoch)
        '''
        Scheduler.step()
        
    return
        
        
def Valid(NET, config, i_epoch):
    torch.no_grad()
    
    Loss = torch.nn.L1Loss().to(config.device)
    
    ValidSet = AllSetLoader(config, kind = 'valid')
    ValidDataLoader = DataLoader(dataset = ValidSet, batch_size = 1, shuffle = False)   
    
    "///Validation///"
    with torch.no_grad():
        loss_log = []
        for i_iter, (lf, dispGT) in tqdm(enumerate(ValidDataLoader), 
                                           total = len(ValidSet)):
            lf = lf.to(config.device)
            dispGT = dispGT.to(config.device)
            # lf.shape = (b u v h w)
            # dispGT.shape = (b h w)
            
            dispPred = NET(lf, dispGT)
            # dispPred.shape = (b c h w), c = 1
            
            loss_i = Loss(dispPred.squeeze(), dispGT)
            loss_log.append(loss_i.data.cpu())
        
        loss_avg = np.array(loss_log).mean()
        log_info = "[Valid] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {}".format(i_epoch, loss_avg)
        with open("logs/valid_log.txt", "a") as f:
            f.write(log_info)
            f.write("\n")
        print(log_info)
        
    return
    
if __name__ == "__main__":
    config = getConfig()
    if not (os.path.exists(config.save_path) and os.path.isdir(config.save_path)):
        os.makedirs("models")
    if not (os.path.exists(config.log_path) and os.path.isdir(config.log_path)):
        os.makedirs("logs")
    Train(config)
    
    
    