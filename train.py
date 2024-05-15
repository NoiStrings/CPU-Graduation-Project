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
    config.model_path = config.cwd + "/models/OACC-Net.pth.tar"
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
    config.n_epochs = 10          # lr scheduler updating frequency
    config.max_epochs = 10
    config.batch_size = 8
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
        if os.path.isfile(config.model_path):
            ckpt = load_ckpt(config, config.model_path)
            NET.load_state_dict(ckpt['state_dict'], strict = False)
            start_epoch = ckpt['epoch']
            print("Notification: Pretrained Model Loaded.")
        else:
            print("Notification: No Pretrained Model Available.")
    
    
    '''///Train///'''
    for i_epoch in range(start_epoch, config.max_epochs):
        TrainSet = TrainSetLoader(config)
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
                loss_i = Loss(dispPred.squeeze(), dispGTs.squeeze())
                Optimizer.zero_grad()
                loss_i.backward()
                Optimizer.step()
                
            num_iters += 1
            total_loss += loss_i.item()
            
        loss_avg = total_loss / num_iters
        
        save_ckpt(NET, i_epoch + 1)
            
        log_info = "[Train] " + time.ctime()[4:-5] + "\t epoch: {:0>4} | loss: {:.5f}".format(i_epoch, loss_avg)
        with open("logs/train_log.txt", "a") as f:
            f.write(log_info)
            f.write("\n")
        print(log_info)  
          
        if i_epoch % 10 == 9: 
            save_ckpt(NET, i_epoch, 'models/OACC-Net{}.pth.tar'.format(i_epoch + 1))
            Valid(NET, config, i_epoch + 1)

        Scheduler.step()
        
    return
        
        
def Valid(NET, config, i_epoch):
    NET.eval()
    torch.no_grad()
    
    Loss = torch.nn.MSELoss().to(config.device)
    
    ValidSet = AllSetLoader(config, kind = 'valid')
    ValidDataLoader = DataLoader(dataset = ValidSet, batch_size = 1, shuffle = False)  
    
    scene_list = ['boxes', 'cotton', 'dino', 'sideboard']
    
    with open("logs/valid_log.txt", "a") as f:
        f.write("epoch: {:0>4} ==========".format(i_epoch))
        f.write("\n")
        
    outputs = []
    
    "///Validation///"
    with torch.no_grad():
        for i_iter, (lf, dispGT) in enumerate(ValidDataLoader):
            lf = lf.to(config.device)
            dispGT = dispGT.to(config.device)
            # lf.shape = (b u v h w)
            # dispGT.shape = (b h w)
            
            dispPred = NET(lf)
            # dispPred.shape = (b c h w), c = 1
            
            
            loss_i = Loss(dispPred.squeeze(), dispGT.squeeze())
            
            dispPred = dispPred.cpu().numpy()
            dispGT = dispGT.cpu().numpy()
            
            abs_error = np.abs(dispPred - dispGT)
            bad_pixels = np.sum(abs_error > 0.1)
            total_pixels = dispPred.size
            bpr = bad_pixels / total_pixels
            
            outputs.append(dispPred)
            
            log_info = "[Valid] " + time.ctime()[4:-5] + "\t scene: {:<11} | loss: {:.5f} | bpr: {:.5f}".format(scene_list[i_iter], loss_i * 100, bpr)
            with open("logs/valid_log.txt", "a") as f:
                f.write(log_info)
                f.write("\n")
            print(log_info)
            
    outputs = np.stack(outputs)
    np.save('./valid_outputs/outputs_{:0>4}.npy'.format(i_epoch), outputs)

    return


def save_ckpt(model, epoch, filename = 'models/OACC-Net.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    torch.save(state, filename)
    
    
def load_ckpt(config, filename = 'models/OACC-Net.pth.tar'):
    ckpt = torch.load(filename, map_location = {'cuda:0': config.device})
    return ckpt


if __name__ == "__main__":
    config = getConfig()
    if not (os.path.exists(config.save_path) and os.path.isdir(config.save_path)):
        os.makedirs("models")
    if not (os.path.exists(config.log_path) and os.path.isdir(config.log_path)):
        os.makedirs("logs")
    Train(config)
    
    
    