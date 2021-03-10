import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from utils import load_model
from dataloader import *
from torch.utils.data import DataLoader,Dataset
from models.metrics import *

def model_accuracy(model,dataloader,epoch):
    start = time.time()
    is_cuda = torch.cuda.is_available()
    device = torch.device(3)
    accuracy_list = []

    for k in range(epoch):
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                x,n,target = data
        
                if is_cuda:
                    x = x.float().cuda(device)
                    n = n.float().cuda(device)
                    target = target.float().cuda(device)
        
                y_hat = model(x,n)
                accuracy, _, _, _ = confuse_matrix(target,y_hat)
                accuracy = accuracy * 100
                accuracy_list.append(accuracy)

    return accuracy_list


def model_auprc():
    return None

def model_auroc():
    return None

if __name__ == "__main__":
    device = torch.device(3)
    model = load_model("SeFT_01.pt").cuda(device)
    PATH = '/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/'
    df_b = pd.read_csv(PATH + 'B/set-b/B-dataset.csv')
    dataset_b = MyDataLoader(df_b,1024)
    dataloader_b = DataLoader(dataset_b, shuffle=False, batch_size=256,drop_last=True)
    
    accuracy_list = model_accuracy(model,dataloader_b,20)
    print(accuracy_list)