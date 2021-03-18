import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from utils import load_model,load_data
from dataloader import *
from torch.utils.data import DataLoader,Dataset
from models.metrics import *

def test_model(model,dataloader,epoch):
    start = time.time()
    is_cuda = torch.cuda.is_available()
    device = torch.device(3)
    accuracy_list = []
    precision_list = []
    recall_list = []

    for k in range(epoch):
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                x,n,target = data
        
                if is_cuda:
                    x = x.float().cuda(device)
                    n = n.float().cuda(device)
                    target = target.float().cuda(device)
        
                y_hat = model(x,n)
                accuracy, precision, recall, f1 = confuse_matrix(target,y_hat)
                accuracy = accuracy * 100
                accuracy_list.append(accuracy)
                precision_list.append(round(precision,2))
                recall_list.append(round(recall,2))

    return accuracy_list, precision_list, recall_list


def get_dataloader(bs=64):
    test_df = load_data(forwhat=True)#test df
    dataset = MyDataLoader(test_df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=bs,drop_last=True)
    return dataloader

if __name__ == "__main__":
    dataset_c = MyDataLoader(df_c,1024)
    dataloader_c = DataLoader(dataset_c, shuffle=False, batch_size=256,drop_last=True)
    
    accuracy_list, precision_list, recall_list = test_model(model,dataloader_c,10)
    print('+========== accuracy_list ==========+')
    print(accuracy_list)
    print('+========== precision_list ==========+')
    print(precision_list)
    print('+========== recall_list ==========+')
    print(recall_list)