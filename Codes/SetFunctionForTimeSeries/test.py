import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy,auroc
from utils import load_model,load_data
from dataloader import *
from torch.utils.data import DataLoader,Dataset
#from models.metrics import *

def test_model(model,dataloader,epoch):
    start = time.time()
    is_cuda = torch.cuda.is_available()
    device = torch.device(3)
    ACCURACY = []
    AUROC = []
    AUPRC = []

    for k in range(epoch):
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                x,n,target = data
                if is_cuda:
                    x = x.float().cuda(device)
                    n = n.float().cuda(device)
                    target = target.float().cuda(device)
                y_hat = model(x,n)

                '''using torchmetrics'''
                accuracy_score = accuracy(y_hat,target.long())
                auroc_score = auroc(y_hat,target.long(),pos_label=1)
                #auprc_score = ??

                '''using my metrics'''
                #accuracy,_,_,_ = confuse_matrix(target,y_hat)
                #TPR ,FPR = roc(target,y_hat)
                #auroc_score = auroc(TPR,FPR)
                #PRECISION, RECALL = prc(target,y_hat)
                #auprc_score = auprc(PRECISION,RECALL)

                
                ACCURACY.append(accuracy_score.item())
                AUROC.append(auroc_score.item())
                #AUPRC.append(auprc_score)
            
    return ACCURACY,AUROC,AUPRC

def analysis(ACCURACY,AUROC,AUPRC):
    pass

def get_dataloader(bs=64):
    test_df = load_data(forwhat=True)#test df
    dataset = MyDataLoader(test_df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=bs,drop_last=True)
    return dataloader

if __name__ == "__main__":
    device = torch.device(3)
    model = load_model('SeFT_03.pt').cuda(device)
    dataloader = get_dataloader()
    ACCURACY, AUROC, AUPRC = test_model(model,dataloader,1)

    print('+========== ACCURACY_list ==========+')
    print(ACCURACY)
    print('+========== AUROC_list ==========+')
    print(AUROC)
    print('+========== AUPRC_list ==========+')
    print(AUPRC)