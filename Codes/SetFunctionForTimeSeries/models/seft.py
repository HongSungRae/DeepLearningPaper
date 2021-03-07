import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from torchsummary import summary
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataloader import MyDataLoader


class CLASS_(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return None


"""
class SeFT(nn.Module):
    def __init__(self,m=4):
        super().__init__()
        self.set_function = SetFunc()
        self.attention = Attention()
        self.fc1 = nn.Linear(128*m,256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,n):
        ### original input x is used to make Q for attention mechanism ###
        # ▼ encoding process
        e = self.set_function(x,n) # shape : (bs,2)
        e = self.attention(e,n,x)
        # ▼ decoding process
        e = self.relu1(self.fc1(e))
        e = self.dropout(e)
        e = self.relu2(self.fc2(e))
        e = self.fc3(e)
        y = self.sigmoid(e)
        return y
"""


class SetFunc(nn.Module):
    def __init__(self,**kargs):
        super().__init__()
        self.h = H()
        self.g = G()

    def forward(self,x,n):
        x = self.h(x)
        x = x/n
        x = self.g(x)
        return x



class H(nn.Module):
    '''
    x to "d" dimension
    '''
    def __init__(self,d=128):
        super().__init__()
        self.fc1 = nn.Linear(1024*3,1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024,d)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x




class G(nn.Module):
    '''
    d dimension vector to C=2 dimension
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128,64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64,2)
        self.relu2 = nn.ReLU()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x





class Attention(nn.Module):
    def __init__(self,m=4,d=128):
        super().__init__()
        self.m = m
        self.d = d
        self.r_asterisk = []
    
    def forward(self,f_S,n,x):
        bs = n.shape[0]
        n_list = n.tolist()
        query = torch.zeros(x.shape[0],self.m,self.d)
        for k in range(bs):#생각해보니 그냥 고정 size로 박아버리고 zero padding주면 어떨까
        for i in range(self.m):
            # f_S를 늘려주고 x와 concat
            # n.tolist()[62][0]
            key = torch.matmul(matrix,weight)
            = ??/np.sqrt(self.d)
            self.r_asterisk.append(r)
        else: 
            concat_r = self.r_asterisk[0]
    
        for j in range(self.m-1):
            concat_r = torch.cat((concat_r,self.r_asterisk[i+1]),dim=1)
        return concat_r






if __name__ == '__main__':
    df = pd.read_csv('/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/A/set-a/A-dataset.csv')
    print(df.head())
    dataset = MyDataLoader(df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64, pin_memory=False)
    x, n, target = next(iter(dataloader))
    
    #model = SeFT()
    model = SetFunc()
    y = model(x,n)
    print(y)