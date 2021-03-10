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



class SeFT(nn.Module):
    def __init__(self,m=4,d=128):
        super().__init__()
        self.m = m
        self.d = d
        self.set_function = SetFunc()
        self.attention1 = Attention()
        self.attention2 = Attention()
        self.attention3 = Attention()
        self.attention4 = Attention()
        self.fc1 = nn.Linear(128*m,256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,n):
        bs = x.shape[0]
        ###
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        query = torch.zeros(bs,self.m,self.d)
        if is_cuda:
            query = query.float().to(device)
        ###
        ### original input x is used to make Q for attention mechanism ###
        # ▼ encoding process
        e = self.set_function(x,n) # shape : (bs,2)
        r_1 = self.attention1(e,n,x,query)
        r_2 = self.attention2(e,n,x,query)
        r_3 = self.attention3(e,n,x,query)
        r_4 = self.attention4(e,n,x,query)
        r_asterisk = torch.cat((r_1,r_2,r_3,r_4),dim=1)
        # ▼ decoding process
        e = r_asterisk.view(r_asterisk.shape[0],-1)
        e = self.fc1(e)
        e = self.relu1(e)
        e = self.dropout(e)
        e = self.fc2(e)
        e = self.relu2(e)
        e = self.fc3(e)
        y = self.sigmoid(e)
        return y



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
    def __init__(self,d=128):
        super().__init__()
        self.d = d
        self.h = H()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,f_S,n,x,query):
        #####
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        #####

        bs = x.shape[0]
        n_list = n.tolist()

        matrix = torch.zeros(bs,1024,5)
        matrix[0:,0:,0:2] = f_S.view(bs,1,2)
        matrix[0:,0:,2:] = x
        weight = torch.randn(5,self.d)
        r_i = torch.zeros(64,128)
        if is_cuda:
            matrix = matrix.float().to(device)
            weight = weight.float().to(device)
            r_i = r_i.float().to(device)
        key = torch.matmul(matrix,weight)
        e_ji = torch.sum(key[0:,0:,0:]*query[0:,0,0:].view(64,1,128),2).view(bs,1024,1)/np.sqrt(self.d)
        for j in range(bs): # Masking
            length = int(n_list[j][0])
            e_ji[j,length:,0] = -1000.0
        a_ji = self.softmax(e_ji) # [64,1024,1]
        h_s = self.h(x) # [64,128]
            
        # calculate sum(a_ji*h_s) = r_i
        for k in range(bs):
            temp = torch.sum(a_ji[k]*h_s[k],dim=0)
            r_i[k] = temp

        return r_i

""" 
############
class Attention above this is more clear and works well
############

class Attention(nn.Module):
    def __init__(self,m=4,d=128):
        super().__init__()
        self.m = m
        self.d = d
        self.h1 = H()
        self.h2 = H()
        self.h3 = H()
        self.h4 = H()
        self.softmax = nn.Softmax(dim=1)
        self.h_list = [self.h1,self.h2,self.h3,self.h4]
        self.r_asterisk = []
    
    def forward(self,f_S,n,x):

        #####
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        #####

        bs = x.shape[0]
        n_list = n.tolist()
        query = torch.zeros(bs,self.m,self.d)
        if is_cuda:
            query = query.float().to(device)
        for i in range(self.m):
            matrix = torch.zeros(bs,1024,5)
            matrix[0:,0:,0:2] = f_S.view(bs,1,2)
            matrix[0:,0:,2:] = x
            weight = torch.randn(5,self.d)
            r_i = torch.zeros(64,128)
            if is_cuda:
                matrix = matrix.float().to(device)
                weight = weight.float().to(device)
                r_i = r_i.float().to(device)
            key = torch.matmul(matrix,weight)
            e_ji = torch.sum(key[0:,0:,0:]*query[0:,0,0:].view(64,1,128),2).view(bs,1024,1)/np.sqrt(self.d)
            for j in range(bs): # Masking
                length = int(n_list[j][0])
                e_ji[j,length:,0] = -1000.0
            a_ji = self.softmax(e_ji) # [64,1024,1]
            h_s = self.h_list[i](x) # [64,128]
            
            # r_i 연산하기
            for k in range(bs):
                temp = torch.sum(e_ji[i]*h_s[i],dim=0)
                r_i[i] = temp
            self.r_asterisk.append(r_i)
        else: 
            concat_r = self.r_asterisk[0]
    
        for l in range(self.m-1):
            concat_r = torch.cat((concat_r,self.r_asterisk[l+1]),dim=1)
        return concat_r
"""





if __name__ == '__main__':
    df = pd.read_csv('/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/A/set-a/A-dataset.csv')
    print(df.head())
    dataset = MyDataLoader(df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64, pin_memory=False)
    x, n, target = next(iter(dataloader))
    x = x.float().cuda()
    n = n.float().cuda()
    target = target.float().cuda()
    
    model = SeFT().cuda()
    y = model(x,n)
    print(y)