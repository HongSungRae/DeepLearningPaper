import torch
import numpy as np
import torch.nn as nn
from torch.summary import summary


class CLASS_(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return None



class SeFT(nn.Module):
    def __init__(self,m=4):
        super().__init__()
        self.h = H()
        self.g = G()
        self.attention = Attention()
        self.fc1 = nn.Linear(128*m,256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,n):
        original_input = x.clone().detach()
        x = self.h(x,n)
        x = self.g(x) # f'(S)
        x = self.attention(x,n,original_input)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        y = self.sigmoid(x)
        return y


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
    def __init__(self):
        super().__init__()
    
    def forward(self,f_S,n,x):
        f_S를 늘려주고 x와 concat
        return None





if __name__ == '__main__':
    pass