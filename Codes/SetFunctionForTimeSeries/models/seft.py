import torch
import numpy as np
import torch.nn as nn


class CLASS_(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return None



class SeFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = H()
        self.g = G()
        self.attention = Attention()
        self.fc1 = nn.Linear(??,??)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(??,??)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(??,??)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.h(x)
        x = self.g(x)
        x = self.attention(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class H(nn.Module):
    '''
    x to "d" dimension
    '''
    def __init__(self,d=128):
        super().__init__()
        self.
    def forward(self):
        return None


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

if __name__ == '__main__':
    pass