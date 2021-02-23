import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np
import pandas as pd


'''
class Transformer(nn.Module):
    def __init__(self,d_embed=3,d_k=3,seq_len=500)
    pass


class Encoder(nn.Module):
    def __init__(self,d_embed=3):
        self.d_embed = d_embed
    pass


class Decoder(nn.Module):
    pass


class FeedForward(nn.Module):
    pass



class MultiHeadAttn(nn.Module):

    def forward(self,x): # residual and normalization
        return outcome + x
    pass
'''

class SDPAttn(nn.Module):
    def __init__(self,d_embed,d_k,h,n,**kwargs):
        super().__init__()
        self.q_layer = QKV_FCLayer(d_embed,d_k,h)
        self.k_layer = QKV_FCLayer(d_embed,d_k,h)
        self.v_layer = QKV_FCLayer(d_embed,d_k,h)
        self.pad_masking = PadMasking(**kwargs)
    
    def forward(self,x):
        Q,_ = self.q_layer(x)
        K,_ = self.k_layer(x)
        V,n = self.v_layer(x)
        outcome = torch.matmul(Q,torch.transpose(K,-1,-2))
        outcome = outcome/np.sqrt(d_k)
        outcome = self.pad_masking(outcome,n)
        #행별로 softmax
        outcome = torch.matmul(outcome,V)
        return outcome

class PadMasking(nn.Module):
    def __init__(self,seq_len=500,**kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.mask_matrix = torch.ones(self.seq_len,self.seq_len) * (-10e+8)
    
    def forward(self,x,n):
        self.mask_matrix[0:,0:n,0:n] = 1.0 #n,n까지만 1로 해준다
        x = x*self.mask_matrix
        return x


class QKV_FCLayer(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        self.h = h
        self.w = w
        self.FC = torch.randn(self.h,self.w)

    def forward(self,x):
        n = x.shape[-2]
        x = torch.matmul(x,self.FC)
        return x,n



if __name__=='__main__':
    dummy = torch.zeros(410,3)
    fc = FCLayer(3,3,6)
    print(fc(dummy))