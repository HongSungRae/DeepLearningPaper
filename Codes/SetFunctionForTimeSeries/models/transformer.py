import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np
import pandas as pd


'''
class Transformer(nn.Module):
    def __init__(self,d_embed=3,d_k=128,seq_len=500,h1=2,h2=2,h3=2,N1=4,N2=4):
        super().__init__()
        self.encoder = Encoder(d_embed,d_k,seq_len,h1,N1)
        self.decoder = Decoder(h2,h3,N2)

    def forward(self,x,n):
        x = self.encoder(x,n)
        x = self.decoder(x)
        return x




class Decoder(nn.Module):
    def __init__(self,h2,h3,N2):
        super().__init__()
        pass
    
    def forward(self,x):
        return None




class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self,x):
        return None
'''




class Encoder(nn.Module):
    def __init__(self,d_embed,d_k,seq_len,h1,N1):
        super().__init__()
        self.N1 = N1
        self.list_multiheadattn = []
        self.list_feedforward = []
        
        for i in range(self.N1):
            self.multiheadattn = MultiHeadAttn(d_embed,d_k,seq_len,h1)
            self.feedforward = FeedForward(d_embed,d_k,seq_len,h1)
            self.list_multiheadattn.append(self.multiheadattn)
            self.list_feedforward.append(self.feedforward)
        
        #self.multiheadattn = MultiHeadAttn(d_embed,d_k,seq_len,h1)
        #self.feedforward = FeedForward(d_embed,d_k)

    def forward(self,x,n):
        for i in range(self.N1):
            x = self.list_multiheadattn[i](x,n)
            x = self.list_feedforward[i](x)
        return x



class FeedForward(nn.Module):
    def __init__(self,d_embed,d_k,seq_len,h1,d_ff=256):
        super().__init__()
        self.fc1 = FCLayer(d_embed,d_ff)
        self.relu = nn.ReLU()
        self.fc2 = FCLayer(d_ff,d_embed)
        self.layer_norm = nn.LayerNorm([seq_len,d_embed])

    def forward(self,x):
        context = self.fc1(x)
        context = self.relu(context)
        context = self.fc2(context)
        context = context + x
        context = self.layer_norm(context)
        return context



class MultiHeadAttn(nn.Module):
    def __init__(self,d_embed,d_k,seq_len,h1):
        super().__init__()
        self.sdpattn = SDPAttn(d_embed,d_k,seq_len,h1)
        self.d_model = d_k*h1
        self.fc = FCLayer(self.d_model,d_embed)
        self.layer_norm = nn.LayerNorm([seq_len,d_embed])

    def forward(self,x,n): # residual and normalization
        outcome = self.sdpattn(x,n)
        outcome = self.fc(outcome)
        outcome = outcome + x
        outcome = self.layer_norm(outcome)
        return outcome




class SDPAttn(nn.Module):
    def __init__(self,d_embed,d_k,seq_len,h1):
        super().__init__()
        self.d_k = d_k
        self.d_model = d_k*h1
        self.q_layer = FCLayer(d_embed,self.d_model)
        self.k_layer = FCLayer(d_embed,self.d_model)
        self.v_layer = FCLayer(d_embed,self.d_model)
        self.pad_masking = PadMasking(seq_len)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x,n):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)
        outcome = torch.matmul(Q,torch.transpose(K,-1,-2))
        outcome = outcome/np.sqrt(self.d_k)
        outcome = self.pad_masking(outcome,n)
        outcome = self.softmax(outcome)
        outcome = torch.matmul(outcome,V)
        return outcome




class PadMasking(nn.Module):
    def __init__(self,seq_len=500):
        super().__init__()
        self.seq_len = seq_len
        self.mask_matrix = torch.ones(self.seq_len,self.seq_len) * (-10e+8)
    
    def forward(self,x,n):
        self.mask_matrix[:,0:n,0:n] = x
        return self.mask_matrix




class FCLayer(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        self.h = h
        self.w = w
        self.matrix = torch.randn(self.h,self.w)

    def forward(self,x):
        x = torch.matmul(x,self.matrix)
        bias = torch.randn(x.size())
        x = x + bias
        return x





if __name__=='__main__':
    x, n, target = (torch.randn(128,500,3),410,0)

    encoder = Encoder(d_embed=3,d_k=128,seq_len=500,h1=2,N1=4)
    context = encoder(x,n)