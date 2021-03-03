import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np
import pandas as pd



class Transformer(nn.Module):
    def __init__(self,d_embed=3,d_k=128,seq_len=1024,h1=2,h2=2,h3=2,N1=4,N2=4):
        super().__init__()
        self.encoder = Encoder(d_embed,d_k,seq_len,h1,N1)
        self.linear1 =  nn.Linear(seq_len*d_embed,1024)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,n):
        c = self.encoder(x,n)
        c = c.reshape(c.shape[0],-1)
        y_pred = self.linear1(c)
        y_pred = self.relu(y_pred)
        y_pred = self.linear2(y_pred)
        y_pred = self.sigmoid(y_pred)
        return y_pred





class Masked_MultiHeadAttn(nn.Module):
    def __init__(self,d_embed,d_k,seq_len,h1):
        super().__init__()
        self.masked_multiheadattn = MultiHeadAttn(d_embed,d_k,seq_len,h1,decoding=True)
    
    def forward(self,x,n):
        outcome = self.masked_multiheadattn(x,n)
        return outcome





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
    def __init__(self,d_embed,d_k,seq_len,h1,decoding=False):
        super().__init__()
        self.sdpattn = SDPAttn(d_embed,d_k,seq_len,h1,decoding)
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
    def __init__(self,d_embed,d_k,seq_len,h1,decoding):
        super().__init__()
        self.d_k = d_k
        self.d_model = d_k*h1
        self.decoding = decoding
        self.q_layer = FCLayer(d_embed,self.d_model)
        self.k_layer = FCLayer(d_embed,self.d_model)
        self.v_layer = FCLayer(d_embed,self.d_model)
        self.pad_masking = PadMasking(seq_len)
        self.sub_masking = SubMasking(seq_len)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x,n,c=None,**kargs):
        if c == None:
            Q = self.q_layer(x)
            K = self.k_layer(x)
            V = self.v_layer(x)
        else:
            Q = self.q_layer(c)
            K = self.k_layer(c)
            V = self.v_layer(x)
        outcome = torch.matmul(Q,torch.transpose(K,-1,-2))
        outcome = outcome/np.sqrt(self.d_k)
        outcome = self.pad_masking(outcome,n)
        if self.decoding == True:
            outcome = self.sub_masking(outcome,n)
        outcome = self.softmax(outcome)
        outcome = torch.matmul(outcome,V)
        return outcome




class SubMasking(nn.Module):
    def __init__(self,seq_len=500):
        super().__init__()
        self.seq_len = seq_len

    def forward(self,x,n):
        if len(x.shape) == 3:
            for i in range(x.shape[0]):
                for j in range(n-1):
                    x[i,j,j+1:] = -10e+8
        else:
            for j in range(n-1):
                x[j,j+1:] = -10e+8
        
        return x



class PadMasking(nn.Module):
    def __init__(self,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.mask_matrix = torch.ones(self.seq_len,self.seq_len,device='cuda') * (-10e+8)
    
    def forward(self,x,n):
        print(type(n))
        if len(x.shape) == 3:
            batch = x.shape[0]
            for i in range(batch):
                temp = self.mask_matrix
                temp[0:n,0:n] = x[i,:int(n[i]),:int(n[i])]
                x[i] = temp
            return x
        else:
            self.mask_matrix[0:n,0:n] = x[0:n,0:n]



class FCLayer(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        self.h = h
        self.w = w
        self.matrix = torch.randn(self.h,self.w)
        self.is_cuda = torch.cuda.is_available()
        #self.device = torch.device('cuda' if self.is_cuda else 'cpu')

    def forward(self,x):
        x = torch.matmul(x,self.matrix)
        if x.get_device()==0:
            bias = torch.randn(x.size(),device='cuda')
        else:
            bias = torch.randn(x.size())
        x = x + bias
        return x





if __name__=='__main__':
    x, n, target = (torch.randn([128,1024,3]),410,0)
    '''
    encoder = Encoder(d_embed=3,d_k=128,seq_len=512,h1=2,N1=4)
    context = encoder(x,n)
    print('context :' ,context.shape)
    '''
    transformer = Transformer()
    y_pred = transformer(x,n)
    print('y_pred :',y_pred.shape)
    print(y_pred)