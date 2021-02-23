import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np
import pandas as pd



class Transformer(nn.Module):
    def __init__(self,d_embed=3,d_k=3,seq_len=500,h1=8,h2=8,h3=8,N1=6,N2=6):
        super().__init__()
        # hyper-parameters #
        self.d_embed = d_embed
        self.d_k = d_k
        self.seq_len = seq_len
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.N1 = N1
        self.N2 = N2
        # # # # # # # # # # # 
        self.encoder = Encoder(self.d_k,self.seq_len,self.h1,self.N1)
        self.decoder = Decoder(self.h2,self.h3,self.N2)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self,x)
    pass



class Generator(nn.Module):
    pass

class Encoder(nn.Module):
    def __init__(self,d_embed=3,d_k,seq_len,h1,N1):
        super().__init__()
        self.d_embed = d_embed
        self.d_k = d_k
        self.seq_len = seq_len
        self.h1 = h1
        self.N1 = N1
        self.list_multiheadattn = []
        self.list_feedforward = []
        for i in range(self.N1):
            self.list_multiheadattn.append(MultiHeadAttn(self.d_embed,self.d_k,self.seq_len,self.h1))
            self.list_feedforward.append(FeedForward(self.d_embed,self.d_k))
        #self.multiheadattn = MultiHeadAttn(self.d_embed,self.d_k,self.seq_len,self.h1)
        #self.feedforward = FeedForward(self.d_embed,self.d_k)

    def forward(self,x):
        for i in range(self.N1):
            x = self.list_multiheadattn[i](x)
            x = self.list_feedforward[i](x)
        return x



class FeedForward(nn.Module):
    pass



class MultiHeadAttn(nn.Module):

    def forward(self,x): # residual and normalization
        return outcome + x
    pass


class SDPAttn(nn.Module):
    def __init__(self,d_embed,d_k,h,**kwargs):
        super().__init__()
        self.d_k = d_k
        self.q_layer = QKV_FCLayer(d_embed,d_k*h)
        self.k_layer = QKV_FCLayer(d_embed,d_k*h)
        self.v_layer = QKV_FCLayer(d_embed,d_k*h)
        self.pad_masking = PadMasking(**kwargs)
    
    def forward(self,x):
        Q,_ = self.q_layer(x)
        K,_ = self.k_layer(x)
        V,n = self.v_layer(x)
        print(n)
        outcome = torch.matmul(Q,torch.transpose(K,-1,-2))
        outcome = outcome/np.sqrt(self.d_k)
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
        self.mask_matrix[0:n,0:n] = 1.0
        print(self.mask_matrix[0])
        
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
    dummy = torch.ones(410,3) # dataloader에서 가공
    sdp = SDPAttn(3,3,8)
    print(sdp(dummy)[-1])


    model = Transformer()
    y = model(dummy)