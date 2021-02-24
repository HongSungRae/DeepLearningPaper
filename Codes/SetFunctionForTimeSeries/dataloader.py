import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class myDataLoader(Dataset):
    def __init__(self,df,seq_len,d_embed=3):
        self.df = df
        self.seq_len = seq_len
        self.d_embed = d_embed
        self.df = self.df.dropna(axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        temp_x = self.df.iloc[index,0]
        temp_x = torch.FloatTensor(temp_x)
        n = len(temp_x)
        x = torch.ones(self.seq_len,self.d_embed)
        x[0:n] = temp_x
        target = self.df.iloc[index,1]
        return x, n, target



if __name__ == '__main__':
    main()