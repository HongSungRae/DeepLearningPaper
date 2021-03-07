import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MyDataLoader(Dataset):
    def __init__(self,df,seq_len,d_embed=3):
        self.df = df
        self.seq_len = seq_len
        self.d_embed = d_embed
        self.df = self.df.dropna(axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        temp_x = self.df.iloc[index,1] #str
        temp_x = eval(temp_x) # '[(),(),()]' -> eval -> [(),(),()]
        temp_x = torch.FloatTensor(temp_x)
        n = torch.tensor(len(temp_x)).view(-1)
        #n = len(temp_x)
        x = torch.ones(self.seq_len,self.d_embed)
        x[0:len(temp_x)] = temp_x
        target = torch.tensor(self.df.iloc[index,2]).view(-1)
        #target = self.df.iloc[index,2]
        return x, n, target



if __name__ == '__main__':
    df = pd.read_csv('/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/A/set-a/A-dataset.csv')
    print(df.head())
    dataset = MyDataLoader(df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64, pin_memory=False)

    x, n, target = next(iter(dataloader))
    print("x.shape :",x.shape)
    print("n :",n.shape)
    print("target :",target.shape)