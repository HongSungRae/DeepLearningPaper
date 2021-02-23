import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class myDataLoader(Dataset):
    def __init__(self,df,seq_len):
        self.df = df
        self.seq_len = seq_len
        self.df = self.df.dropna(axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        x = self.df.iloc[index,0]
        x = torch.FloatTensor(x)
        n = len(x)
        # !!self.seq_len에 맞게 가공해야함!!
        target = self.df.iloc[index,1]
        return x, n, target



if __name__ == '__main__':
    main()