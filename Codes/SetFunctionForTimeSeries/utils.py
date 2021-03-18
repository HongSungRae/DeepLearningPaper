import torch
import torch.nn as nn
import numpy as np


def cuda_checker():
    # 현재 Setup 되어있는 device 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))


def load_model(model_name):
    PATH = '/daintlab/data/sr/paper/setfunction/trained_models/'
    model = torch.load(PATH+model_name)
    model.eval()
    return model


def load_data(forwhat=False):
    PATH = '/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/'
    df_a = pd.read_csv(PATH + 'A/set-a/A-dataset.csv')
    df_b = pd.read_csv(PATH + 'B/set-b/B-dataset.csv')
    df_c = pd.read_csv(PATH + 'C/set-c/C-dataset.csv')
    full_df = pd.concat([df_a,df_b,df_c])
    del df_a,df_b,df_c    
    train_size = int(0.8 * len(full_df))
    if forwhat==False:#train dataset
        return full_df[0:train_size]
    else:#test dataset
        return full_df[train_size:]

if __name__ == '__main__':
    print(torch.__version__)
    cuda_checker()

    model = load_model('SeFT_01.pt')