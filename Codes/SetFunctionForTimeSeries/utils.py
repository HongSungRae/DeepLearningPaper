import torch
import torch.nn as nn
import numpy as np


def cuda_checker():
    # 현재 Setup 되어있는 device 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

if __name__ == '__main__':
    print(torch.__version__)
    cuda_checker()