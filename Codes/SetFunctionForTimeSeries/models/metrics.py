import numpy as np
import pandas as pd
import torch
import torch.nn as nn

"""
SeFT model uses BinaryCrossEntropyLoss for training.
nn.BCELoss() gives convenient use.

For test(or validation), Accuracy, AUPRC and AUROC
should be calculated

metrics.py contains this kinds of metrics
"""

def get_1_or_0(y_hat):
    bs = y.shape[0]
    for i in range(bs):
        if y_hat[i] > 0.5:
            y_hat[i] = 1.0
        else:
            y_hat[i] = 0.0
    return y_hat


def accuracy(y,y_hat):
    bs = y.shape[0]
    correct = 0
    y_hat = get_1_or_0(y_hat)
    
    for i in range(bs):
        if y[i] == y_hat[i]:
            correct += 1
    score = (correct/bs)*100
    return score


def auprc(y,y_hat):
    return None

if  __name__ == "__main__":
    y = torch.ones(64,1)
    y_hat = torch.randn(64,1)
    print(accuracy(y,y_hat))
    pass