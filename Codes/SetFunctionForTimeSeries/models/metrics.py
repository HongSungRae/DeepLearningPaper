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
    bs = y_hat.shape[0]
    for i in range(bs):
        if y_hat[i] > 0.5:
            y_hat[i] = 1.0
        else:
            y_hat[i] = 0.0
    return y_hat


def confuse_matrix(y,y_hat):
    bs = y.shape[0]
    correct = 0
    y_hat = get_1_or_0(y_hat)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(bs):
        true = int(y[i])
        predict = int(y_hat[i])
        if (true==1) and (predict==1):
            tp += 1
        elif (true==1) and (predict==0):
            fn += 1
        elif (true==0) and (predict==0):
            tn += 1
        else:
            fp += 1
    #print(tp,tn,fp,fn,tp+tn+fp+fn)
    eps = 1e-3
    accuracy = (tp+tn)/bs
    precision = tp/(tp+fp+eps)
    recall = tp/(tp+fn+eps)
    f1 = 2*(precision*recall)/(precision+recall+eps)
    return accuracy, precision, recall, f1


def auprc(precision,recall):
    """
    plz input precision and recall as list form.
    """
    auprc_precision = [0] + precision
    auprc_recall = [0] + recall
    auprc = 0
    for i in range(1, len(auprc_precision)):
        temp_auprc = (auprc_Precision[i - 1] + auprc_precision[i]) * (auprc_recall[i] - auprc_recall[i - 1]) / 2
        auprc += temp_auprc
    return precision_list, recall_list


def auroc(y,y_hat):
    bs = y.shape[0]
    return None


if  __name__ == "__main__":
    y = torch.ones(64,1)
    y_hat = torch.randn(64,1)
    print(accuracy(y,y_hat))