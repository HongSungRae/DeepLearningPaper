import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy,auroc

"""
SeFT model uses BinaryCrossEntropyLoss for training.
nn.BCELoss() gives convenient use.

For test(or validation), Accuracy, AUPRC and AUROC
should be calculated

metrics.py contains this kinds of metrics
"""

def neg_or_pos(y_hat,threshold):
    bs = y_hat.shape[0]
    for i in range(bs):
        if y_hat[i] >= threshold:
            y_hat[i] = 1.0
        else:
            y_hat[i] = 0.0
    return y_hat



def confusion_matrix(y_hat,y,threshold=0.5):
    bs = y.shape[0]
    correct = 0
    y_hat = neg_or_pos(y_hat,threshold)
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
    eps = 1e-3
    accuracy = (tp+tn)/bs
    precision = tp/(tp+fp+eps)
    recall = tp/(tp+fn+eps)
    f1 = 2*(precision*recall)/(precision+recall+eps)
    #print(tp,tn,fp,fn)
    return accuracy, precision, recall, f1


def prc(y_hat,y):
    assert y.shape==y_hat.shape

    PRECISION = []
    RECALL = []
    eps = 10e-8

    y = y.view(y.shape[0]).tolist()
    y_hat = y_hat.view(y_hat.shape[0]).tolist() # it returns probabilities
    df = pd.DataFrame(data={'y':y,'y_hat':y_hat})
    df = df.sort_values(by='y_hat',ascending=False) # 내림차순

    tp = 0
    P = 0 + eps # denominator of precision. TP + FP
    R = len(df[df['y']==1]) + eps # denominator of recall. TP + FN
    for i in df['y_hat']:
        P += 1
        temp = df[df['y_hat']>=i]
        if temp.iloc[-1,0] == 1:
            tp += 1
        precision = tp/P
        recall = tp/R
        PRECISION.append(precision)
        RECALL.append(recall)
    return PRECISION,RECALL



def my_auprc(y_hat,y,ret_list=False):
    PRECISION, RECALL = prc(y_hat,y)
    score = 0
    for i in range(len(RECALL)-1):
        temp = (RECALL[i+1]-RECALL[i])*(PRECISION[i]+PRECISION[i+1])/2
        score += temp
    if ret_list==True:
        return PRECISION,RECALL,score
    else:
        return score




def roc(y_hat,y):
    assert y.shape==y_hat.shape

    FPR = []
    TPR = []
    y = y.view(y.shape[0]).tolist()
    y_hat = y_hat.view(y_hat.shape[0]).tolist() # it returns probabilities
    df = pd.DataFrame(data={'y':y,'y_hat':y_hat})
    df = df.sort_values(by='y_hat',ascending=False) # 내림차순
    eps = 10e-8
    P = len(df[df['y']==1.0]) + eps
    N = len(df[df['y']==0.0]) + eps
    for i in df['y_hat']:
        temp = df[df['y_hat'] >= i]
        TP = len(temp[temp['y'] == 1])
        FP = len(temp[temp['y'] == 0])
        tmp_TPR = TP/P
        tmp_FPR = FP/N
        TPR.append(tmp_TPR)
        FPR.append(tmp_FPR)
    return TPR, FPR


def my_auroc(y_hat,y,ret_list=False):
    TPR,FPR = roc(y_hat,y)
    score = 0
    TPR = [0] + TPR + [1]
    FPR = [0] + FPR + [1]
    for i in range(len(TPR)-1):
        temp = (TPR[i]+TPR[i+1])*(FPR[i+1]-FPR[i])/2
        score += temp
    if ret_list==True:
        return TPR,FPR,score
    else:
        return score


if  __name__ == "__main__":
    y = neg_or_pos(torch.randn(64,1),0)
    sigmoid = nn.Sigmoid()
    y_hat = torch.randn(64,1)
    y_hat = sigmoid(y_hat)

    TPR, FPR = roc(y_hat,y)
    print(my_auroc(y_hat,y))
    print(auroc(y_hat,y.long(),pos_label=1))
    print(my_auprc(y_hat,y))