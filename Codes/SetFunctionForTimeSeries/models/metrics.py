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

def neg_or_pos(y_hat,threshold):
    bs = y_hat.shape[0]
    for i in range(bs):
        if y_hat[i] > threshold:
            y_hat[i] = 1.0
        else:
            y_hat[i] = 0.0
    return y_hat


def confuse_matrix(y,y_hat,threshold=0.5):
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


def roc(y,y_hat):
    assert y.shape==y_hat.shape

    FPR = []
    TPR = []
    y = neg_or_pos(y,0.5)
    y = y.view(y.shape[0]).tolist()
    y_hat = y_hat.view(y_hat.shape[0]).tolist() # it returns probabilities
    df = pd.DataFrame(data={'y':y,'y_hat':y_hat})
    df = df.sort_values(['y_hat'],ascending=[False]) # 내림차순
    eps = 10e-3
    P = len(df[df['y']==1.0]) + eps
    N = len(df[df['y']==0.0]) + eps
    for i in df['y_hat']:
        tmp_p = df[df['y_hat'] >= i]
        TP = len(tmp_p[tmp_p['y'] == 1.0])
        tmp_TPR = TP/P
        tmp_n = df[df['y_hat'] >= i]
        FP = len(tmp_n[tmp_n['y'] == 0.0])
        tmp_FPR = FP/N
        TPR.append(tmp_TPR)
        FPR.append(tmp_FPR)
    return TPR, FPR


def auroc(TPR,FPR,ret_list=False):
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
    y = torch.ones(64,1)
    y_hat = torch.randn(64,1)
    TPR, FPR = roc(y,y_hat)
    TPR,FPR,score = auroc(TPR,FPR,True)
    print(score) # in this dummy setting, outcomes would be absolutely same whenever you run the code
                 # because all dummy target == 1.0(true)