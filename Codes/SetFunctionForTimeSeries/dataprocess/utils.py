import pandas as pd
import os
import time
import numpy as np


def check():
    '''
    A set의 어떤 데이터가
    더 있는지 (4002!=4000) 확인
    '''
    start = time.time()
    root = '/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/'
    A_root = root + 'A'
    A_outcomes = pd.read_csv(A_root+'/Outcomes-a.csv')
    A_outcomes = A_outcomes['RecordID'].tolist()
    A_list = next(os.walk(A_root+'/set-a'))[2]

    for i in range(len(A_list)):
        ID = int(list(A_list[i].split('.'))[0])
        if ID not in A_outcomes:
            print(ID)
    else:
        print("=== Done ! ===")
        print('Time taken : {}'.format(time.time()-start))


def time_encoding(h,m,i,tau=500):
    t = h*60+m
    k = i//2
    if i%2 == 0:
        t = np.sin(t/t**(2*k/tau))
    else:
        t = np.cos(t/t**(2*k/tau))
    return round(t,6)




if __name__=='__main__':
    check()
    t = time_encoding(10,17,30)
    print(t)