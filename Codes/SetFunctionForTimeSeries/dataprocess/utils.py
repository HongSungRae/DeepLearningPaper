import pandas as pd
import os
import time


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

if __name__=='__main__':
    check()