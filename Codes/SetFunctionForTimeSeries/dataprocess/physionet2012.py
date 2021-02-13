import pandas as pd
import os

root = '/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/'
A_root = root + 'A'
B_root = root + 'B'
C_root = root + 'C'


def get_length():
    A_outcomes = pd.read_csv(A_root+'/Outcomes-a.csv')
    B_outcomes = pd.read_csv(B_root+'/Outcomes-b.csv')
    C_outcomes = pd.read_csv(C_root+'/Outcomes-c.csv')
    print(A_outcomes.head(10))
    print('target A : {}, target B : {}, target C : {}'.format(len(A_outcomes),len(B_outcomes),len(C_outcomes)))
    print('input A : {}. input B : {}, input C : {}'.format(
        len(next(os.walk(A_root+'/set-a'))[2]),
        len(next(os.walk(B_root+'/set-b'))[2]),
        len(next(os.walk(C_root+'/set-c'))[2])))

def txt_to_csv(target_folder):
    txts = next(os.walk(target_folder))[2]
    length = len(txts)
    for i in range(length):
        print(txts[i])
        a = pd.read_csv(target_folder+txts[i])
    print('done')    

    #os.walk(target_folder)


if __name__=='__main__':
    get_length()

    txt_to_csv(A_root+'/set-a/')