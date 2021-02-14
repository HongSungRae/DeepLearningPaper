import pandas as pd
import numpy as np
import os
import time
from utils import time_encoding

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



def txt_to_csv(folder,target):
    start = time.time()

    target = pd.read_csv(target)
    df = pd.DataFrame(np.zeros(4000,2),columns=['S','y'])

    general_features = ["RecordID","Age","Gender","Height","ICUType","Weight"]
    features = ["Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
                "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
                "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP", "NISysABP", "Na",
                "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
                "TroponinT", "Urine", "WBC", "pH"]

    txts = next(os.walk(folder))[2] #list 4000

    for i in range(len(txt)):
        data = pd.read_csv(folder+txts[i]) # col0 : Time, col1 : Parameter, col3 : Value
        ID = int(list(txt[i].split('.'))[0])
        length = 0
        S_i = []

        '''기초정보행지우기'''
        for j in range(len(data)):
            time = list(map(int,data.iloc[j,0].split(':')))
            if time[0] == 0 and time[1] == 0:
                #해당행 지우기
            else:
                length = len(data)
                break
        

        '''time에 대한 프로세싱을 전부 해주고 mordality로 sorting'''
        for k in range(len(data)):
            time = list(map(int,data.iloc[k,0].split(':')))
            data.iloc[k,0] = #타임 프로세싱
            data.iloc[k,1] = features.index(data.iloc[k,1])
        else:
            #sorting



        '''s_i = (t,z,m)꼴로 만들어 S_i에 넣고 label(target) 붙여주기
            그 다음, df에 추가하기
        '''
        for k in range(len(data)):
            s_i = (data.iloc[k,0],
                   data.iloc[k,2],
                   data.iloc[k,1])
            S_i.append(s_i)
        else:
            target = ###
            df.iloc[i,0] = S_i
            df.iloc[i,1] = target
    
    print('=== Done ! ===')
    print('Time taken : {}'.format(time.time()-start))


if __name__=='__main__':
    #get_length()
    #txt_to_csv(A_root+'/set-a/',A_root+'/Outcomes-a.csv')