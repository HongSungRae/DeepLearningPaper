import pandas as pd
import numpy as np
import os
import time as t
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



def txt_to_csv(folder,target,filename):
    start = t.time()

    target = pd.read_csv(target)
    target_ID = list(map(int,target['RecordID'].tolist()))
    target_y = target['In-hospital_death'].tolist()

    df = pd.DataFrame(np.zeros([4000,2]),columns=['S','y'])
    df['S'] = df['S'].astype('object')

    general_features = ["RecordID","Age","Gender","Height","ICUType","Weight"]
    features = ["Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
                "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
                "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP", "NISysABP", "Na",
                "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
                "TroponinT", "Urine", "WBC", "pH",None]
    exclusion = [140501, 150649, 140936, 143656, 141264, 145611,
                 142998, 147514, 142731, 150309, 155655, 156254]

    txts = next(os.walk(folder))[2] #list 4000

    for i in range(len(txts)):
        data = pd.read_csv(folder+txts[i]) # col0 : Time, col1 : Parameter, col3 : Value
        data = data.dropna(axis=0)
        length = len(data)
        ID = int(list(txts[i].split('.'))[0])
        
        if ID in exclusion:
            continue

        print(ID)
        S_i = []

        '''기초정보행지우기'''
        for j in range(length):
            time = list(map(int,data.iloc[0,0].split(':')))
            if time[0] == 0 and time[1] == 0:
                data = data.drop(data.index[0],axis=0)
            else:
                length = len(data)
                break
        

        '''time에 대한 프로세싱을 전부 해주고 mordality로 sorting'''
        for k in range(length):
            time = list(map(int,data.iloc[k,0].split(':')))
            data.iloc[k,0] = time_encoding(time[0],time[1],k)
            data.iloc[k,1] = features.index(data.iloc[k,1])
        else:
            # sorting
            data = data.sort_values(['Parameter'],ascending=[False])
            length = len(data)



        '''s_i = (t,z,m)꼴로 만들어 S_i에 넣고 label(target) 붙여주기
            그 다음, df에 추가하기
        '''
        for k in range(length):
            s_i = (data.iloc[k,0],
                   data.iloc[k,2],
                   data.iloc[k,1])
            S_i.append(s_i)
        
            
        if (len(S_i)>=3) and (len(S_i)<=1024):
            y = target_y[target_ID.index(ID)]
            df.at[i,'S'] = S_i
            df.loc[i,'y'] = y
        else:
            continue
    
    df.to_csv(folder + filename + ".csv",header=True,index=True)
    print('=== Done ! ===')
    print('Time taken : {}'.format(t.time()-start))
            


if __name__=='__main__':
    #get_length()
    txt_to_csv(A_root+'/set-a/',A_root+'/Outcomes-a.csv','A-dataset')
    #txt_to_csv(B_root+'/set-b/',B_root+'/Outcomes-b.csv','B-dataset')
    #txt_to_csv(C_root+'/set-c/',C_root+'/Outcomes-c.csv','C-dataset')