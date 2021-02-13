import pandas as pd
import os
import time

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
    target = pd.read_csv(target)
    general_features = ["RecordID","Age","Gender","Height","ICUType","Weight"]
    features = ["Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
                "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
                "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP", "NISysABP", "Na",
                "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
                "TroponinT", "Urine", "WBC", "pH"]

    txts = next(os.walk(folder))[2]
    length = len(txts)
    for i in range(length):
        data = pd.read_csv(folder+txts[i])
        ID = data.iloc[0,2]
        #print(txts[i])
        #for j in range(6): # 일부 애들은 00:00에 너무 많은게 조사됨.
        for k in range(6,len(data)):
            time = list(map(int,data.iloc[k,0].split(':')))
            s_i = ((time[0]*60+time[1])*0.1,
                    data.iloc[k,2],
                    features.index(data.iloc[k,1]))
            print(s_i)
            break

        # 처음 012345 는 기본정보 -> 논문에서는 안 쓴것으로 보임
        # col0 : Time, col1 : Parameter, col3 : Value
        # Time은 HH:MM 꼴이지만 어느순간부터 HH:MM:00 이런식으로 나오는 애들이 있음
        # set을 어떻게 저장? List? tensor? dataframe?
    else:
        print('=== Done ! ===')


if __name__=='__main__':
    #get_length()
    txt_to_csv(A_root+'/set-a/',A_root+'/Outcomes-a.csv')