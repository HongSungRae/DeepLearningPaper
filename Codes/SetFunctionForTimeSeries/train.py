from models.transformer_fclayer import Transformer
from models.seft import SeFT
from dataloader import MyDataLoader
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time



def train_model(model,dataloader,epoch):
    start = time.time()
    is_cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if is_cuda else 'cpu')
    device = torch.device(3)

    model = model
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.BCELoss()

    total_batch = len(dataloader)
    print('total batch : {}'.format(total_batch))
    train_loss_list = []

    for eph in range(epoch):
        print('epoch / epochs = {} / {}'.format(eph+1,epoch))
        loss_learning = 0.0
        sum_loss = 0.0
        
        for i,data in enumerate(dataloader):
            x, n, target = data
            
            if is_cuda:
                x = x.float().cuda(device)
                n = n.float().cuda(device)
                target = target.float().cuda(device)
            
            optimizer.zero_grad()
            y_hat = model(x,n)

            loss = criterion(y_hat.float(),target.float()) # hat이 먼저 target이 나중에 와야한다
            loss_learning += loss.item()
            sum_loss += loss.item()
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optimizer.step()

            torch.cuda.empty_cache()

            if i % 10 == 9:    # print every 10 mini-batches
                print('[epoch : %d, iter : %5d] loss: %.3f' %
                      (eph + 1, i + 1, loss_learning / 10))
                #print(torch.sum(model.fc3.weight))
                loss_learning = 0.0
        else:
            train_loss_list.append(round(sum_loss/i,5))
    else:
        print("Training has completed. Time taken : {} Seconds".format(time.time()-start))
    return model, train_loss_list


def train_and_save(model,dataloader,epoch,save_name):
    PATH = '/daintlab/data/sr/paper/setfunction/trained_models/'
    trained_model, train_loss_list = train_model(model,dataloader,epoch)
    torch.save(trained_model, PATH + save_name)
    return train_loss_list

if __name__ == '__main__':
    #model = Transformer().cuda()
    device = torch.device(3)
    model = SeFT().cuda(device)

    PATH = '/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/'


    '''
    """ train A Set """
    ##################
    df_a = pd.read_csv(PATH + 'A/set-a/A-dataset.csv')
    dataset_a = MyDataLoader(df_a,1024)
    dataloader_a = DataLoader(dataset_a, shuffle=False, batch_size=64,drop_last=True)
    train_loss_list = train_and_save(model,dataloader_a,30,"SeFT_01.pt")
    print(train_loss_list)
    ##################
    """            """
    '''
    """ train B Set """
    ##################
    df_b = pd.read_csv(PATH + 'B/set-b/B-dataset.csv')
    dataset_b = MyDataLoader(df_b,1024)
    dataloader_b = DataLoader(dataset_b, shuffle=False, batch_size=64,drop_last=True)
    train_loss_list = train_and_save(model,dataloader_b,30,"SeFT_02.pt")
    print(train_loss_list)
    ##################
    """            """