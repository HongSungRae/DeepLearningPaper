from models.transformer_fclayer import Transformer
from models.seft import SeFT
from dataloader import MyDataLoader
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd



def train_model(model,dataloader,epoch):
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

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
        count = 0
        for i,data in enumerate(dataloader):
            count += 1
            x, n, target = data
            '''
            if is_cuda:
                x = x.float().cuda()
                #n = n.float().cuda()
                target = target.float().cuda()
            '''
            optimizer.zero_grad()
            y_hat = model(x,n)
            print(y_hat.shape)
            loss = criterion(y_hat.float(),target.float()) # hat이 먼저 target이 나중에 와야한다
            loss_learning += loss
            sum_loss += loss
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph=True)
            optimizer.step()
           
            if i % 10 == 9:    # print every 10 mini-batches
                print('[epoch : %d, iter : %5d] loss: %.3f' %
                      (eph + 1, i + 1, loss_learning / 10))
                print(torch.sum(model.linear2.weight))
                loss_learning = 0.0
        else:
            train_loss_list.append(round(sum_loss/count,5))
    return model, train_loss_list



if __name__ == '__main__':
    #model = Transformer().cuda()
    model = SeFT()
    df = pd.read_csv('/daintlab/data/sr/paper/setfunction/tensorflow_datasets/root/tensorflow_datasets/downloads/extracted/A/set-a/A-dataset.csv')
    dataset = MyDataLoader(df,1024)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=64)

    trained_model, train_loss_list = train_model(model,dataloader,30)
    print(train_loss_list)