import torch
from network import MoEGCL
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.nn.functional as F




Dataname = 'LGG'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=300)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



args.temperature_f = float(args.temperature_f)




dataset, dims, view, data_size, class_num = load_data(args.dataset)


data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def fine_tune(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs = model(xs)
        commonz, S = model.MoEGF(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

if not os.path.exists('./models'):
    os.makedirs('./models')
model = MoEGCL(view, dims, args.low_feature_dim, args.high_feature_dim,  device)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, args.temperature_f, device).to(device)


epoch = 1
while epoch <= args.rec_epochs:
    pre_train(epoch)
    epoch += 1

while epoch <= args.rec_epochs + args.fine_tune_epochs:
    fine_tune(epoch)

    if epoch == args.rec_epochs + args.fine_tune_epochs:
        valid(model, device, dataset, view, data_size, class_num)
        state = model.state_dict()
        torch.save(state, './models/' + args.dataset + '.pth')
        print('Saving model...')
    epoch += 1

    