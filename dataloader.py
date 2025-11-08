import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import random
import sklearn


class LGG():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'LGG.mat')
        self.Y = data['Y'].astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
        
    def __len__(self):
        return 267
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    
    elif dataset == "LGG":
        dataset = LGG('./data/')
        dims = [2000, 2000, 333, 209]
        view = 4
        data_size = 267
        class_num = 3
    
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
