from torch import nn
from torch.nn.functional import normalize
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MoE, self).__init__()
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x, A_list):
        
        weights = self.gating(x)
        
        outputs = torch.stack(A_list, dim=2)
        
        weights = weights.unsqueeze(1).expand_as(outputs)
        
        res = torch.sum(outputs * weights, dim=2)
        
        return res


class Gating(nn.Module):
    def __init__(self, input_dim,
                num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

        

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        logits = torch.softmax(self.layer4(x), dim=1)
       
        return logits


class MoEGCL(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(GCFAggMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim*view, high_feature_dim),
        )
        self.view = view

        self.MoE = MoE(low_feature_dim*view, view)

        self.ffn = FeedForward(d_model=low_feature_dim*view, d_ff = 256)
        
    
   
    
    
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return  xrs, zs, hs

    def computeA(self, x, mode):
        if mode == 'cos':
            a = F.normalize(x, p=2, dim=1)
            b = F.normalize(x.T, p=2, dim=0)
            A = torch.mm(a, b)
            A = (A + 1) / 2
        if mode == 'kernel':
            x = torch.nn.functional.normalize(x, p=1.0, dim=1)
            a = x.unsqueeze(1)
            A = torch.exp(-torch.sum(((a - x.unsqueeze(0)) ** 2) * 1000, dim=2))

        if mode == 'knn':
            dis2 = (-2 * x.mm(x.t())) + torch.sum(torch.square(x), axis=1, keepdim=True) + torch.sum(
                torch.square(x.t()), axis=0, keepdim=True)
            A = torch.zeros(dis2.shape).cuda()
            A[(torch.arange(len(dis2)).unsqueeze(1), torch.topk(dis2, 10, largest=False).indices)] = 1
            A = A.detach()
        if mode=='sigmod':

            A=1/(1+torch.exp(-torch.mm(x,x.T)))

        return A


    def MoEGF(self, xs):
        zs = []
        Alist = []
        for v in range(self.view):
            x = xs[v]

            A = self.computeA(F.normalize(x), mode='knn')
            
            Alist.append(A)

            z = self.encoders[v](x)
            zs.append(z)
            
        commonz = torch.cat(zs, 1)

        Gf = self.MoE(commonz, Alist)
        
        z_refined, at = computegcn(commonz, Gf)

        z_refined = self.ffn(z_refined)

        z_refined = normalize(self.Common_view(z_refined), dim=1)
        
        
        return z_refined, Gf


def computegcn(z, Af):
    D = torch.sum(Af, dim=1)
    D = 1 / D
    D = torch.sqrt(D)
    D = torch.diag(D)
    ATi = torch.matmul(torch.matmul(D, Af), D)
    z_refine = torch.matmul(ATi, torch.matmul(ATi, z))

    return z_refine, ATi



def graph_fusion(graphlist, wlist):
    torch.cuda.empty_cache()
    with torch.no_grad():
        fusiongraph =torch.sum(torch.mul(graphlist, wlist.T.unsqueeze(dim=1)), dim=0)/ (torch.sum(wlist))
    for v in range(graphlist.shape[0]):
        w = 1 / (2 * (torch.norm((graphlist[v] - fusiongraph), p='fro')))
        wlist.T[v] = w

    return fusiongraph, wlist