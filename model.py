import torch
import torch.nn as nn
import torch.nn.functional as F

class PNet(nn.Module):
    def __init__(self, nhidden, dropout=0.3):
        super(PNet, self).__init__()
        self.nhidden = nhidden

        self.conv0a = torch.nn.Conv1d(3, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, nhidden*3),
            )

    def forward(self, x):
        # point
        # mlp (64,64)
        batch_size = x.size(0)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        x = self.fc(x)
        # reshape
        x = x.view(batch_size, self.nhidden, 3)
        
        return x


class GNet(nn.Module):
    def __init__(self, nhidden, dropout=0.3):
        super(GNet, self).__init__()
        self.nhidden = nhidden

        self.fcn = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, nhidden),
            )

    def forward(self, x, z):
        # normal
        batch_size = x.size(0)
        
        # apply
        out = self.fcn(z)
        z = z.unsqueeze(2)

        assert x.size(1) == self.nhidden and x.size(2) == 3
        x = torch.matmul(x, z)
        x = x.squeeze(2)

        out = torch.cat((x, out), dim=1)
        return out


class Net(nn.Module):
    def __init__(self, nfeat, nhidden, dropout=0.3, outdims=3):
        super(Net, self).__init__()
        self.nfeat = nfeat
        self.nhidden = nhidden
        self.dropout = dropout
        self.outdims = outdims

        self.pnet = PNet(nhidden=self.nhidden, dropout=self.dropout)
        self.GNetList = nn.ModuleList([GNet(
            nhidden=self.nhidden, 
            dropout=self.dropout) for _ in range(nfeat)])

        self.fc = nn.Sequential(
            nn.BatchNorm1d(2*nfeat*nhidden),
            nn.Linear(2*nfeat*nhidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, outdims),
            )


    def forward(self, x, z):
        # x:(batch_size, 3, k), z:(batch_size, nfeat*3)
        x = self.pnet(x)
        out = torch.cat([subnet(x, z[:, i*3:i*3+3]) for i, subnet in enumerate(self.GNetList)], dim=1)
        out = self.fc(out)
        out = F.normalize(out)

        return out

