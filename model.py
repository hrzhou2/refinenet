import torch
import torch.nn as nn
import torch.nn.functional as F

class PointModule(nn.Module):
    def __init__(self, nhidden, dropout=0.3, out_channel=3):
        super(PointModule, self).__init__()
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
            nn.Linear(256, nhidden*out_channel),
            )

    def forward(self, x):
        # process point data

        # pointnet
        # mlp (64,64)
        batch_size = x.shape[0]
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)

        # output dim
        x = self.fc(x)
        
        return x


class HeightModule(nn.Module):
    def __init__(self, nhidden, nfeat, dropout=0.3, out_channel=3):
        super(HeightModule, self).__init__() 

        self.conv = nn.Sequential(
            nn.Conv2d(nfeat, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, nhidden*out_channel),
            )

    def forward(self, x):
        # process height matrix

        batch_size = x.shape[0]

        # output feature        
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


class Connection(nn.Module):
    def __init__(self, nhidden, dropout=0.3):
        super(Connection, self).__init__()
        self.nhidden = nhidden

        self.fcn = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, nhidden),
            )

    def forward(self, normals, feat_list):
        # normal refinement

        batch_size = normals.shape[0]

        # skip connection
        identity = normals

        # apply
        out_feat = []
        for feat in feat_list:
            normals = normals.unsqueeze(2) # (B, x, 1)
            feat = feat.view(batch_size, self.nhidden, -1) # (B, h, x)
            normals = torch.matmul(feat, normals).squeeze(2) # (B, h)
            out_feat.append(normals)

        # collect all features
        residual = self.fcn(identity)
        out_feat.append(residual)

        return torch.cat(out_feat, dim=1)
        


class Net(nn.Module):
    def __init__(self, config, nfeat, outdims=3):
        super(Net, self).__init__()
        # network parameters
        self.nfeat = nfeat
        self.nhidden = config.network.feat_dim
        self.dropout = config.network.dropout
        self.outdims = outdims

        # Feature Modules
        self.features = config.feature.in_features
        channel = 3
        for feat_ in self.features:
            # Build blocks
            if feat_ == 'points':
                self.point_block = PointModule(nhidden=self.nhidden, dropout=self.dropout, out_channel=channel)
            elif feat_ == 'heights':
                self.height_block = HeightModule(nhidden=self.nhidden, nfeat=self.nfeat, dropout=self.dropout, out_channel=channel)
            else:
                raise ValueError('Unknown feature module: {:s}'.format(feat_))
            # Update feature dim
            channel = self.nhidden

        self.connect_blocks = nn.ModuleList([Connection(nhidden=self.nhidden, dropout=self.dropout) for _ in range(self.nfeat)])

        # decoder layer
        num_blocks = len(self.features)+1
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(num_blocks*self.nfeat*self.nhidden),
            nn.Linear(num_blocks*self.nfeat*self.nhidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, outdims),
            )


    def forward(self, data):

        # Feature Modules
        feat_list = []
        for feat_ in self.features:
            # Get processed features
            if feat_ == 'points':
                feat = self.point_block(data['points'].transpose(2, 1))
                feat_list.append(feat)
            elif feat_ == 'heights':
                feat = self.height_block(data['heights'])
                feat_list.append(feat)

        # refine normal and output
        nf = data['nf']
        out = torch.cat([block(nf[:, idx*3:idx*3+3], feat_list) for idx, block in enumerate(self.connect_blocks)], dim=1)

        # output normals
        out = self.decoder(out)
        out = F.normalize(out)

        return out

