'''
==============================================================

RefineNet Point Cloud Normal Refinement Network
-> Training on PCPNet

==============================================================

Author: Haoran Zhou
Date: 2022-3-15

==============================================================
'''


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import json
import time
import argparse
from test import test_models
from importlib import import_module
from utils.easydict import EasyDict as edict
from pcpnet import MultiFeatureDataset as Dataset
from pcpnet import RandomPointcloudPatchSampler
from utils.loss import angle_degrees, compute_loss


TRAIN_NAME = __file__.split('.')[0]


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config
#       \******************/
#


parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='RefineNet Training on PCPNet.', help='description')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--model', type=str, default='model', help='Model to use')
parser.add_argument('--id_cluster', type=int, default=None, help='Network for i-th cluster.')
parser.add_argument('--trainset', type=str, default=None, help='Specify other training set')
parser.add_argument('--validateset', type=str, default=None, help='Specify other validation set')
parser.add_argument('--testset', type=str, default=None, help='Specify other test set')
# Testing
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--output', type=int, default=True, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default=None, help='Pretrained model for testing.')
parser.add_argument('--sparse_patches', type=int, default=True, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')

args =  parser.parse_args()


def PCPNetConfig():

    cfg = edict()

    #############
    # Directories
    #############

    cfg.dir = edict()
    cfg.dir.result = './results'
    cfg.dir.test = './test'


    ################
    # Dataset Config
    ################

    cfg.dataset = edict()

    # Path to dataset
    cfg.dataset.pointcloud_dir = '<*PATH-TO-YOUR-DATASET*>/pcpnet'
    cfg.dataset.normal_dir = '<*PATH-TO-NORMALS*>/initial_normals'
    cfg.dataset.cluster_dir = './cluster'
    cfg.dataset.train_shape_filenames = 'trainingset_whitenoise.txt'
    cfg.dataset.validate_shape_filenames = 'validationset_whitenoise.txt'
    cfg.dataset.test_shape_filenames = 'testset_whitenoise.txt'

    # dataset
    cfg.dataset.patches_per_shape = 100000
    cfg.dataset.cluster = 1


    ####################
    # Feature Processing
    ####################

    cfg.feature = edict()
    cfg.feature.patch_features = ['normal']

    # normal processing
    cfg.feature.filter_radius = 0.03
    cfg.feature.sigma_s = [1.0, 2.0]
    cfg.feature.sigma_r = [0.1, 0.2, 0.35, 0.5]
    cfg.feature.self_included = True

    # patch points
    cfg.feature.query = 'knn'
    cfg.feature.center = 'point'
    cfg.feature.query_k = 100
    cfg.feature.query_radius = 0.03
    cfg.feature.points_per_patch = 300

    # batch normals pca reorientation
    cfg.feature.use_pca = True


    #######################
    # Network Configuration
    #######################

    cfg.network = edict()
    cfg.network.feat_dim = 64
    cfg.network.dropout = 0.3


    ##########
    # Training
    ##########

    cfg.train = edict()
    cfg.train.lr = 0.0001
    cfg.train.batch_size = 1024
    cfg.train.max_epochs = 1000
    cfg.train.num_workers = 8
    cfg.train.patience = 20

    # Normal loss function
    # 'mse_loss': element-wise mean square error
    # 'ms_euclidean': mean square euclidean distance
    # 'ms_oneminuscos': mean square 1-cos(angle error)
    cfg.train.normal_loss = 'mse_loss'

    # optimizer
    cfg.train.momentum = 0.9
    cfg.train.weight_decay = 0.02


    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def main(config):

    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    # Set up folders for logs and checkpoints
    #timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.gmtime())
    outdir = os.path.join(config.dir.result, TRAIN_NAME, 'model_cidx{:d}'.format(config.dataset.cluster))
    config.dir.result = outdir
    indir = config.dataset.pointcloud_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('Output Path: {:s}'.format(outdir))


    ########################
    # Load Train/Val Dataset
    ########################

    # Build dataset
    train_dataset = Dataset(config, shape_list_filename=config.dataset.train_shape_filenames)
    validate_dataset = Dataset(config, shape_list_filename=config.dataset.validate_shape_filenames)

    # Dataset sampler
    train_sampler = RandomPointcloudPatchSampler(train_dataset, patches_per_shape=config.dataset.patches_per_shape)
    validate_sampler = RandomPointcloudPatchSampler(validate_dataset, patches_per_shape=config.dataset.patches_per_shape)

    # Data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                sampler=train_sampler,
                                                batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset,
                                                sampler=validate_sampler,
                                                batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers // 2)

    nfeatures = len(config.feature.sigma_s) * len(config.feature.sigma_r) + int(config.feature.self_included)

    # dataset info
    print('Data Preparation Done:')
    print('Trainset {}, Samples {}'.format(len(train_dataset), len(train_sampler)))
    print('Validateset {}, Samples {}'.format(len(validate_dataset), len(validate_sampler)))
    print()


    ######################
    # Create Network Model
    ######################

    MODEL = import_module(args.model)
    net = MODEL.Net(nfeat=nfeatures, nhidden=config.network.feat_dim, dropout=config.network.dropout)
    net.to(device)


    ##################
    # Training Manager
    ##################

    # save config file
    config_filename = os.path.join(outdir, 'config.json')
    with open(config_filename,'w') as file:
        json.dump(config, file, indent=4, sort_keys=True)

    # save argument file
    args_filename = os.path.join(outdir, 'args_training.pth')
    torch.save(args, args_filename)


    print("Start training...")
    manager = Manager(net, config)

    # Start training
    manager.train(net, train_dataloader, validate_dataloader, config, device)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer
#       \***************/
#

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param model: network object
        :param config: config object
        """

        ############
        # Parameters
        ############
        
        # Epoch index
        self.epoch = 0

        # optimizer
        self.optimizer = optim.SGD(net.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay, momentum=config.train.momentum)
        # lr scheduler
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[], gamma=0.1)

        # record file
        self.train_record_file = open(os.path.join(config.dir.result, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(config.dir.result, 'testing.txt'), 'w')

        # eval
        self.metric = 'rmse'
        self.best_metric = float('inf')
        self.best_epoch = -1

        # save training parameters
        self.ckpt_filename = os.path.join(config.dir.result, 'ckpt-best.pth')


    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()


    def train(self, net, train_dataloader, validate_dataloader, config, device):

        # Record File
        print('Testing Record:')
        self.test_record('#epoch mean_degree mean_rmse | #best_epoch best_{}'.format(self.metric))

        # Start training
        train_batch_num = len(train_dataloader)
        bad_counter = 0

        for epoch in range(config.train.max_epochs):

            self.epoch = epoch
            total_loss = 0
            cnt = 0
            all_degrees = []

            learning_rate = self.optimizer.param_groups[0]['lr']

            self.train_record('EPOCH #{}'.format(str(epoch)))

            net.train()

            for batch_id, data in enumerate(train_dataloader):

                # unpack data
                nf, points, gt = data[0], data[1], data[2]

                nf = nf.to(device)
                points = points.to(device)
                gt = gt.to(device)
                points = points.transpose(2, 1)

                # forward
                output = net(points, nf)

                # get loss
                loss = compute_loss(output, gt, loss_type=config.train.normal_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # get normal angular error
                cnt += points.size(0)
                total_loss += loss.item()

                degrees = angle_degrees(output.detach().cpu().numpy(), gt.detach().cpu().numpy())
                all_degrees.append(degrees)


            # update learning rate
            self.scheduler.step()

            # training loss
            train_loss = total_loss/train_batch_num

            # compute average metrics
            all_degrees = np.concatenate(all_degrees)
            train_degree = np.mean(all_degrees)
            train_rmse = np.sqrt(np.mean(np.square(all_degrees)))

            # Start testing
            val_loss, val_degree, val_rmse = self.validate(net, validate_dataloader, config, device)

            # Save best checkpoints
            current_metric = val_degree if self.metric == 'degree' else val_rmse
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = self.epoch
                bad_counter = 0
                self.train_record('Current best epoch #{} saved in file: {}'.format(self.best_epoch, self.ckpt_filename))
                torch.save(net.state_dict(), self.ckpt_filename)
            else:
                bad_counter += 1

            # training record
            self.train_record('train loss = {}, degree = {}, rmse = {}'.format(train_loss, train_degree, train_rmse))
            self.train_record('val loss = {}, degree = {}, rmse = {}'.format(val_loss, val_degree, val_rmse))
            # testing record
            self.test_record('#{:d} {:.4f} {:.4f} | #{:d} {:.4f}'.format(self.epoch, val_degree, val_rmse, self.best_epoch, self.best_metric))

            # Stop training
            if bad_counter >= config.train.patience:
                break


    def validate(self, net, validate_dataloader, config, device):

        # testing
        total_loss = 0
        cnt = 0
        all_degrees = []
        val_batch_num = len(validate_dataloader)

        net.eval()

        for i, data in enumerate(validate_dataloader):

            # unpack data
            nf, points, gt = data[0], data[1], data[2]

            nf = nf.to(device)
            points = points.to(device)
            gt = gt.to(device)
            points = points.transpose(2, 1)

            # forward
            with torch.no_grad():
                output = net(points, nf)

            loss = compute_loss(output, gt, loss_type=config.train.normal_loss)
            total_loss += loss.item()
            cnt += points.size(0)

            degrees = angle_degrees(output.detach().cpu().numpy(), gt.detach().cpu().numpy())
            all_degrees.append(degrees)

        # average loss
        val_loss = total_loss/val_batch_num

        # compute mean errors
        all_degrees = np.concatenate(all_degrees)
        mean_degree = np.mean(all_degrees)
        mean_rmse = np.sqrt(np.mean(np.square(all_degrees)))

        return val_loss, mean_degree, mean_rmse


if __name__ == '__main__':

    config = PCPNetConfig()

    # Update config 
    if args.id_cluster is not None:
        config.dataset.cluster = args.id_cluster
    if args.trainset is not None:
        config.dataset.train_shape_filenames = args.trainset
    if args.validateset is not None:
        config.dataset.validate_shape_filenames = args.validateset
    if args.testset is not None:
        config.dataset.test_shape_filenames = args.testset

    # STARTS HERE
    if args.test:
        test_models(config, args)
    else:
        main(config)


