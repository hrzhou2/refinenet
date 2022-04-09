'''
==============================================================

RefineNet Point Cloud Normal Refinement Network
-> Configuration on PCPNet

==============================================================

Author: Haoran Zhou
Date: 2022-4-8

==============================================================
'''


from utils.easydict import EasyDict as edict


def Config():

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

    # prediction (gt features)
    cfg.feature.patch_features = ['normal']
    # input features
    cfg.feature.in_features = ['points']

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

    # height map
    cfg.feature.map_size = 7

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


