clear;clc;close all;
addpath('kdtree');
addpath('IO');
addpath('npy-matlab-master/npy-matlab');

%% configuration
% Estimator model
model_dir = '..\GeoModels\model';

% path
path_pointcloud = '..\data\pcpnet\';
path_normal = '..\results\initial_normals';

path_output = fullfile(model_dir, 'train');
warning off MATLAB:MKDIR:DirectoryExists
mkdir(path_output);

% Default Model parameters
sigma_s = [1.0, 2.0];
sigma_r = [0.1, 0.2, 0.35, 0.5];
rotate_feature = true; self_included = true;
map_size = 7;

% filtering
radius = 0.03;

% cluster
pca_k = 3; cluster_k = 4; cluster_threshold = 0;

% set true to create a new cluster model
is_training = true;

%% Collect data for training

Estimator = GeoNormal(sigma_s, sigma_r, rotate_feature, self_included, ...
                map_size, pca_k, cluster_k, cluster_threshold);


Estimator.features_from_normal(path_normal, path_pointcloud, path_output, radius, is_training);


% save estimator
save(fullfile(model_dir, 'Estimator.mat'), 'Estimator');
writeNPY(Estimator.cluster_model_.pca_mean_, fullfile(model_dir, 'pca_mean_.npy'));
writeNPY(Estimator.cluster_model_.pca_comp_, fullfile(model_dir, 'pca_comp_.npy'));
writeNPY(Estimator.cluster_model_.cluster_center_, fullfile(model_dir, 'cluster_center_.npy'));

