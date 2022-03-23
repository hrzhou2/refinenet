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

path_output = fullfile(model_dir, 'test');
warning off MATLAB:MKDIR:DirectoryExists
mkdir(path_output);

% filtering
radius = 0.03;

% testing
is_training = false;

%% Collect data for testing
load(fullfile(model_dir, 'Estimator.mat'));

Estimator.features_from_normal(path_normal, path_pointcloud, path_output, radius, is_training);
