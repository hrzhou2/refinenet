clear;clc;close all;
addpath('kdtree');
addpath('IO');
addpath('cvx');
cvx_setup

%%
% Specify your dataset path here
path_noisy = '..\data\pcpnet\';
path_result = '..\results\initial_normals';
shape_filenames = 'trainingset_no_noise.txt';

% Set up folders for saving normals
warning off MATLAB:MKDIR:DirectoryExists
mkdir(path_result);

% Estimator parameters
Ks = [50, 100];
theta_threshold = 45;
max_sizeA = 2;

%%
% read list filenames
fid = fopen(fullfile(path_noisy, shape_filenames), 'r');
files = {};
while true
    shape_name = fgetl(fid);
    if shape_name == -1
        break;
    end
    files{end+1} = shape_name;
end
fclose(fid);

% Compute shape initial normals
nfiles = length(files);
mean_errors = zeros(1, nfiles);
pnums = zeros(1, nfiles);
for i = 1:nfiles
    filename = files{i};
    V = read_xyz(fullfile(path_noisy, [filename, '.xyz']));
    pts = V';
    pnums(i) = length(pts);
    
    % compute normals
    [normals, ~] = NormalEstimate_fps_ani(pts, Ks, 150, theta_threshold, max_sizeA);
    N = normals';
    
    % Compute error
    Ng = read_xyz(fullfile(path_noisy, [filename, '.normals']));
    N = reorient_normals(N, Ng);
    
    errors = sum((N - Ng).^2);
    errors = acosd(1 - errors/2);
    mean_errors(i) = mean(errors);
    
    write_xyz(fullfile(path_result, [filename, '.xyz']), V, N);
end


%% Record
record_filename = [shape_filenames(1:end-4), '_ErrorInfo.txt'];
fid = fopen(fullfile(path_result, record_filename), 'w');
fprintf(fid, '%s: %d shapes, %d total points, %f (mean degree)\n\n', shape_filenames(1:end-4), nfiles, sum(pnums), ...
    sum(pnums .* mean_errors) / sum(pnums));

fprintf(fid, 'filename  pnums  nfeats  krange  mean_degree\n');
for i = 1:nfiles
    fprintf(fid, '%s  %d  %f\n', files{i}, pnums(i), mean_errors(i));
end
fclose(fid);

