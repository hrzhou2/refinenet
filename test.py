'''
==============================================================

RefineNet Point Cloud Normal Refinement Network
-> Testing Models

==============================================================

Author: Haoran Zhou
Date: 2022-3-15

==============================================================
'''


import numpy as np
import torch
import sys
import os
import json
from importlib import import_module
from datasets import MultiFeatureDataset as Dataset
from datasets import collate_fn
from utils.easydict import EasyDict as edict
from utils.loss import angle_degrees


# ----------------------------------------------------------------------------------------------------------------------
#
#         Test Models
#       \***************/
#

def test_models(config, args):

    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    # Get pretrained model directory
    if args.pretrained is None:
        raise ValueError('Please specify a path for testing.')
    config.dir.result = os.path.join(config.dir.result, args.pretrained)
    config.dir.test = os.path.join(config.dir.test, args.pretrained)
    model_dir = config.dir.result
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Testing output path: {:s}'.format(config.dir.test))

    # Save predicted results
    outdir = os.path.join(config.dir.test, 'prediction')
    if args.output and not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get cluster models
    list_models = os.listdir(config.dir.result)
    list_models = [model_name for model_name in list_models if model_name.startswith('model_cidx')]
    if len(list_models) == 0:
        raise ValueError('Find {:d} models. Please check the model path.'.format(len(list_models)))


    ###############
    # Start Testing
    ###############

    # predicted results
    model_preds = {}
    model_samples = {}
    model_degrees = {}
    model_rmse = {}
    model_clusters = {}

    # predict results for cluster models
    for model_name in list_models:

        # load model config
        config_filename = os.path.join(model_dir, model_name, 'config.json')
        with open(config_filename, 'r') as file:
            model_config = json.load(file)
        model_config = edict(model_config)

        # predicted normals
        model_preds[model_name] = []
        model_clusters[model_name] = model_config.dataset.cluster

        # Load testing dataset
        test_dataset = Dataset(model_config, shape_list_filename=config.dataset.test_shape_filenames)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=model_config.train.batch_size,
                                                    num_workers=model_config.train.num_workers,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn)
        nfeatures = len(model_config.feature.sigma_s) * len(model_config.feature.sigma_r) + int(model_config.feature.self_included)

        # Load network model
        MODEL = import_module(args.model)
        net = MODEL.Net(model_config, nfeat=nfeatures)
        net.to(device)
        model_filename = os.path.join(model_dir, model_name, 'ckpt-best.pth')
        net.load_state_dict(torch.load(model_filename, map_location='cpu'))


        #################
        # Predict Normals
        #################

        net.eval()

        test_batch_num = len(test_dataloader)
        all_degrees = []
        cnt = 0
        total_shape = len(test_dataset.shape_names)
        last_shape = -1

        # predict all cluster samples
        for batch_idx, data in enumerate(test_dataloader):

            # unpack data
            for k, v in data.items():
                if k != 'trans':
                    data[k] = v.to(device)

            # forward
            with torch.no_grad():
                output = net(data)

            # get normal angular error
            gt = data['normal']
            degrees = angle_degrees(output.detach().cpu().numpy(), gt.detach().cpu().numpy())
            all_degrees.append(degrees)

            # get real normals, transform back to real-world coordinates
            trans = data['trans'].to(device)
            trans_normals = torch.matmul(output.unsqueeze(1), trans.transpose(1, 2)).squeeze()
            model_preds[model_name].append(trans_normals.detach().cpu().numpy())

            # track process
            shape_idx, _ = test_dataset.shape_index(cnt)
            if shape_idx > last_shape:
                last_shape = shape_idx
                print('{:s}: shape {:d}/{:d}'.format(model_name, shape_idx, total_shape-1))

            cnt += output.size(0)


        # compute mean errors
        all_degrees = np.concatenate(all_degrees)
        mean_degree = np.mean(all_degrees)
        mean_rmse = np.sqrt(np.mean(np.square(all_degrees)))
        # model errors
        model_samples[model_name] = cnt
        model_degrees[model_name] = mean_degree
        model_rmse[model_name] = mean_rmse
        print(model_name, cnt, mean_degree, mean_rmse)

        # collect predicted results
        model_preds[model_name] = np.concatenate(model_preds[model_name], axis=0)


    ########################
    # Save Predicted Results
    ########################

    shape_samples = []
    shape_degrees = []
    shape_rmse = []
    subset_degrees = []
    subset_rmse = []
    offsets = [0] * len(list_models)

    # collect results for all shapes
    for shape_name in test_dataset.shape_names:
        # get shape points total count and cluster index
        cidx_filename = os.path.join(config.dataset.normal_dir, shape_name+'.cidx.npy')
        shape_cidx = np.load(cidx_filename)
        shape_points_count = shape_cidx.shape[0]

        # get gt normals
        normals_filename = os.path.join(config.dataset.pointcloud_dir, shape_name+'.normals.npy')
        gt_normals = np.load(normals_filename)

        # collect model results for this shape
        shape_preds = np.zeros((shape_points_count, 3), dtype=np.float32)
        for model_idx, model_name in enumerate(list_models):
            # shape cluster index
            cidx = (shape_cidx == model_clusters[model_name])
            num_cidx = np.sum(cidx.astype('int'))

            # collect results
            preds = model_preds[model_name]
            shape_preds[cidx, :] = preds[offsets[model_idx] : offsets[model_idx] + num_cidx, :]
            offsets[model_idx] += num_cidx

        # shape errors
        degrees = angle_degrees(shape_preds, gt_normals)
        shape_samples.append(degrees.shape[0])
        shape_degrees.append(np.mean(degrees))
        shape_rmse.append(np.sqrt(np.mean(np.square(degrees))))

        if args.sparse_patches:
            pidx = np.loadtxt(os.path.join(config.dataset.pointcloud_dir, shape_name+'.pidx'), dtype=np.int32)
            degrees = degrees[pidx]
            subset_degrees.append(np.mean(degrees))
            subset_rmse.append(np.sqrt(np.mean(np.square(degrees))))

        # save results
        if args.output:
            preds_filename = os.path.join(outdir, shape_name+'.normals')
            np.savetxt(preds_filename, shape_preds)

    # check output
    for model_idx, model_name in enumerate(list_models):
        if offsets[model_idx] != model_preds[model_name].shape[0]:
            raise ValueError('Unmatched size for {:s}: offset {} != preds.shape[0] {}'.format(model_name, offsets[model_idx], model_preds[model_name].shape[0]))

    # Record
    with open(os.path.join(config.dir.test, 'testing.txt'), 'w') as file:
        # record model scores
        file.write('model\tnsamples\tdegree\trmse\n')
        for model_idx, model_name in enumerate(list_models):
            file.write('{:s}\t{:d}\t{:.4f}\t{:.4f}\n'.format(model_name, model_samples[model_name], model_degrees[model_name], model_rmse[model_name]))
        file.write('-----------------------------\n\n')

        # record shape scores
        file.write('shape\tnsamples\tdegree\trmse\n')
        for shape_idx, shape_name in enumerate(test_dataset.shape_names):
            file.write('{:s}\t{:d}\t{:.4f}\t{:.4f}\n'.format(shape_name, shape_samples[shape_idx], shape_degrees[shape_idx], shape_rmse[shape_idx]))
        # overall scores
        overall_degree = sum([d*n for d, n in zip(shape_degrees, shape_samples)]) / sum(shape_samples)
        overall_rmse = sum([d*n for d, n in zip(shape_rmse, shape_samples)]) / sum(shape_samples)
        file.write('MEAN\t{:.4f}\t{:.4f}\n\n'.format(overall_degree, overall_rmse))

        # record sparse patches
        if args.sparse_patches:
            message = 'Sparse Patch Indices'
            file.write('-'*len(message) + '\n')
            file.write(message + '\n')
            file.write('-'*len(message) + '\n')

            # record shape scores
            file.write('shape\tdegree\trmse\n')
            for shape_idx, shape_name in enumerate(test_dataset.shape_names):
                file.write('{:s}\t{:.4f}\t{:.4f}\n'.format(shape_name, subset_degrees[shape_idx], subset_rmse[shape_idx]))
            # average shape scores
            file.write('MEAN\t{:.4f}\t{:.4f}\n\n'.format(sum(subset_degrees) / len(subset_degrees), sum(subset_rmse) / len(subset_rmse)))



