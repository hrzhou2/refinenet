'''
==============================================================

RefineNet Point Cloud Normal Refinement Network
-> Data Processing

==============================================================

Author: Haoran Zhou
Date: 2022-3-13

==============================================================
'''


import os
import numpy as np
import torch
from cpp_wrappers.cpp_normals import cpp_normals
from cpp_wrappers.cpp_height import cpp_height


# ----------------------------------------------------------------------------------------------------------------------
#
#        Functions for data processing
#       \***********************/
#


def get_normal_features(points, normals, radius, sigma_s, sigma_r, self_included):
    '''
    Compute filtered normals from multi-scale parameters.
    '''

    points = points.copy()
    normals = normals.copy()

    # search radius
    bbox = np.max(points, axis=0) - np.min(points, axis=0)
    bbox_size = np.linalg.norm(bbox)
    radius = radius*bbox_size

    # parameters
    sigma_s = np.array(sigma_s, dtype=np.float32)
    sigma_r = np.array(sigma_r, dtype=np.float32)
    self_included = int(self_included)

    # Normal filtering
    new_normals = cpp_normals.normal_filtering(points, normals, sigma_s, sigma_r, radius, self_included)

    return new_normals


def get_height_features(points, features, rotation, map_size, query_k):
    '''
    Compute height map features.
    '''

    points = points.copy()
    features = features.copy()
    rotation = rotation.copy()

    # Get features
    heights = cpp_height.height_distances(points, features, rotation, map_size, query_k)

    return heights


def get_cluster_indices(normals, cluster_dir):
    '''
    Divide samples into clusters by normal features.

    Args:
        normals: (N, dim)
    Returns:
        cluster_idx: (N,)
    '''

    normals_t = normals.transpose()
    dim, N = normals_t.shape

    # load existing cluster info
    pca_comp = np.load(os.path.join(cluster_dir, 'pca_comp_.npy')).copy(order='C').astype(np.float32) # (k, dim)
    pca_mean = np.load(os.path.join(cluster_dir, 'pca_mean_.npy')).copy(order='C').astype(np.float32) # (dim, 1)
    cluster_center = np.load(os.path.join(cluster_dir, 'cluster_center_.npy')).copy(order='C').astype(np.float32) # (cl, k)

    # pca scores
    pca_values = np.matmul(pca_comp, normals_t - pca_mean) # (k, N)

    # cluster scores
    cluster_scores = 2*np.matmul(cluster_center, pca_values) - np.sum(cluster_center**2, axis=1, keepdims=True) # (cl, N)
    cluster_idx = np.argmax(cluster_scores, axis=0) # (N,)

    return cluster_idx


def batch_normals_pca(normals, all_positive=True, descending=False):
    '''
    Apply pca on normals for consistent global directions.
    Args:
        normals: (B, dim), point-wise normal features
    Returns:
        normals: (B, dim)
        trans: (B, 3, 3)
    '''

    B, dim = normals.shape
    normals = normals.view(B, -1, 3) # (B, f, 3)

    # rotation matrix sorted by eigenvalues
    trans, s, _ = torch.linalg.svd(normals.permute(0, 2, 1))
    if not descending:
        #s_idx = torch.sort(s, dim=-1)[1]
        #trans = trans.permute(0, 2, 1)
        #id0 = torch.arange(0, B).view(-1,1)
        #trans = trans[id0, s_idx, :].permute(0, 2, 1).contiguous()
        # default is descending order 
        trans = trans[:, :, [2,1,0]]

    # to positive
    if all_positive:
        is_positive = torch.sum(normals[:, 0, :] * trans[:, :, -1], dim=-1) > 0
        is_positive = is_positive.float()*2 - 1
        trans = trans * is_positive.view(-1, 1, 1)

    # rotate normal features
    normals = torch.matmul(normals, trans).view(B, -1)

    return normals, trans


def reorient_normals(normals, gt_normals):
    '''
    Normal reorientation according to gt. (unoriented estimate)
    '''

    flags = np.sum(normals * gt_normals, axis=1) > 0
    flags = flags.astype(np.float32)*2 - 1
    normals = normals * np.expand_dims(flags, axis=1)

    return normals

