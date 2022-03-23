from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from data_utils import get_normal_features, get_cluster_indices, batch_normals_pca


# ----------------------------------------------------------------------------------------------------------------------
#
#        Dataset Sampler
#       \***********************/
#


class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):
    '''
    Point Cloud Random Sampler with Limited Patches per Shape
    '''

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


# ----------------------------------------------------------------------------------------------------------------------
#
#        Dataset Class
#       \***********************/
#


class MultiFeatureDataset(data.Dataset):
    '''
    Point Cloud Dataset for Multiple Features.
    '''
    
    def __init__(self, config, shape_list_filename, seed=None, identical_epochs=False, cache_capacity=100):

        # initialize parameters
        self.root = config.dataset.pointcloud_dir
        self.normal_dir = config.dataset.normal_dir
        self.cluster_dir = config.dataset.cluster_dir
        self.cluster = config.dataset.cluster

        # get features
        self.patch_features = config.feature.patch_features
        self.use_pca = config.feature.use_pca
        self.shape_radius = config.feature.query_radius

        # patch points parameters
        self.query_type = config.feature.query
        self.center = config.feature.center
        if self.query_type == 'knn':
            self.query_k = config.feature.query_k
        elif self.query_type == 'ball':
            self.query_radius = config.feature.query_radius
            self.points_per_patch = config.feature.points_per_patch
        else:
            raise ValueError('Unknown query type: {:s}'.format(self.query_type))

        # parameters
        self.identical_epochs = identical_epochs
        self.seed = seed

        # Predict normals or curvatures
        self.include_normals = False
        self.include_curvatures = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))


        ######################
        # Process Point Clouds
        ######################

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, MultiFeatureDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(self.root, shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        #self.shape_names = [x.split()[0] for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        offset = 0
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load point cloud data
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            # Process normal features
            initial_filename = os.path.join(self.normal_dir, shape_name+'.xyz')
            nf_filename = os.path.join(self.normal_dir, shape_name+'.nf.npy')
            if not os.path.exists(nf_filename):
                in_normals = np.loadtxt(initial_filename).astype('float32')
                in_normals = in_normals[:,3:6]
                normal_features = get_normal_features(pts, 
                                                    in_normals, 
                                                    radius=config.feature.filter_radius, 
                                                    sigma_s=config.feature.sigma_s, 
                                                    sigma_r=config.feature.sigma_r,
                                                    self_included=config.feature.self_included)
                np.save(nf_filename, normal_features)

            # compute cluster indices
            cidx_filename = os.path.join(self.normal_dir, shape_name+'.cidx.npy')
            if not os.path.exists(cidx_filename):
                normal_features = torch.from_numpy(np.load(nf_filename))

                # reorient before cluster
                normal_features, shape_trans = batch_normals_pca(normal_features)
                normal_features = normal_features.numpy()
                np.save(os.path.join(self.normal_dir, shape_name+'.rot.npy'), shape_trans)

                # divide into clusters
                shape_cidx = get_cluster_indices(normal_features, cluster_dir=self.cluster_dir)
                shape_cidx += 1

                np.save(cidx_filename, shape_cidx)

            # include gt normals
            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                curvatures = np.loadtxt(curv_filename).astype('float32')
                np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            shape = self.shape_cache.get(shape_ind)

            # point count in each shape
            if shape.cidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.cidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.shape_radius)

            # total point offset
            offset += pts.shape[0]


    def __getitem__(self, index):

        # retrieve shape and the center point (index)
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.cidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.cidx[patch_ind]

        ##################
        # Get Patch Points
        ##################

        # kdtree search
        if self.query_type == 'knn':
            patch_point_dis, patch_point_inds = shape.kdtree.query(shape.pts[center_point_ind, :], k=self.query_k)
            patch_radius = np.amax(patch_point_dis)
        elif self.query_type == 'ball':
            patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], self.query_radius))
            patch_radius = self.query_radius
            point_count = min(self.points_per_patch, len(patch_point_inds))
            # if there are too many neighbors, pick a random subset
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]
        else:
            raise ValueError('Unknown query type: {:s}'.format(self.query_type))

        # get patch points
        patch_pts = torch.from_numpy(shape.pts[patch_point_inds, :])
        
        # center patch (central point at origin - but avoid changing padded zeros)
        if self.center == 'mean':
            patch_pts = patch_pts - patch_pts.mean(0)
        elif self.center == 'point':
            patch_pts = patch_pts - torch.from_numpy(shape.pts[center_point_ind, :])
        elif self.center == 'none':
            pass # no centering
        else:
            raise ValueError('Unknown patch centering option: %s' % (center))

        # normalize by radius
        patch_pts = patch_pts / patch_radius


        ####################
        # Get Other Features
        ####################

        # get normal features
        patch_nf = torch.from_numpy(shape.nf[center_point_ind, :])

        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        if self.include_curvatures:
            patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            patch_curv = patch_curv * self.patch_radius_absolute[shape_ind]


        # apply global rotation to pts and other features (according to normal features)
        if self.use_pca:
            trans = torch.from_numpy(shape.rot[center_point_ind, :, :])
            # rotate normals and pts
            patch_nf = torch.matmul(patch_nf.view(-1,3), trans).view(-1).contiguous()
            patch_pts = torch.matmul(patch_pts, trans)
            if self.include_normals:
                patch_normal = torch.matmul(patch_normal, trans)
        else:
            trans = torch.eye(3).float()

        # if self.use_pca:

        #     patch_nf = patch_nf.unsqueeze(0)

        #     # rotation matrix sorted by eigenvalues
        #     patch_nf, trans = batch_normals_pca(patch_nf)
        #     patch_nf, trans = patch_nf.squeeze(0), trans.squeeze(0)
        #     '''
        #     trans, s, _ = torch.svd(patch_nf.t())
        #     s_idx = torch.sort(s)[1]
        #     trans = trans[:, s_idx]

        #     # to positive
        #     if torch.sum(patch_nf[0, :] * trans[:, -1]) < 0:
        #         trans = -trans

        #     # rotate normal features
        #     patch_nf = torch.matmul(patch_nf, trans).view(-1)'''

        #     # rotate other features
        #     patch_pts = torch.matmul(patch_pts, trans)
        #     if self.include_normals:
        #         patch_normal = torch.matmul(patch_normal, trans)
        # else:
        #     trans = torch.eye(3).float()


        ######################
        # Collect All Features
        ######################

        patch_feats = ()

        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal,)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        return (patch_nf,) + (patch_pts,) + patch_feats + (trans,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    # do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
    # so modifying the points would make the kdtree give incorrect results
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        nf_filename = os.path.join(self.normal_dir, self.shape_names[shape_ind]+'.nf')
        cidx_filename = os.path.join(self.normal_dir, self.shape_names[shape_ind]+'.cidx')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        rot_filename = os.path.join(self.normal_dir, self.shape_names[shape_ind]+'.rot') if self.use_pca else None
        
        # load all data (npy)
        pts = np.load(point_filename+'.npy')
        nf = np.load(nf_filename+'.npy')

        if rot_filename != None:
            rot = np.load(rot_filename+'.npy')
        else:
            rot = None

        if normals_filename != None:
            normals = np.load(normals_filename+'.npy')
        else:
            normals = None

        if curv_filename != None:
            curvatures = np.load(curv_filename+'.npy')
        else:
            curvatures = None

        if cidx_filename != None:
            cluster_indices = np.load(cidx_filename+'.npy')
            cluster_indices = (cluster_indices == self.cluster)
            cluster_indices = np.arange(cluster_indices.shape[0])[cluster_indices].astype('int')
        else:
            cluster_indices = None


        sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
        kdtree = spatial.cKDTree(pts, 10)

        return Shape(pts=pts, nf=nf, kdtree=kdtree, normals=normals, curv=curvatures, cidx=cluster_indices, rot=rot)


# ----------------------------------------------------------------------------------------------------------------------
#
#        Point Cloud Shape Container
#       \***********************/
#

class Shape():
    def __init__(self, pts, nf, kdtree, normals=None, curv=None, cidx=None, rot=None):
        self.pts = pts
        self.nf = nf
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.cidx = cidx
        self.rot = rot


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


