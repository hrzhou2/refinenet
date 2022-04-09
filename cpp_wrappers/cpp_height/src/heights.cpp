
#include "heights.h"
#include <math.h>


// ******************
// Compute Height Map
// ******************

/*
Compute height maps (HMPs) for the entire point cloud, which utilizes the input features and Rots.
Inputs:
    pts: (num_points, 3) xyz coordinates for input point cloud.
    feature: (num_points, 3*nfeatures) features of filtered normals for each point (already oriented)
    Rot: (num_points, 3, 3) rotation matrix for each point
    mapsize: int HMP size (mapsize, mapsize)
    k_knn: int k-nearest neighbors to determine HMP length
Outputs:
    ret: (num_points, nfeatures, mapsize, mapsize) returned HMP matrices
*/
void compute_height_map(vector<PointXYZ>& points, vector<float>& feature, vector<float>& Rot, int nfeatures, int mapsize, int k_knn, vector<float>& ret)
{
    // ***********
    // Point Cloud
    // ***********

    PointCloud cloud;

    cloud.pts = vector<PointXYZ>(points.begin(), points.end());

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree;

    // Pointer to trees
    my_kd_tree* kdtree;

    // Build KDTree for the first batch element
    kdtree = new my_kd_tree(3 /*dims*/, cloud, tree_params);
    kdtree->buildIndex();

    int num_points = points.size();

    // **********************
    // Get Height Map Feature
    // **********************
    
    float hcnt = 0;
    int cnt = 0, total_length = num_points*nfeatures*mapsize*mapsize;
    int ind[3] = { mapsize*mapsize*nfeatures, mapsize*mapsize, mapsize };
    ret.resize(total_length);

    vector< pair<size_t, float> > ret_pairs;
    nanoflann::SearchParams params;
    params.sorted = true;
    for (int i = 0; i < num_points; i++)
    {
        Vector3 point_center(cloud.pts[i].x, cloud.pts[i].y, cloud.pts[i].z);

        // compute boundary length of height map : maxdis
        vector<size_t> idxsInRange(k_knn);
        vector<float> distances(k_knn);
        float query_pt[3] = {cloud.pts[i].x, cloud.pts[i].y, cloud.pts[i].z};
        kdtree->knnSearch(&query_pt[0], k_knn, &idxsInRange[0], &distances[0]);
        float maxdis = 0;

        // sorted?
        for (uint j = 0; j < distances.size(); j++)
        {
            if(distances[j] > maxdis) maxdis = distances[j];
        }
        maxdis = sqrt(maxdis);

        /*MyMatrix rot_T(&Rot[i*9]);
        rot_T = rot_T.transpose();
        MyPoint xr(rot_T.mat[0][0], rot_T.mat[0][1], rot_T.mat[0][2]),
                yr(rot_T.mat[1][0], rot_T.mat[1][1], rot_T.mat[1][2]);*/

        Vector3 xr(Rot[i*9], Rot[i*9+3], Rot[i*9+6]),
                yr(Rot[i*9+1], Rot[i*9+4], Rot[i*9+7]);

        for(int f = 0; f < nfeatures; f++)
        {
            Vector3 nf(feature[i*nfeatures*3+f*3], feature[i*nfeatures*3+f*3+1], feature[i*nfeatures*3+f*3+2]);
            //nf = rot_T * nf;

            float sigma_r = maxdis / (mapsize/2);
            float bin_radius = sigma_r * 0.9;
            // x axis
            Vector3 vx = nf.cross(yr);
            if(vx[0]==0 && vx[1]==0 && vx[2]==0) vx = yr; // if nf==yr
            vx.normalize();
            vx = vx*sigma_r;
            // y axis
            Vector3 vy = nf.cross(xr);
            if(vy[0]==0 && vy[1]==0 && vy[2]==0) vx = xr; // if nf==xr
            vy.normalize();
            vy = vy*sigma_r;

            Vector3 bin0 = point_center - vx*(mapsize/2) - vy*(mapsize/2);
            // for each bin
            for(int r = 0; r < mapsize; r++)
                for(int c = 0; c < mapsize; c++)
                {
                    Vector3 bin_center = bin0 + vx*r + vy*c;
                    float query_pt[3] = { bin_center[0], bin_center[1], bin_center[2] };
                    ret_pairs.clear();
                    kdtree->radiusSearch(&query_pt[0], bin_radius*bin_radius, ret_pairs, params);

                    if(ret_pairs.size() <= 0) continue;
                    // if there is only one point
                    if(ret_pairs.size() == 1 && mapsize%2 == 1 && c == mapsize/2 && r == mapsize/2) continue;

                    // weighted average of signed heights
                    float sum_height = 0, sum_weight = 0;
                    for (uint k = 0; k < ret_pairs.size(); k++)
                    {
                        int pidx = (int)ret_pairs[k].first;
                        Vector3 point_knn(cloud.pts[pidx].x, cloud.pts[pidx].y, cloud.pts[pidx].z), vec_knn = point_knn - bin_center;
                        float height = vec_knn.dot(nf);
                        float dis_bc = ret_pairs[k].second;
                        float weight = exp( -(dis_bc) / (2*sigma_r*sigma_r) );
                        sum_weight += weight;
                        sum_height += height * weight;
                    }
                    if(sum_weight == 0) continue;
                    sum_height /= sum_weight;
                    if(sum_height == 0) continue;

                    // normalize according to patch radius
                    sum_height = sum_height / maxdis;
                    
                    // return array
                    ret[ i*ind[0] + f*ind[1] + r*ind[2] + c ] = sum_height;
                    hcnt += fabs(sum_height);
                    cnt++;
                }
        }
    }
    // normalize
    //float havg = hcnt/cnt;
    //for(int i = 0; i < total_length; i++)
    //    ret[i] /= havg;

    delete kdtree;
}


