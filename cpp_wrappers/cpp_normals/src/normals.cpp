
#include "normals.h"
#include <math.h>


// ****************
// Normal Filtering
// ****************


void normal_filtering_multiscale(vector<PointXYZ>& points, vector<PointXYZ>& normals, vector<float>& sigma_s, vector<float>& sigma_r, float radius, int self_included, vector<float>& ret_normals)
{
    // ********************
    // Initialize variables
    // ********************

    // Square radius
    float r2 = radius * radius;


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


    // ************************
    // Compute Filtered Normals
    // ************************

    // Initialize output
    int feat_dim = sigma_s.size() * sigma_r.size() + self_included;
    ret_normals.resize(points.size() * feat_dim * 3);

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

    // compute normals
    int ret_index = 0;
    int num_points = points.size();
    float neigh_thre = 0.5*radius;

    for (int pidx = 0; pidx < num_points; pidx++)
    {
        //reference to the current point
        float query_pt[3] = { cloud.pts[pidx][0], cloud.pts[pidx][1], cloud.pts[pidx][2] };
        auto normal0 = normals[pidx];

        // ball query
        vector<pair<size_t, float>> ret_matches;
        size_t nMatches = kdtree->radiusSearch(&query_pt[0], r2, ret_matches, search_params);

        // include original normals
        if (self_included > 0)
        {
            for(int i = 0; i<3; i++)
                {
                    ret_normals[ret_index] = normal0[i];
                    ret_index++;
                }
        }

        // for each pair of parameters
        for (auto& sigma_s_ : sigma_s)
            for (auto& sigma_r_ : sigma_r)
            {
                // get sigma
                float ss = sigma_s_ * neigh_thre, rr = 2 * sigma_r_ * sigma_r_;
                ss = ss*ss;
                Vector3 n_sum(0,0,0);

                // for each neighbor
                for (auto& match : ret_matches)
                {
                    float dis2 = match.second;

                    // normal distance
                    auto normal1 = normals[match.first];
                    float cos_theta = normal0.x*normal1.x + normal0.y*normal1.y + normal0.z*normal1.z;
                    if (cos_theta < 0)
                        cos_theta = -cos_theta;
                    float n_dis2 = (1-cos_theta)*2;

                    // weighted average
                    float weight = exp(-dis2 / ss) * exp(-n_dis2 / rr);

                    Vector3 n1(normal1.x, normal1.y, normal1.z);
                    n_sum += weight * n1;
                }

                n_sum.normalize();

                // collect normals
                for(int i = 0; i<3; i++)
                {
                    ret_normals[ret_index] = n_sum[i];
                    ret_index++;
                }
            }
        
    }

    delete kdtree;
}


