
#include "neighbors.h"
#include <math.h>


//****************
// compute normals
//****************

void knn_neighbor(vector<PointXYZ>& queries, vector<int>& neighbors_indices, int pidx, int k)
{
    // Point Cloud
    PointCloud points;

    points.pts = vector<PointXYZ>(queries.begin(), queries.end());

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<double, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree;

    // Pointer to trees
    my_kd_tree* kdtree;

    // Build KDTree for the first batch element
    kdtree = new my_kd_tree(3 /*dims*/, points, tree_params);
    kdtree->buildIndex();

    //reference to the current point
    //printf("query point: %f, %f, %f\n", points.pts[pidx][0], points.pts[pidx][1], points.pts[pidx][2]);
    double query_pt[3] = { points.pts[pidx][0], points.pts[pidx][1], points.pts[pidx][2] };

    vector<size_t> idxsInRange(k);
    vector<double> dis(k);

    size_t nMatches = kdtree->knnSearch(&query_pt[0], k, &idxsInRange[0], &dis[0]);
    //printf("nMatches:%d\n", nMatches);

    // return neighbor indices
    neighbors_indices.resize(idxsInRange.size());
    for (int i = 0; i < idxsInRange.size(); i++)
    {
        neighbors_indices[i] = idxsInRange[i];
    }

    delete kdtree;
}


void compute_pca_normals_knn(vector<PointXYZ>& queries, vector<float>& normals, int k)
{
    // Point Cloud
    PointCloud points;

    points.pts = vector<PointXYZ>(queries.begin(), queries.end());

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree;

    // Pointer to trees
    my_kd_tree* kdtree;

    // Build KDTree for the first batch element
    kdtree = new my_kd_tree(3 /*dims*/, points, tree_params);
    kdtree->buildIndex();

    // Initialize output
    normals.resize(queries.size() * 3);

    // compute pca normals
    int ret_index = 0;
    int num_points = queries.size();
    for (int pidx = 0; pidx < num_points; pidx++)
    {
        //reference to the current point
        //printf("query point: %f, %f, %f\n", points.pts[pidx][0], points.pts[pidx][1], points.pts[pidx][2]);
        float query_pt[3] = { points.pts[pidx][0], points.pts[pidx][1], points.pts[pidx][2] };

        vector<size_t> idxsInRange(k);
        vector<float> dis(k);

        size_t nMatches = kdtree->knnSearch(&query_pt[0], k, &idxsInRange[0], &dis[0]);
        //printf("Neighborhood size:%d, %d\n", nMatches, idxsInRange.size());

        //**************
        // PCA via Eigen
        //**************

        Vector3 mean = Vector3::Zero();
        Matrix3 cov = Matrix3::Zero();

        // mean point
        for(int i=0; i<nMatches; i++){
            const Vector3 v1(points.pts[idxsInRange[i]][0], points.pts[idxsInRange[i]][1], points.pts[idxsInRange[i]][2]);
            mean+=v1;
        }
        mean /= nMatches;

        // covariance matrix
        for(int i=0; i<nMatches; i++){
            const Vector3 v1(points.pts[idxsInRange[i]][0], points.pts[idxsInRange[i]][1], points.pts[idxsInRange[i]][2]);
            Vector3 v = v1-mean;
            cov += v*v.transpose();
        }

        Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullV);
        Matrix3 P =  svd.matrixV().transpose();

        Vector3 normal = P.row(2);
        normal.normalize();

        // collect normals
        //printf("Normal: %f, %f, %f\n", normal[0], normal[1], normal[2]);
        for(int i = 0; i<3; i++)
        {
            normals[ret_index] = normal[i];
            ret_index++;
        }
    }

    delete kdtree;
}



// ****************
// Normal Filtering
// ****************


void normal_filtering_radius(vector<PointXYZ>& points, vector<PointXYZ>& normals, float radius, float sigma_s, float sigma_r, vector<float>& ret_normals)
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
    ret_normals.resize(points.size() * 3);

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

    // compute normals
    int ret_index = 0;
    int num_points = points.size();

    float neigh_thre = 0.5*radius;
    float ss = sigma_s * neigh_thre, rr = 2 * sigma_r * sigma_r;
    ss = ss*ss;

    for (int pidx = 0; pidx < num_points; pidx++)
    {
        //reference to the current point
        float query_pt[3] = { cloud.pts[pidx][0], cloud.pts[pidx][1], cloud.pts[pidx][2] };

        // ball query
        vector<pair<size_t, float>> ret_matches;
        size_t nMatches = kdtree->radiusSearch(&query_pt[0], r2, ret_matches, search_params);

        auto normal0 = normals[pidx];
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

    delete kdtree;
}


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


void normal_reorient(vector<float>& normals, int num_points, int feat_dim, vector<float>& rotation)
{

    rotation.resize(num_points*9);

    // rotate per point normal features
    for (int pidx = 0; pidx < num_points; pidx++)
    {
        // covariance matrix
        Matrix3 cov = Matrix3::Zero();

        // per point normal features
        for (int feat_idx = 0; feat_idx < feat_dim; feat_idx++)
        {
            const Vector3 v(normals[pidx*feat_dim*3 + 3*feat_idx], normals[pidx*feat_dim*3 + 3*feat_idx + 1], normals[pidx*feat_dim*3 + 3*feat_idx + 2]);
            cov += v*v.transpose();
            if (pidx==0)
                printf("%f, %f, %f\n", v(0), v(1), v(2));
        }

        if (pidx==0)
            printf("cov\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n", cov(0,0), cov(0,1), cov(0,2), cov(1,0), cov(1,1), cov(1,2), cov(2,0), cov(2,1), cov(2,2));
        // Eigen vectors
        Eigen::JacobiSVD<Matrix3> svd(cov, Eigen::ComputeFullV);
        Matrix3 V = svd.matrixV().transpose();
        if (pidx==0)
            printf("V\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n", V(0,0), V(0,1), V(0,2), V(1,0), V(1,1), V(1,2), V(2,0), V(2,1), V(2,2));

        // collect data
        //size_t size_in_bytes = 9 * sizeof(float);
        //memcpy(&rotation[pidx*9], V.data(), size_in_bytes);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                rotation[pidx*9 + r*3+c] = V(r,c);

    }
}


