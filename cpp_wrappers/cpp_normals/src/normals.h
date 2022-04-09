

#include "../../cpp_utils/cloud/cloud.h"
#include "../../cpp_utils/nanoflann/nanoflann.hpp"
#include "../../cpp_utils/Eigen/Dense"

#include <set>
#include <cstdint>

using namespace std;


typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector2d Vector2;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::MatrixX3d MatrixX3;
typedef Eigen::MatrixX3i MatrixX3i;
typedef Eigen::VectorXd VectorX;
typedef Eigen::MatrixXd MatrixX;
typedef Eigen::Vector3i Vector3i;


void normal_filtering_multiscale(vector<PointXYZ>& points, 
                                vector<PointXYZ>& normals, 
                                vector<float>& sigma_s,
                                vector<float>& sigma_r,
                                float radius, 
                                int self_included,
                                vector<float>& ret_normals);

