#include "ceres-error-terms/block-pose-prior-error-term-v2.h"

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <ceres-error-terms/common.h>
#include <ceres-error-terms/parameterization/quaternion-param-jpl.h>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>
#include <memory>

namespace ceres_error_terms {
// 获取位姿先验误差项的残差
BlockPosePriorErrorTermV2::BlockPosePriorErrorTermV2(
    const aslam::Transformation& T_G_S_measured,
    const Eigen::Matrix<double, 6, 6>& covariance)
    : q_S_G_measured_(
          T_G_S_measured.getRotation().toImplementation().inverse()),
      p_G_S_measured_(
          T_G_S_measured
              .getPosition()) {  // 获取q_S_G_measured_和p_G_S_measured_,分别对应于测量值的旋转和平移
  // 获得协方差矩阵的反平方根
  Eigen::Matrix<double, 6, 6> L = covariance.llt().matrixL();
  sqrt_information_matrix_.setIdentity();  // 设置单位阵
  L.triangularView<Eigen::Lower>().solveInPlace(
      sqrt_information_matrix_);  // 三角分解求解，TriangularView提供了密集矩阵的三角形分块的视图,并允许对其执行优化操作。：https://zhuanlan.zhihu.com/p/536749950
  // Tip:
  // Eigen 的 triangularView
  // 函数可以用来视图一个矩阵的下三角部分或上三角部分。这个函数可以用来给一个矩阵取下三角部分或上三角部分的值，而不需要复制它。这个函数可以用来线性代数运算中的一些优化。

  // 例如：

  // Eigen::MatrixXd A = Eigen::MatrixXd::Random(4,4);
  // Eigen::MatrixXd L = A.triangularViewEigen::Lower();
  // // L is now a view of the lower triangular part of A

  // Eigen::MatrixXd U = A.triangularViewEigen::Upper();
  // // U is now a view of the upper triangular part of A

  // 使用 Eigen::Lower 取出矩阵的下三角部分，使用 Eigen::Upper
  // 取出矩阵的上三角部分。

  // 注意：这个函数返回的是一个视图，对返回的视图的操作会影响到原始矩阵的值。

  // inverse_orientation_prior_ =
  //    Eigen::Quaterniond(orientation_prior_).inverse().coeffs();
}

}  // namespace ceres_error_terms
