#include "ceres-error-terms/block-pose-prior-error-term.h"

#include <glog/logging.h>
#include <maplab-common/quaternion-math.h>

namespace ceres_error_terms {

bool BlockPosePriorErrorTerm::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  CHECK_NOTNULL(parameters);
  CHECK_NOTNULL(residuals);

  Eigen::Map<const Eigen::Vector4d> orientation_current(
      parameters
          [kIdxPose]);  // 将parameters[kIdxPose]拿到前四个数值，并传Eigen::Vector4d类型
  Eigen::Map<const Eigen::Vector3d> position_current(
      parameters[kIdxPose] +
      poseblocks::
          kOrientationBlockSize);  // 将parameters[kIdxPose]拿到后三个数值，并传入Eigen::Vector3d类型

  Eigen::Vector4d delta_orientation;  // 旋转误差
  common::positiveQuaternionProductJPL(
      orientation_current, inverse_orientation_prior_,
      delta_orientation);  // 旋转误差求解
  CHECK_GE(delta_orientation(3), 0.);

  // Calculate residuals.
  Eigen::Map<Eigen::Matrix<double, poseblocks::kResidualSize, 1> >
      residual_vector(residuals);  // 将residuals传入Eigen::Matrix<double,
                                   // poseblocks::kResidualSize, 1>类型
  residual_vector.head<3>() =
      2.0 * delta_orientation.head<3>();  // 旋转误差的前三个数值
  residual_vector.tail<3>() = position_current - position_prior_;  // 位置误差

  // 根据信息矩阵的平方根进行加权
  residual_vector = sqrt_information_matrix_ * residual_vector;  // 误差加权

  if (jacobians) {
    // 关于当前姿势的Jacobian w.r.t.系数
    if (jacobians[kIdxPose]) {
      Eigen::Map<PoseJacobian> J(
          jacobians[kIdxPose]);  // 将jacobians[kIdxPose]传入J

      Eigen::Matrix<double, 4, 3, Eigen::RowMajor>
          theta_local_prior;  // 旋转误差关于旋转量的Jacobian

      // 使用JPL四元组参数化是因为我们的四元组的内存布局是JPL
      JplQuaternionParameterization parameterization;
      parameterization.ComputeJacobian(
          orientation_current.data(), theta_local_prior.data());

      J.setZero();
      J.block<3, 4>(0, 0) =
          4.0 * theta_local_prior.transpose();  // 旋转误差关于旋转量的Jacobian
      J.block<3, 3>(3, 4) =
          Eigen::Matrix3d::Identity();  // 旋转误差关于位置量的Jacobian

      // 根据信息矩阵的平方根加权
      J = sqrt_information_matrix_ * J;
    }
  }

  return true;
}

}  // namespace ceres_error_terms
