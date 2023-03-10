#include <ceres-error-terms/pose-prior-error-term.h>
#include <glog/logging.h>
#include <maplab-common/quaternion-math.h>

namespace ceres_error_terms {

bool PosePriorErrorTerm::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  CHECK_NOTNULL(parameters);
  CHECK_NOTNULL(residuals);

  Eigen::Map<const Eigen::Vector4d> orientation_current(
      parameters[kIdxOrientation]);
  Eigen::Map<const Eigen::Vector3d> position_current(parameters[kIdxPosition]);

  Eigen::Vector4d delta_orientation;
  common::positiveQuaternionProductJPL(
      orientation_current, inverse_orientation_prior_, delta_orientation);
  CHECK_GE(delta_orientation(3), 0.0);

  // Calculate residuals.
  Eigen::Map<Eigen::Matrix<double, poseblocks::kResidualSize, 1> >
      residual_vector(residuals);
  residual_vector.head<3>() = 2.0 * delta_orientation.head<3>();  // 旋转误差
  residual_vector.tail<3>() = position_current - position_prior_;  // 平移误差

  // Weight according to the square root of information matrix.
  residual_vector = sqrt_information_matrix_ * residual_vector;  // 误差加权

  if (jacobians) {
    // Jacobian w.r.t. current pose.
    if (jacobians[kIdxOrientation]) {
      Eigen::Map<OrientationJacobian> J(
          jacobians[kIdxOrientation]);  // 旋转雅克比

      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_prior;

      // JPL quaternion parameterization is used because our memory layout
      // of quaternions is JPL.
      JplQuaternionParameterization parameterization;
      parameterization.ComputeJacobian(
          orientation_current.data(), theta_local_prior.data());  // 计算雅克比

      J.setZero();
      J.block<3, 4>(0, 0) = 4.0 * theta_local_prior.transpose();  // 旋转雅克比

      // Add the weighting according to the square root of information matrix.
      J = sqrt_information_matrix_ * J;  // 旋转雅克比加权
    }
    if (jacobians[kIdxPosition]) {
      Eigen::Map<PositionJacobian> J(jacobians[kIdxPosition]);
      J.setZero();
      J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();  // 平移雅克比
      // Add the weighting according to the square root of information matrix.
      J = sqrt_information_matrix_ * J;  // 平移雅克比加权
    }
  }

  return true;
}

}  // namespace ceres_error_terms
