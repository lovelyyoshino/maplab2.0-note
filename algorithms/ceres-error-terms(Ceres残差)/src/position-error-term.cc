#include <ceres-error-terms/position-error-term.h>
#include <glog/logging.h>
#include <maplab-common/quaternion-math.h>

namespace ceres_error_terms {

bool PositionErrorTerm::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const {
  CHECK_NOTNULL(parameters);
  CHECK_NOTNULL(residuals);

  Eigen::Map<const Eigen::Vector3d> position_current(
      parameters[kIdxPose] +
      poseblocks::
          kOrientationBlockSize);  // 将位置参数转换为Eigen::Vector3d类型

  // Calculate residuals.
  Eigen::Map<Eigen::Matrix<double, positionblocks::kResidualSize, 1> >
      residual_vector(residuals);
  residual_vector = position_current - position_prior_;

  // 根据信息矩阵的平方根进行加权
  residual_vector = sqrt_information_matrix_ * residual_vector;

  if (jacobians) {
    // Jacobian w.r.t. current pose.
    if (jacobians[kIdxPose]) {
      Eigen::Map<PositionJacobian> J(
          jacobians[kIdxPose]);  // 将雅克比转换为Eigen::Matrix<double, 3, 6,
                                 // Eigen::RowMajor>类型

      J.setZero();
      J.block<3, 3>(0, 4) = Eigen::Matrix3d::Identity();  // 平移雅克比

      // Add the weighting according to the square root of information matrix.
      J = sqrt_information_matrix_ * J;
    }
  }
  return true;
}

}  // namespace ceres_error_terms
