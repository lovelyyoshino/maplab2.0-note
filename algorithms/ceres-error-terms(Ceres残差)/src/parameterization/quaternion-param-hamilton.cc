#include <ceres-error-terms/parameterization/quaternion-param-hamilton.h>
#include <glog/logging.h>
#include <maplab-common/pose_types.h>

namespace ceres_error_terms {

bool HamiltonQuaternionParameterization::Plus(
    const double* x, const double* delta,
    double* x_plus_delta)
    const {  // Plus函数是对四元数进行加法运算，ceres 中 Quaternion 是 Hamilton
             // Quaternion，遵循 Hamilton
             // 乘法法则。：https://www.cnblogs.com/JingeTU/p/11707557.html
  CHECK_NOTNULL(x);             // 检查x是否为空
  CHECK_NOTNULL(delta);         // 检查delta是否为空
  CHECK_NOTNULL(x_plus_delta);  // 检查x_plus_delta是否为空

  Eigen::Map<const Eigen::Vector3d> vector_delta(
      delta);  // 将delta转换为Eigen::Vector3d类型
  double square_norm_delta = vector_delta.squaredNorm();  // 计算delta的模的平方
  if (square_norm_delta > 1.0) {
    // 这个delta太大了 -- 不是一个有效的错误四元数
    square_norm_delta = 1.0;
  }
  if (square_norm_delta > 0.0) {
    Eigen::Quaterniond delta_quat(
        sqrt(1.0 - 0.25 * square_norm_delta), 0.5 * vector_delta[0],
        0.5 * vector_delta[1], 0.5 * vector_delta[2]);  // 计算delta的四元数
    delta_quat.normalize();  // 对delta的四元数进行归一化

    const pose::Quaternion quaternion_delta(
        delta_quat);  // 将delta的四元数转换为pose::Quaternion类型

    const Eigen::Map<const Eigen::Quaterniond> quaternion_x(
        x);  // 将x转换为Eigen::Quaterniond类型
    Eigen::Map<Eigen::Quaterniond> quaternion_x_plus_delta(
        x_plus_delta);  // 将x_plus_delta转换为Eigen::Quaterniond类型
    quaternion_x_plus_delta =
        common::positiveQuaternionProductHamilton(
            quaternion_delta, pose::Quaternion(quaternion_x))
            .toImplementation();  // 计算x_plus_delta,使用的是Hamilton乘法，即：https://blog.csdn.net/honyniu/article/details/112135273
    quaternion_x_plus_delta.normalize();  // 对x_plus_delta进行归一化
    CHECK_GE(
        quaternion_x_plus_delta.w(), 0.);  // 检查x_plus_delta的w是否大于等于0
  } else {
    memcpy(
        x_plus_delta, x,
        4 * sizeof(
                *x));  // 如果delta的模的平方小于等于0，直接将x赋值给x_plus_delta
  }
  return true;
}

bool HamiltonQuaternionParameterization::ComputeJacobian(
    const double* x,
    double* jacobian)
    const {  // ComputeJacobian函数是计算Hamilton四元数雅克比矩阵
  CHECK_NOTNULL(x);         // 检查x是否为空
  CHECK_NOTNULL(jacobian);  // 检查jacobian是否为空
  const Eigen::Map<const Eigen::Quaterniond> quat_x(
      x);  // 将x转换为Eigen::Quaterniond类型

  Eigen::Quaterniond quat_x_copy = quat_x;
  if (quat_x_copy.w() < 0.) {                      // 如果x的w小于0
    quat_x_copy.coeffs() = -quat_x_copy.coeffs();  // 将x的四元数的系数取反
  }
  CHECK_GE(quat_x_copy.w(), 0);  // 检查x的w是否大于等于0

  // Hamilton[w,x,y,z]公约乘法和JPL[x,y,z,w]公约内存布局的Jacobian，：https://blog.csdn.net/weixin_42099090/article/details/107355681

  // 80-chars convention violated to keep readability
  //   4x3 Jacobian，从RPY映射到四元数环境空间的雅克比矩阵
  jacobian[0] = quat_x_copy.w() * 0.5;
  jacobian[1] = quat_x_copy.z() * 0.5;
  jacobian[2] = -quat_x_copy.y() * 0.5;  // NOLINT
  jacobian[3] = -quat_x_copy.z() * 0.5;
  jacobian[4] = quat_x_copy.w() * 0.5;
  jacobian[5] = quat_x_copy.x() * 0.5;  // NOLINT
  jacobian[6] = quat_x_copy.y() * 0.5;
  jacobian[7] = -quat_x_copy.x() * 0.5;
  jacobian[8] = quat_x_copy.w() * 0.5;  // NOLINT
  jacobian[9] = -quat_x_copy.x() * 0.5;
  jacobian[10] = -quat_x_copy.y() * 0.5;
  jacobian[11] = -quat_x_copy.z() * 0.5;  // NOLINT

  return true;
}

bool HamiltonYawOnlyQuaternionParameterization::Plus(
    const double* x, const double* delta,
    double* x_plus_delta) const {  // Plus函数是计算Hamilton yaw角的加法
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(delta);
  CHECK_NOTNULL(x_plus_delta);

  double delta_yaw = delta[0];  // delta_yaw是delta的yaw角
  double square_norm_delta = delta_yaw * delta_yaw;  // 计算delta的模的平方
  if (square_norm_delta > 1.0) {
    // This delta is too large -- not a valid error quaternion.
    square_norm_delta = 1.0;
  }
  if (square_norm_delta > 0.0) {
    Eigen::Quaterniond tmp(
        sqrt(1.0 - 0.25 * square_norm_delta), 0, 0,
        0.5 * delta_yaw);  // 计算delta的四元数
    tmp.normalize();       // 对delta的四元数进行归一化

    const pose::Quaternion quaternion_delta(tmp);

    const Eigen::Map<const Eigen::Quaterniond> quaternion_x(x);
    Eigen::Map<Eigen::Quaterniond> quaternion_x_plus_delta(x_plus_delta);
    quaternion_x_plus_delta =
        common::signedQuaternionProductHamilton(
            quaternion_delta, pose::Quaternion(quaternion_x))
            .toImplementation();  // 计算x_plus_delta,使用的是Hamilton乘法，即：https://blog.csdn.net/honyniu/article/details/112135273
    quaternion_x_plus_delta.normalize();

    if (quaternion_x_plus_delta.w() < 0.) {
      quaternion_x_plus_delta.coeffs() = -quaternion_x_plus_delta.coeffs();
    }
    CHECK_GE(quaternion_x_plus_delta.w(), 0.);
  } else {
    memcpy(x_plus_delta, x, 4 * sizeof(*x));
  }
  return true;
}

bool HamiltonYawOnlyQuaternionParameterization::ComputeJacobian(
    const double* x, double* jacobian)
    const {  // ComputeJacobian函数是计算Hamilton yaw角的雅克比矩阵
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(jacobian);
  const Eigen::Map<const Eigen::Quaterniond> quat_x(x);

  Eigen::Quaterniond quat_x_copy = quat_x;
  if (quat_x_copy.w() < 0.) {
    quat_x_copy.coeffs() = -quat_x_copy.coeffs();
  }
  CHECK_GE(quat_x_copy.w(), 0);

  // Jacobian for Hamilton [w,x,y,z] convention multiplication
  // and JPL [x,y,z,w] convention memory layout
  //
  // 4x1 Jacobian，从四元数环境空间映射到只有偏航的切线空间
  jacobian[0] = -quat_x_copy.y() * 0.5;
  jacobian[1] = quat_x_copy.x() * 0.5;
  jacobian[2] = quat_x_copy.w() * 0.5;
  jacobian[3] = -quat_x_copy.z() * 0.5;

  return true;
}

}  // namespace ceres_error_terms
