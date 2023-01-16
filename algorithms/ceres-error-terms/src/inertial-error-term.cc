#include "ceres-error-terms/inertial-error-term.h"

#include <ceres-error-terms/parameterization/quaternion-param-jpl.h>
#include <imu-integrator/imu-integrator.h>
#include <maplab-common/quaternion-math.h>

namespace ceres_error_terms {

void InertialErrorTerm::IntegrateStateAndCovariance(
    const InertialState& current_state,
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data,
    InertialState* next_state, InertialStateCovariance* phi_accum,
    InertialStateCovariance* Q_accum)
    const {  // 这个函数是计算IMU的积分的状态和协方差，来完成优化,这部分看的难度较大，可以先看pose-prior-error-term.cc
  CHECK_NOTNULL(next_state);  // 检查next_state是否为空
  CHECK_NOTNULL(phi_accum);
  CHECK_NOTNULL(Q_accum);

  Eigen::Matrix<double, 2 * imu_integrator::kImuReadingSize, 1>
      debiased_imu_readings;              // 去除重力加速度的IMU数据
  InertialStateCovariance phi;            // 状态转移矩阵
  InertialStateCovariance new_phi_accum;  // 新的状态转移矩阵
  InertialStateCovariance Q;              // 过程噪声协方差
  InertialStateCovariance new_Q_accum;    // 新的过程噪声协方差

  Q_accum->setZero();        // 初始化Q_accum
  phi_accum->setIdentity();  // 初始化phi_accum

  typedef Eigen::Matrix<double, imu_integrator::kStateSize, 1>
      InertialStateVector;  // IMU状态向量
  InertialStateVector current_state_vec, next_state_vec;
  current_state_vec = current_state.toVector();  // 将IMU状态转换为向量

  for (int i = 0; i < imu_data.cols() - 1; ++i) {  // 遍历IMU数据
    CHECK_GE(imu_timestamps(0, i + 1), imu_timestamps(0, i))
        << "IMU measurements not properly ordered";  // 检查IMU数据是否按照时间顺序排列

    const Eigen::Block<
        InertialStateVector, imu_integrator::kGyroBiasBlockSize, 1>
        current_gyro_bias =
            current_state_vec.segment<imu_integrator::kGyroBiasBlockSize>(
                imu_integrator::kStateGyroBiasOffset);  // 当前陀螺仪偏置
    const Eigen::Block<
        InertialStateVector, imu_integrator::kAccelBiasBlockSize, 1>
        current_accel_bias =
            current_state_vec.segment<imu_integrator::kAccelBiasBlockSize>(
                imu_integrator::kStateAccelBiasOffset);  // 当前加速度计偏置

    debiased_imu_readings << imu_data.col(i).segment<3>(
                                 imu_integrator::kAccelReadingOffset) -
                                 current_accel_bias,
        imu_data.col(i).segment<3>(imu_integrator::kGyroReadingOffset) -
            current_gyro_bias,
        imu_data.col(i + 1).segment<3>(imu_integrator::kAccelReadingOffset) -
            current_accel_bias,
        imu_data.col(i + 1).segment<3>(imu_integrator::kGyroReadingOffset) -
            current_gyro_bias;  // 去除重力加速度的IMU数据

    const double delta_time_seconds =
        (imu_timestamps(0, i + 1) - imu_timestamps(0, i)) *
        imu_integrator::kNanoSecondsToSeconds;  // IMU数据的时间间隔
    integrator_.integrate(
        current_state_vec, debiased_imu_readings, delta_time_seconds,
        &next_state_vec, &phi, &Q);  // 计算IMU的积分

    current_state_vec = next_state_vec;                    // 更新当前状态
    new_Q_accum = phi * (*Q_accum) * phi.transpose() + Q;  // 更新过程噪声协方差

    Q_accum->swap(new_Q_accum);
    new_phi_accum = phi * (*phi_accum);  // 更新状态转移矩阵
    phi_accum->swap(new_phi_accum);
  }

  *next_state = InertialState::fromVector(next_state_vec);  // 更新下一个状态
}

bool InertialErrorTerm::Evaluate(
    double const* const* parameters, double* residuals_ptr,
    double** jacobians) const {  // 这个函数是计算残差和雅克比矩阵
  enum {
    kIdxPoseFrom,
    kIdxGyroBiasFrom,
    kIdxVelocityFrom,
    kIdxAccBiasFrom,
    kIdxPoseTo,
    kIdxGyroBiasTo,
    kIdxVelocityTo,
    kIdxAccBiasTo
  };

  // 对于Ceres来说，保持Jacobians的行主位，Eigen默认为列主位
  typedef Eigen::Matrix<
      double, imu_integrator::kErrorStateSize,
      imu_integrator::kGyroBiasBlockSize, Eigen::RowMajor>
      GyroBiasJacobian;  // 陀螺仪偏置雅克比矩阵
  typedef Eigen::Matrix<
      double, imu_integrator::kErrorStateSize,
      imu_integrator::kVelocityBlockSize, Eigen::RowMajor>
      VelocityJacobian;  // 速度雅克比矩阵
  typedef Eigen::Matrix<
      double, imu_integrator::kErrorStateSize,
      imu_integrator::kAccelBiasBlockSize, Eigen::RowMajor>
      AccelBiasJacobian;  // 加速度计偏置雅克比矩阵
  typedef Eigen::Matrix<
      double, imu_integrator::kErrorStateSize,
      imu_integrator::kStatePoseBlockSize, Eigen::RowMajor>
      PoseJacobian;  // 位姿雅克比矩阵

  const double* q_from_ptr = parameters[kIdxPoseFrom];  // 从哪个位置读取信息
  const double* bw_from_ptr = parameters[kIdxGyroBiasFrom];
  const double* v_from_ptr = parameters[kIdxVelocityFrom];
  const double* ba_from_ptr = parameters[kIdxAccBiasFrom];
  const double* p_from_ptr =
      parameters[kIdxPoseFrom] + imu_integrator::kStateOrientationBlockSize;

  const double* q_to_ptr = parameters[kIdxPoseTo];
  const double* bw_to_ptr = parameters[kIdxGyroBiasTo];
  const double* v_to_ptr = parameters[kIdxVelocityTo];
  const double* ba_to_ptr = parameters[kIdxAccBiasTo];
  const double* p_to_ptr =
      parameters[kIdxPoseTo] + imu_integrator::kStateOrientationBlockSize;

  Eigen::Map<const Eigen::Vector4d> q_I_M_from(q_from_ptr);  // 将指针完成映射
  Eigen::Map<const Eigen::Vector3d> b_g_from(bw_from_ptr);
  Eigen::Map<const Eigen::Vector3d> v_M_from(v_from_ptr);
  Eigen::Map<const Eigen::Vector3d> b_a_from(ba_from_ptr);
  Eigen::Map<const Eigen::Vector3d> p_M_I_from(p_from_ptr);

  Eigen::Map<const Eigen::Vector4d> q_I_M_to(q_to_ptr);
  Eigen::Map<const Eigen::Vector3d> b_g_to(bw_to_ptr);
  Eigen::Map<const Eigen::Vector3d> v_M_I_to(v_to_ptr);
  Eigen::Map<const Eigen::Vector3d> b_a_to(ba_to_ptr);
  Eigen::Map<const Eigen::Vector3d> p_M_I_to(p_to_ptr);

  Eigen::Map<Eigen::Matrix<double, imu_integrator::kErrorStateSize, 1> >
      residuals(residuals_ptr);  // 残差

  // 整合IMU的测量结果
  InertialState begin_state;
  begin_state.q_I_M = q_I_M_from;  // 将起始状态的四元数赋值
  begin_state.b_g = b_g_from;
  begin_state.v_M = v_M_from;
  begin_state.b_a = b_a_from;
  begin_state.p_M_I = p_M_I_from;

  // 如果线性化点没有改变，则重复使用之前的积分
  const bool cache_is_valid = integration_cache_.valid &&
                              (integration_cache_.begin_state == begin_state);
  if (!cache_is_valid) {
    integration_cache_.begin_state = begin_state;
    IntegrateStateAndCovariance(
        integration_cache_.begin_state, imu_timestamps_, imu_data_,
        &integration_cache_.end_state, &integration_cache_.phi_accum,
        &integration_cache_.Q_accum);  // 计算积分

    integration_cache_.L_cholesky_Q_accum.compute(
        integration_cache_.Q_accum);  // 计算Q的cholesky分解
    integration_cache_.valid = true;  // 标记为有效
  }
  CHECK(integration_cache_.valid);  // 检查是否有效

  if (residuals_ptr) {                 // 如果有残差
    Eigen::Quaterniond quaternion_to;  // 四元数转化
    quaternion_to.coeffs() = q_I_M_to;

    Eigen::Quaterniond quaternion_integrated;
    quaternion_integrated.coeffs() =
        integration_cache_.end_state.q_I_M;  // 计算出的end_state四元数

    Eigen::Vector4d delta_q;
    common::positiveQuaternionProductJPL(
        q_I_M_to, quaternion_integrated.inverse().coeffs(),
        delta_q);  // 计算四元数的差值
    CHECK_GE(delta_q(3), 0.);

    residuals <<
        // 虽然我们的四元数表示法是Hamilton，但底层内存
        // 的布局是JPL的，因为Eigen.
        2. * delta_q.head<3>(),
        b_g_to - integration_cache_.end_state.b_g,
        v_M_I_to - integration_cache_.end_state.v_M,
        b_a_to - integration_cache_.end_state.b_a,
        p_M_I_to - integration_cache_.end_state.p_M_I;

    integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(residuals);
  } else {
    LOG(WARNING)
        << "Skipped residual calculation, since residual pointer was NULL";
  }

  if (jacobians != NULL) {  // 如果有雅克比矩阵
    if (!cache_is_valid) {  // 如果线性化点没有改变
      InertialJacobianType& J_end =
          integration_cache_.J_end;  // 结束的雅克比矩阵
      InertialJacobianType& J_begin =
          integration_cache_.J_begin;  // 开始的雅克比矩阵

      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_begin;
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> theta_local_end;
      // 这是将错误状态提升到状态的雅克比。JPL
      // 四元数参数化被使用，因为我们的四元数的内存布局是 JPL。
      JplQuaternionParameterization parameterization;
      parameterization.ComputeJacobian(q_I_M_to.data(), theta_local_end.data());
      parameterization.ComputeJacobian(
          q_I_M_from.data(), theta_local_begin.data());

      // 计算边的末端的Jacobian系数
      J_end.setZero();
      J_end.block<3, 4>(0, 0) = 4.0 * theta_local_end.transpose();
      J_end.block<12, 12>(3, 4) = Eigen::Matrix<double, 12, 12>::Identity();

      // 由于Ceres将实际的Jacobian和Jacobian分离开来的,所以我们应用局部参数化的逆向。然后Ceres可以在此基础上应用局部参数化的Jacobian，最后我们得到正确的Jacobian。这是必要的，因为我们
      // 传播状态为错误状态。
      J_begin.setZero();
      J_begin.block<3, 4>(0, 0) =
          -4.0 * integration_cache_.phi_accum.block<3, 3>(0, 0) *
          theta_local_begin.transpose();
      J_begin.block<3, 12>(0, 4) =
          -integration_cache_.phi_accum.block<3, 12>(0, 3);
      J_begin.block<12, 4>(3, 0) =
          -4.0 * integration_cache_.phi_accum.block<12, 3>(3, 0) *
          theta_local_begin.transpose();
      J_begin.block<12, 12>(3, 4) =
          -integration_cache_.phi_accum.block<12, 12>(3, 3);

      // Invert and apply by using backsolve.
      integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(J_end);
      integration_cache_.L_cholesky_Q_accum.matrixL().solveInPlace(J_begin);
    }

    const InertialJacobianType& J_end = integration_cache_.J_end;
    const InertialJacobianType& J_begin = integration_cache_.J_begin;

    if (jacobians[kIdxPoseFrom] != NULL) {
      Eigen::Map<PoseJacobian> J(jacobians[kIdxPoseFrom]);
      J.leftCols<imu_integrator::kStateOrientationBlockSize>() =
          J_begin.middleCols<imu_integrator::kStateOrientationBlockSize>(
              imu_integrator::
                  kStateOrientationOffset);  // 当存在Pose信息的时候就将Jacobian矩阵的J_begin赋值到J的leftCols中，详细需要看论文推导
      J.rightCols<imu_integrator::kPositionBlockSize>() =
          J_begin.middleCols<imu_integrator::kPositionBlockSize>(
              imu_integrator::kStatePositionOffset);
    }
    if (jacobians[kIdxGyroBiasFrom] != NULL) {
      Eigen::Map<GyroBiasJacobian> J(jacobians[kIdxGyroBiasFrom]);
      J = J_begin.middleCols<imu_integrator::kGyroBiasBlockSize>(
          imu_integrator::kStateGyroBiasOffset);
    }
    if (jacobians[kIdxVelocityFrom] != NULL) {
      Eigen::Map<VelocityJacobian> J(jacobians[kIdxVelocityFrom]);
      J = J_begin.middleCols<imu_integrator::kVelocityBlockSize>(
          imu_integrator::kStateVelocityOffset);
    }
    if (jacobians[kIdxAccBiasFrom] != NULL) {
      Eigen::Map<AccelBiasJacobian> J(jacobians[kIdxAccBiasFrom]);
      J = J_begin.middleCols<imu_integrator::kAccelBiasBlockSize>(
          imu_integrator::kStateAccelBiasOffset);
    }

    if (jacobians[kIdxPoseTo] != NULL) {
      Eigen::Map<PoseJacobian> J(jacobians[kIdxPoseTo]);
      J.leftCols<imu_integrator::kStateOrientationBlockSize>() =
          J_end.middleCols<imu_integrator::kStateOrientationBlockSize>(
              imu_integrator::kStateOrientationOffset);
      J.rightCols<imu_integrator::kPositionBlockSize>() =
          J_end.middleCols<imu_integrator::kPositionBlockSize>(
              imu_integrator::kStatePositionOffset);
    }
    if (jacobians[kIdxGyroBiasTo] != NULL) {
      Eigen::Map<GyroBiasJacobian> J(jacobians[kIdxGyroBiasTo]);
      J = J_end.middleCols<imu_integrator::kGyroBiasBlockSize>(
          imu_integrator::kStateGyroBiasOffset);
    }
    if (jacobians[kIdxVelocityTo] != NULL) {
      Eigen::Map<VelocityJacobian> J(jacobians[kIdxVelocityTo]);
      J = J_end.middleCols<imu_integrator::kVelocityBlockSize>(
          imu_integrator::kStateVelocityOffset);
    }
    if (jacobians[kIdxAccBiasTo] != NULL) {
      Eigen::Map<AccelBiasJacobian> J(jacobians[kIdxAccBiasTo]);
      J = J_end.middleCols<imu_integrator::kAccelBiasBlockSize>(
          imu_integrator::kStateAccelBiasOffset);
    }
  }
  return true;
}

} /* namespace ceres_error_terms */
