#ifndef IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_
#define IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_

#include <cmath>
#include <glog/logging.h>
#include <iomanip>
#include <limits>
#include <maplab-common/geometry.h>
#include <maplab-common/quaternion-math.h>

#include "imu-integrator/common.h"

namespace imu_integrator {

template <typename ScalarType>
void ImuIntegratorRK4::integrate(
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,
    const Eigen::Matrix<ScalarType, 2 * kImuReadingSize, 1>&
        debiased_imu_readings,
    const ScalarType delta_time_seconds,
    Eigen::Matrix<ScalarType, kStateSize, 1>* next_state,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* next_phi,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* next_cov)
    const {  // imu积分器的核心函数，使用rk4算法进行积分，输入为current_state为当前状态，debiased_imu_readings为imu数据，delta_time_seconds为时间间隔，next_state为输出的下一状态，next_phi为输出的下一状态的雅克比矩阵，next_cov为输出的下一状态的协方差矩阵
  // The (next_phi, next_cov) pair is optional; both pointers have to be null to
  // skip calculations.
  LOG_IF(FATAL, static_cast<bool>(next_phi) != static_cast<bool>(next_cov))
      << "next_phi and next_cov have to be either both valid or be null";
  bool calculate_phi_cov =
      (next_phi != nullptr) &&
      (next_cov != nullptr);  // 判断是否计算雅克比矩阵和协方差矩阵

  ScalarType o5 = static_cast<ScalarType>(0.5);  // 0.5转为模板类型

  next_state->setZero();
  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k1 =
      debiased_imu_readings.template block<kImuReadingSize, 1>(
          0, 0);  // 获取imu数据的第一行

  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k23;
  interpolateImuReadings(
      debiased_imu_readings, delta_time_seconds, o5 * delta_time_seconds,
      &imu_readings_k23);  // 对imu数据进行插值，得到imu数据的第二行和第三行

  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k4 =
      debiased_imu_readings.template block<kImuReadingSize, 1>(
          kImuReadingSize, 0);  // 获取imu数据的第四行

  Eigen::Matrix<ScalarType, kStateSize, 1> state_der1;  // 保存状态的导数
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der2;
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der3;
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der4;
  getStateDerivativeRungeKutta(
      imu_readings_k1, current_state,
      &state_der1);  // 计算状态的导数,使用的是rk4算法

  getStateDerivativeRungeKutta(
      imu_readings_k23,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + o5 * delta_time_seconds * state_der1),
      &state_der2);  // 计算状态的导数,使用的是rk4算法
  getStateDerivativeRungeKutta(
      imu_readings_k23,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + o5 * delta_time_seconds * state_der2),
      &state_der3);  // 计算状态的导数,使用的是rk4算法
  getStateDerivativeRungeKutta(
      imu_readings_k4,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + delta_time_seconds * state_der3),
      &state_der4);  // 计算状态的导数,使用的是rk4算法

  // Calculate final state using RK4.
  *next_state = current_state + delta_time_seconds *
                                    (state_der1 + ScalarType(2) * state_der2 +
                                     ScalarType(2) * state_der3 + state_der4) /
                                    ScalarType(6);  // 使用rk4算法计算下一状态

  if (calculate_phi_cov) {  // 如果需要计算雅克比矩阵和协方差矩阵
    next_phi->setZero();    // 设置下一个时刻的phi为0
    next_cov->setZero();    // 设置下一个时刻的协方差为0

    const ScalarType* state_q_ptr = next_state->head(4).data();  // 获取四元数
    Eigen::Quaternion<ScalarType> B_q_G(state_q_ptr);  // 四元数转换为四元数对象
    B_q_G.normalize();                                 // 归一化四元数

    // Now calculate state transition matrix and covariance.
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        cov_der1;  // 协方差的导数
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der2;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der3;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der4;

    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        transition_der1;  // 状态转移矩阵的导数
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der2;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der3;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der4;

    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> current_cov =
        Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>::
            Zero();  // 当前协方差
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_transition =
            Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>::
                Identity();  // 当前状态转移矩阵

    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k1, current_state, current_cov, current_transition,
        &cov_der1, &transition_der1);  // 计算协方差和状态转移矩阵的导数

    Eigen::Matrix<ScalarType, kStateSize, 1>
        current_state_intermediate;  // 当前状态的中间值
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_cov_intermediate;  // 当前协方差的中间值
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_transition_intermediate;  // 当前状态转移矩阵的中间值

    ScalarType o5 = static_cast<ScalarType>(0.5);
    current_state_intermediate =
        current_state +
        o5 * delta_time_seconds * state_der1;  // 计算当前状态的中间值
    current_cov_intermediate =
        current_cov +
        o5 * delta_time_seconds * cov_der1;  // 计算当前协方差的中间值
    current_transition_intermediate =
        current_transition +
        o5 * delta_time_seconds *
            transition_der1;  // 计算当前状态转移矩阵的中间值
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k23, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der2,
        &transition_der2);  // 计算协方差和状态转移矩阵的导数

    current_state_intermediate =
        current_state +
        o5 * delta_time_seconds * state_der2;  // 计算当前状态的中间值
    current_cov_intermediate =
        current_cov +
        o5 * delta_time_seconds * cov_der2;  // 计算当前协方差的中间值
    current_transition_intermediate =
        current_transition +
        o5 * delta_time_seconds *
            transition_der2;  // 计算当前状态转移矩阵的中间值
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k23, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der3,
        &transition_der3);  // 计算协方差和状态转移矩阵的导数

    current_state_intermediate =
        current_state + delta_time_seconds * state_der3;
    current_cov_intermediate = current_cov + delta_time_seconds * cov_der3;
    current_transition_intermediate =
        current_transition + delta_time_seconds * transition_der3;
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k4, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der4,
        &transition_der4);  // 计算协方差和状态转移矩阵的导数

    *next_cov =
        current_cov + delta_time_seconds *
                          (cov_der1 + static_cast<ScalarType>(2) * cov_der2 +
                           static_cast<ScalarType>(2) * cov_der3 + cov_der4) /
                          static_cast<ScalarType>(6);  // 计算下一时刻的协方差
    *next_phi = current_transition;

    next_phi->template block<3, 15>(0, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(0, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(0, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(0, 0) +
         transition_der4.template block<3, 15>(0, 0)) /
        ScalarType(6.0);  // 计算下一时刻的状态转移矩阵
    next_phi->template block<3, 15>(6, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(6, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(6, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(6, 0) +
         transition_der4.template block<3, 15>(6, 0)) /
        ScalarType(6.0);  // 计算下一时刻的状态转移矩阵
    next_phi->template block<3, 15>(12, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(12, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(12, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(12, 0) +
         transition_der4.template block<3, 15>(12, 0)) /
        ScalarType(6.0);  // 计算下一时刻的状态转移矩阵
  }
}

template <typename ScalarType>
void ImuIntegratorRK4::getStateDerivativeRungeKutta(
    const Eigen::Matrix<ScalarType, kImuReadingSize, 1>& debiased_imu_readings,
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,
    Eigen::Matrix<ScalarType, kStateSize, 1>* state_derivative)
    const {  // 计算状态导数,输入为debiased_imu_readings表示去除重力后的加速度和角速度,输入为current_state表示当前状态,输出为state_derivative表示状态导数
  CHECK_NOTNULL(state_derivative);

  Eigen::Quaternion<ScalarType> B_q_G(
      current_state.head(4).data());  // 从当前状态中提取四元数
  // As B_q_G is calculated using linearization, it may not be normalized
  // -> we need to do it explicitly before passing to quaternion object.
  ScalarType o5 = static_cast<ScalarType>(0.5);
  B_q_G.normalize();
  Eigen::Matrix<ScalarType, 3, 3> B_R_G;
  common::toRotationMatrixJPL(B_q_G.coeffs(), &B_R_G);  // 四元数转旋转矩阵
  const Eigen::Matrix<ScalarType, 3, 3> G_R_B =
      B_R_G.transpose();  // 旋转矩阵转置

  const Eigen::Matrix<ScalarType, 3, 1> acc_meas(
      debiased_imu_readings.template block<3, 1>(kAccelReadingOffset, 0)
          .data());  // 从debiased_imu_readings中提取加速度
  const Eigen::Matrix<ScalarType, 3, 1> gyr_meas(
      debiased_imu_readings.template block<3, 1>(kGyroReadingOffset, 0)
          .data());  // 从debiased_imu_readings中提取角速度

  Eigen::Matrix<ScalarType, 4, 4> gyro_omega;
  gyroOmegaJPL(gyr_meas, &gyro_omega);  // 计算角速度矩阵

  Eigen::Matrix<ScalarType, 4, 1> q_dot =
      o5 * gyro_omega * B_q_G.coeffs();  // 计算四元数导数
  Eigen::Matrix<ScalarType, 3, 1> v_dot =
      G_R_B * acc_meas -
      Eigen::Matrix<ScalarType, 3, 1>(
          ScalarType(0), ScalarType(0),
          ScalarType(gravity_acceleration_));  // 计算速度导数
  Eigen::Matrix<ScalarType, 3, 1> p_dot = current_state.template block<3, 1>(
      kStateVelocityOffset, 0);  // 计算位置导数

  state_derivative->setZero();  // Bias derivatives are zero.
  state_derivative->template block<4, 1>(kStateOrientationOffset, 0) =
      q_dot;  // 将四元数导数赋值给状态导数
  state_derivative->template block<3, 1>(kStateVelocityOffset, 0) = v_dot;
  state_derivative->template block<3, 1>(kStatePositionOffset, 0) = p_dot;
}

template <typename ScalarType>
void ImuIntegratorRK4::getCovarianceTransitionDerivativesRungeKutta(
    const Eigen::Matrix<ScalarType, kImuReadingSize, 1>& debiased_imu_readings,
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,
    const Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>&
        current_cov,
    const Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>&
        current_transition,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* cov_derivative,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>*
        transition_derivative)
    const {  // 计算协方差和状态转移矩阵导数,输入为debiased_imu_readings表示去除重力后的加速度和角速度,输入为current_state表示当前状态,输入为current_cov表示当前协方差,输入为current_transition表示当前状态转移矩阵,输出为cov_derivative表示协方差导数,输出为transition_derivative表示状态转移矩阵导数
  CHECK_NOTNULL(cov_derivative);
  CHECK_NOTNULL(transition_derivative);

  Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> phi_cont =
      Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>::
          Zero();  // 初始化phi_cont为0

  Eigen::Quaternion<ScalarType> B_q_G(
      current_state.head(4).data());  // 从current_state中提取四元数
  // As B_q_G is calculated using linearization, it may not be normalized
  // -> we need to do it explicitly before passing to quaternion object.
  B_q_G.normalize();
  Eigen::Matrix<ScalarType, 3, 3> B_R_G;
  common::toRotationMatrixJPL(
      B_q_G.coeffs(), &B_R_G);  // 将四元数转换为旋转矩阵
  const Eigen::Matrix<ScalarType, 3, 3> G_R_B =
      B_R_G.transpose();  // 计算旋转矩阵的转置

  Eigen::Matrix<ScalarType, 3, 1> acc_meas(
      debiased_imu_readings.template block<3, 1>(kAccelReadingOffset, 0)
          .data());  // 从debiased_imu_readings中提取加速度
  const Eigen::Matrix<ScalarType, 3, 1> gyr_meas(
      debiased_imu_readings.template block<3, 1>(kGyroReadingOffset, 0)
          .data());  // 从debiased_imu_readings中提取角速度

  const Eigen::Matrix<ScalarType, 3, 3> gyro_skew;
  common::skew(gyr_meas, gyro_skew);  // 计算角速度的反对称矩阵

  const Eigen::Matrix<ScalarType, 3, 3> acc_skew;
  common::skew(acc_meas, acc_skew);  // 计算加速度的反对称矩阵

  phi_cont.template block<3, 3>(0, 3) =
      -Eigen::Matrix<ScalarType, 3, 3>::Identity();
  phi_cont.template block<3, 3>(12, 6) =
      Eigen::Matrix<ScalarType, 3, 3>::Identity();
  phi_cont.template block<3, 3>(0, 0) =
      -gyro_skew;  // 将角速度的反对称矩阵赋值给phi_cont的前三行前三列
  phi_cont.template block<3, 3>(6, 9) =
      -G_R_B;  // 将旋转矩阵的转置赋值给phi_cont的第七行到第九行
  phi_cont.template block<3, 3>(6, 0) =
      -G_R_B *
      acc_skew;  // 将旋转矩阵的转置乘以加速度的反对称矩阵赋值给phi_cont的第七行到第九行

  // Compute *transition_derivative = phi_cont * current_transition blockwise.
  transition_derivative->setZero();
  transition_derivative->template block<3, 15>(0, 0) =
      phi_cont.template block<3, 3>(0, 0) *
          current_transition.template block<3, 15>(0, 0) -
      current_transition.template block<3, 15>(
          3,
          0);  // 将角速度的反对称矩阵乘以当前状态的前三行前十五列赋值给transition_derivative的前三行前十五列
  transition_derivative->template block<3, 15>(6, 0) =
      phi_cont.template block<3, 3>(6, 0) *
          current_transition.template block<3, 15>(0, 0) +
      phi_cont.template block<3, 3>(6, 9) *
          current_transition.template block<3, 15>(
              9,
              0);  // 将旋转矩阵的转置乘以当前状态的第七行到第九行前十五列赋值给transition_derivative的第七行到第九行前十五列
  transition_derivative
      ->template block<3, 15>(12, 0) = current_transition.template block<3, 15>(
      6,
      0);  // 将当前状态的第十三行到第十五行前十五列赋值给transition_derivative的第十三行到第十五行前十五列

  Eigen::Matrix<ScalarType, 15, 15> phi_cont_cov =
      Eigen::Matrix<ScalarType, 15, 15>::Zero();
  phi_cont_cov.template block<3, 15>(0, 0) =
      phi_cont.template block<3, 3>(0, 0) *
          current_cov.template block<3, 15>(0, 0) -
      current_cov.template block<3, 15>(
          3,
          0);  // 将角速度的反对称矩阵乘以当前协方差的前三行前十五列赋值给phi_cont_cov的前三行前十五列
  phi_cont_cov.template block<3, 15>(6, 0) =
      phi_cont.template block<3, 3>(6, 0) *
          current_cov.template block<3, 15>(0, 0) +
      phi_cont.template block<3, 3>(6, 9) *
          current_cov.template block<3, 15>(
              9,
              0);  // 将旋转矩阵的转置乘以当前协方差的第七行到第九行前十五列赋值给phi_cont_cov的第七行到第九行前十五列
  phi_cont_cov.template block<3, 15>(12, 0) =
      current_cov.template block<3, 15>(6, 0);
  *cov_derivative =
      phi_cont_cov +
      phi_cont_cov
          .transpose();  // 将phi_cont_cov的转置加上phi_cont_cov赋值给cov_derivative

  // Relevant parts of Gc * Qc * Gc'.
  cov_derivative->diagonal().template segment<3>(0) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(gyro_noise_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(3) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(gyro_bias_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(6) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(acc_noise_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(9) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(acc_bias_sigma_squared_));
}

template <typename ScalarType>
void ImuIntegratorRK4::interpolateImuReadings(
    const Eigen::Matrix<ScalarType, 2 * kImuReadingSize, 1>& imu_readings,
    const ScalarType delta_time_seconds,
    const ScalarType increment_step_size_seconds,
    Eigen::Matrix<ScalarType, kImuReadingSize, 1>* interpolated_imu_readings)
    const {  // 设置IMU差值数据读取，imu_readings为IMU数据，delta_time_seconds为时间间隔，increment_step_size_seconds为步长，interpolated_imu_readings为差值数据
  CHECK_NOTNULL(interpolated_imu_readings);
  CHECK_GE(delta_time_seconds, 0.0);

  if (delta_time_seconds < std::numeric_limits<ScalarType>::epsilon()) {
    *interpolated_imu_readings =
        imu_readings.template block<kImuReadingSize, 1>(0, 0);
    return;
  }

  *interpolated_imu_readings =
      imu_readings.template block<kImuReadingSize, 1>(0, 0) +
      (imu_readings.template block<kImuReadingSize, 1>(kImuReadingSize, 0) -
       imu_readings.template block<kImuReadingSize, 1>(0, 0)) *
          (increment_step_size_seconds / delta_time_seconds);
}

template <typename ScalarType>
void ImuIntegratorRK4::gyroOmegaJPL(
    const Eigen::Matrix<ScalarType, 3, 1>& gyro_readings,
    Eigen::Matrix<ScalarType, 4, 4>* omega_matrix)
    const {  // 设置角速度矩阵，gyro_readings为角速度，omega_matrix为角速度矩阵
  CHECK_NOTNULL(omega_matrix);

  const ScalarType scalar_type_zero = static_cast<ScalarType>(0.);

  *omega_matrix << scalar_type_zero, gyro_readings[2], -gyro_readings[1],
      gyro_readings[0], -gyro_readings[2], scalar_type_zero, gyro_readings[0],
      gyro_readings[1], gyro_readings[1], -gyro_readings[0], scalar_type_zero,
      gyro_readings[2], -gyro_readings[0], -gyro_readings[1], -gyro_readings[2],
      scalar_type_zero;
}

}  // namespace imu_integrator

#endif  // IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_
