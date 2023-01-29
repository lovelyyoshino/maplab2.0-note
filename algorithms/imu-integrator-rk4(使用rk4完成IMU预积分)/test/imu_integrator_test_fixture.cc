#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <maplab-common/test/testing-entrypoint.h>
#include <maplab-common/test/testing-predicates.h>
#include <memory>

#include "imu-integrator/imu-integrator.h"

using namespace imu_integrator;  // NOLINT

Eigen::Vector4d getOrientationFromState(
    const Eigen::Matrix<double, kStateSize, 1>& state) {  // 从状态中获取四元数
  return state.block<kStateOrientationBlockSize, 1>(kStateOrientationOffset, 0);
}
Eigen::Vector3d getGyroBiasFromState(const Eigen::Matrix<double, kStateSize, 1>&
                                         state) {  // 从状态中获取陀螺仪偏置
  return state.block<kGyroBiasBlockSize, 1>(kStateGyroBiasOffset, 0);
}
Eigen::Vector3d getVelocityFromState(
    const Eigen::Matrix<double, kStateSize, 1>& state) {  // 从状态中获取速度
  return state.block<kVelocityBlockSize, 1>(kStateVelocityOffset, 0);
}
Eigen::Vector3d getAccelBiasFromState(
    const Eigen::Matrix<double, kStateSize, 1>&
        state) {  // 从状态中获取加速度偏置
  return state.block<kAccelBiasBlockSize, 1>(kStateAccelBiasOffset, 0);
}
Eigen::Vector3d getPositionFromState(
    const Eigen::Matrix<double, kStateSize, 1>& state) {  // 从状态中获取位置
  return state.block<kPositionBlockSize, 1>(kStatePositionOffset, 0);
}

class PosegraphErrorTerms : public ::testing::Test {
 protected:
  virtual void SetUp() {
    gravity_magnitude_ = 9.81;
    gyro_noise_sigma_ = 0.0;
    gyro_bias_sigma_ = 0.0;
    acc_noise_sigma_ = 0.0;
    acc_bias_sigma_ = 0.0;
    delta_time_seconds_ = 0.0;
  }

  void constructIntegrator() {  // 构造IMU积分器
    integrator_ = std::shared_ptr<ImuIntegratorRK4>(new ImuIntegratorRK4(
        gyro_noise_sigma_, gyro_bias_sigma_, acc_noise_sigma_, acc_bias_sigma_,
        gravity_magnitude_));
  }

  void integrate() {  // 进行IMU积分
    integrator_->integrate(
        current_state_, debiased_imu_readings_, delta_time_seconds_,
        &next_state_, &next_phi_, &next_cov_);
  }

  std::shared_ptr<ImuIntegratorRK4> integrator_;

  // imu integrator constructor arguments
  double gyro_noise_sigma_;
  double gyro_bias_sigma_;
  double acc_noise_sigma_;
  double acc_bias_sigma_;
  double gravity_magnitude_;

  // imu integrator integrate() arguments
  Eigen::Matrix<double, kStateSize, 1> current_state_;
  Eigen::Matrix<double, 2 * kImuReadingSize, 1> debiased_imu_readings_;
  Eigen::Matrix<double, kStateSize, 1> next_state_;
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> next_phi_;
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> next_cov_;
  double delta_time_seconds_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
