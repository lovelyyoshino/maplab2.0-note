#ifndef REGISTRATION_TOOLBOX_MODEL_REGISTRATION_RESULT_H_
#define REGISTRATION_TOOLBOX_MODEL_REGISTRATION_RESULT_H_

#include <Eigen/Dense>
#include <aslam/common/pose-types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace regbox {

class RegistrationResult {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RegistrationResult() = default;
  explicit RegistrationResult(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& reg,
      const aslam::TransformationCovariance& T_target_source_covariance,
      const aslam::Transformation& T_target_source, const bool has_converged);

  void setRegisteredCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& reg);
  pcl::PointCloud<pcl::PointXYZI>::Ptr getRegisteredCloud() const noexcept;

  void set_T_target_source_covariance(
      const aslam::TransformationCovariance& T_target_source_covariance);
  aslam::TransformationCovariance get_T_target_source_covariance() const
      noexcept;

  void set_T_target_source(const Eigen::Matrix4d& T_target_source);
  void set_T_target_source(const aslam::Transformation& T_target_source);
  aslam::Transformation get_T_target_source() const noexcept;

  bool hasConverged() const noexcept;
  void hasConverged(const bool converged);

 private:
  pcl::PointCloud<pcl::PointXYZI>::Ptr registered_;
  aslam::TransformationCovariance T_target_source_covariance_;
  aslam::Transformation T_target_source_;
  int64_t ts_target_ns_;
  int64_t ts_source_ns_;
  bool has_converged_ = false;
};

}  // namespace regbox

#endif  // REGISTRATION_TOOLBOX_MODEL_REGISTRATION_RESULT_H_
