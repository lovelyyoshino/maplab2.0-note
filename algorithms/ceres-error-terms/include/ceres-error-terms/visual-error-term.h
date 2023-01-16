#ifndef CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_
#define CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>
#include <maplab-common/pose_types.h>

#include "ceres-error-terms/common.h"
#include "ceres-error-terms/parameterization/quaternion-param-jpl.h"

namespace ceres_error_terms {

// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().
template <typename CameraType, typename DistortionType>
class VisualReprojectionError
    : public ceres::SizedCostFunction<
          visual::kResidualSize, visual::kPositionBlockSize,
          visual::kPoseBlockSize, visual::kPoseBlockSize,
          visual::kPoseBlockSize, visual::kPoseBlockSize,
          visual::kOrientationBlockSize, visual::kPositionBlockSize,
          CameraType::parameterCount(), DistortionType::parameterCount()> {
 public:
  // Construct a cost function representing the reprojection error. Sigma is
  // standard deviation (in pixels).
  VisualReprojectionError(
      const Eigen::Vector2d& measurement, double pixel_sigma,
      LandmarkErrorType error_term_type, const CameraType* camera)
      : measurement_(measurement),
        error_term_type_(error_term_type),
        camera_ptr_(camera) {
    CHECK_NOTNULL(camera);
    CHECK_GT(pixel_sigma, 0);
    CHECK(isValidLandmarkErrorTermType(error_term_type_));
    pixel_sigma_inverse_ = 1.0 / pixel_sigma;
  }

  virtual ~VisualReprojectionError() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxLandmarkP,
    kIdxLandmarkBasePose,
    kIdxLandmarkMissionBasePose,
    kIdxImuMissionBasePose,
    kIdxImuPose,
    kIdxCameraToImuQ,
    kIdxCameraToImuP,
    kIdxCameraIntrinsics,
    kIdxCameraDistortion
  };

  // The representation for Jacobians computed by this object.
  typedef Eigen::Matrix<double, visual::kResidualSize,
                        visual::kOrientationBlockSize, Eigen::RowMajor>
      OrientationJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize,
                        visual::kPositionBlockSize, Eigen::RowMajor>
      PositionJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize, visual::kPoseBlockSize,
                        Eigen::RowMajor>
      PoseJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize,
                        CameraType::parameterCount(), Eigen::RowMajor>
      IntrinsicsJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize, Eigen::Dynamic,
                        Eigen::RowMajor>
      DistortionJacobian;

  Eigen::Vector2d measurement_;
  double pixel_sigma_inverse_;
  const LandmarkErrorType error_term_type_;
  const CameraType* camera_ptr_;
};

}  // namespace ceres_error_terms

#include "ceres-error-terms/visual-error-term-inl.h"

#endif  // CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_
