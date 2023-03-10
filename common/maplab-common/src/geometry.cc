#include <glog/logging.h>
#include <maplab-common/geometry.h>
#include <maplab-common/quaternion-math.h>

namespace common {
// Utility method for computing the least-squares average quaternion from a
// vector of quaternia. Derivation is in Appendix A of "Mirror-Based Extrinsic
// Camera Calibration," Hesch et al. WAFR 2008.
Eigen::Matrix<double, 4, 1> ComputeLSAverageQuaternionJPL(
    const VectorOfJPLQuaternia& Gl_q_Gc_vector) {
  CHECK(!Gl_q_Gc_vector.empty());
  if (Gl_q_Gc_vector.size() == 1u) {
    return Gl_q_Gc_vector[0];
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * Gl_q_Gc_vector.size(), 4);
  int i = 0;
  for (const Eigen::Matrix<double, 4, 1>& Gl_q_Gc : Gl_q_Gc_vector) {
    A.block<3, 4>(3 * i, 0) =
        LeftQuaternionJPLMultiplicationMatrix(Gl_q_Gc).block<3, 4>(0, 0);
    ++i;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd_of_A(A, Eigen::ComputeThinV);
  Eigen::Matrix<double, 4, 1> Gl_q_Gc_average =
      quaternionInverseJPL(svd_of_A.matrixV().block<4, 1>(0, 3));
  return Gl_q_Gc_average;
}

double getMaxDisparityRadAngleOfUnitVectorBundle(
    const Aligned<std::vector, Eigen::Vector3d>& unit_incidence_rays) {
  if (unit_incidence_rays.size() < 2u) {
    return 0.0;
  }
  double min_cos_angle = 1.0;
  for (size_t i = 0; i < unit_incidence_rays.size(); ++i) {
    for (size_t j = i + 1; j < unit_incidence_rays.size(); ++j) {
      min_cos_angle = std::min(
          min_cos_angle,
          std::abs(unit_incidence_rays[i].dot(unit_incidence_rays[j])));
    }
  }
  return std::acos(min_cos_angle);
}

namespace geometry {

pose::Transformation yawOnly(const pose::Transformation& original) {
  Eigen::Vector3d rpy =
      RotationMatrixToRollPitchYaw(original.getRotationMatrix());
  rpy.head<2>().setZero();

  pose::Transformation result = original;
  result.getRotation() =
      pose::Quaternion(common::RollPitchYawToRotationMatrix(rpy));

  return result;
}

}  // namespace geometry

}  // namespace common
