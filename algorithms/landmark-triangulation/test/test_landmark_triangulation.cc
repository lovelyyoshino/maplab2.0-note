#include <Eigen/Core>

#include <map-manager/map-manager.h>
#include <maplab-common/test/testing-entrypoint.h>
#include <maplab-common/test/testing-predicates.h>
#include <vi-map/vi-map.h>
#include <vi-mapping-test-app/vi-mapping-test-app.h>

#include "landmark-triangulation/landmark-triangulation.h"

DECLARE_double(elq_min_observation_angle_deg);
DECLARE_uint64(elq_min_observers);
DECLARE_double(elq_max_distance_from_closest_observer);
DECLARE_double(elq_min_distance_from_closest_observer);

namespace landmark_triangulation {

class ViMappingTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    test_app_.loadDataset("./test_maps/common_test_map");
    CHECK_NOTNULL(test_app_.getMapMutable());
  }

  virtual void corruptLandmarks();
  visual_inertial_mapping::VIMappingTestApp test_app_;
};

void ViMappingTest::corruptLandmarks() {
  constexpr double kLandmarkPositionStdDev = 5.0;
  constexpr int kEveryNthToCorrupt = 1;
  test_app_.corruptLandmarkPositions(
      kLandmarkPositionStdDev, kEveryNthToCorrupt);
}

TEST_F(ViMappingTest, TestLandmarkTriangulation) {
  corruptLandmarks();

  retriangulateLandmarks(test_app_.getMapMutable());
  constexpr double kPrecision = 0.1;
  // Original landmarks are triangulated using the optimizer,
  // so here we expect lower precision at a higher performance
  constexpr double kMinPassingLandmarkFraction = 0.8;
  test_app_.testIfLandmarksMatchReference(
      kPrecision, kMinPassingLandmarkFraction);
}

}  // namespace landmark_triangulation

MAPLAB_UNITTEST_ENTRYPOINT
