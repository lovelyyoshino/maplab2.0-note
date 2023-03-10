#ifndef VI_MAP_HELPERS_MISSION_CLUSTERING_COOBSERVATION_H_
#define VI_MAP_HELPERS_MISSION_CLUSTERING_COOBSERVATION_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <vi-map/vi-map.h>

namespace vi_map_helpers {

class MissionCoobservationCachedQuery {
 public:
  MissionCoobservationCachedQuery(
      const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

  bool hasCommonObservations(
      const vi_map::MissionId& mission_id,
      const vi_map::MissionIdSet& other_mission_ids) const;

 private:
  typedef std::unordered_map<vi_map::MissionId, vi_map::LandmarkIdSet>
      MissionLandmarksMap;
  MissionLandmarksMap mission_landmark_map_;
  typedef std::unordered_map<vi_map::MissionId, pose_graph::EdgeIdSet>
      MissionEdgeMap;
  MissionEdgeMap mission_lc_edge_;
};

std::vector<vi_map::MissionIdSet> clusterMissionByLandmarkCoobservations(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

size_t getNumAbsolute6DoFConstraintsForMissionCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

bool hasLcEdgesInMissionCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

bool hasInertialConstraintsInAllMissionsInCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

bool hasVisualConstraintsInAllMissionsInCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

bool hasWheelOdometryConstraintsInAllMissionsInCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

bool has6DoFOdometryConstraintsInAllMissionsInCluster(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids);

}  // namespace vi_map_helpers

#endif  // VI_MAP_HELPERS_MISSION_CLUSTERING_COOBSERVATION_H_
