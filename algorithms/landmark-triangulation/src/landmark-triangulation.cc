#include "landmark-triangulation/landmark-triangulation.h"

#include <functional>
#include <string>
#include <unordered_map>

#include <aslam/triangulation/triangulation.h>
#include <maplab-common/multi-threaded-progress-bar.h>
#include <maplab-common/parallel-process.h>
#include <vi-map/landmark-quality-metrics.h>
#include <vi-map/vi-map.h>

#include "landmark-triangulation/pose-interpolator.h"

namespace landmark_triangulation {
typedef AlignedUnorderedMap<aslam::FrameId, aslam::Transformation>
    FrameToPoseMap;

namespace {

void interpolateVisualFramePoses(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex_id, const vi_map::VIMap& map,
    FrameToPoseMap* interpolated_frame_poses) {
  CHECK(map.hasMission(mission_id));
  CHECK_NOTNULL(interpolated_frame_poses)->clear();
  // Loop over all missions, vertices and frames and add the interpolated poses
  // to the map.
  size_t total_num_frames = 0u;

  // Check if there is IMU data.
  VertexToTimeStampMap vertex_to_time_map;
  PoseInterpolator imu_timestamp_collector;
  imu_timestamp_collector.getVertexToTimeStampMap(
      map, mission_id, &vertex_to_time_map);
  if (vertex_to_time_map.empty()) {
    VLOG(2) << "Couldn't find any IMU data to interpolate exact landmark "
               "observer positions in "
            << "mission " << mission_id;
    return;
  }

  pose_graph::VertexIdList vertex_ids;
  map.getAllVertexIdsInMissionAlongGraph(
      mission_id, starting_vertex_id, &vertex_ids);

  // Compute upper bound for number of VisualFrames.
  const aslam::NCamera& ncamera = map.getMissionNCamera(mission_id);
  const unsigned int upper_bound_num_frames =
      vertex_ids.size() * ncamera.numCameras();

  // Extract the timestamps for all VisualFrames.
  std::vector<aslam::FrameId> frame_ids(upper_bound_num_frames);
  Eigen::Matrix<int64_t, 1, Eigen::Dynamic> pose_timestamps(
      upper_bound_num_frames);
  unsigned int frame_counter = 0u;
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    const vi_map::Vertex& vertex = map.getVertex(vertex_id);
    for (unsigned int frame_idx = 0u; frame_idx < vertex.numFrames();
         ++frame_idx) {
      if (vertex.isFrameIndexValid(frame_idx)) {
        CHECK_LT(frame_counter, upper_bound_num_frames);
        const aslam::VisualFrame& visual_frame =
            vertex.getVisualFrame(frame_idx);
        const int64_t timestamp = visual_frame.getTimestampNanoseconds();
        // Only interpolate if the VisualFrame timestamp and the vertex
        // timestamp do not match.
        VertexToTimeStampMap::const_iterator it =
            vertex_to_time_map.find(vertex_id);
        if (it != vertex_to_time_map.end() && it->second != timestamp) {
          pose_timestamps(0, frame_counter) = timestamp;
          frame_ids[frame_counter] = visual_frame.getId();
          ++frame_counter;
        }
      }
    }
  }

  // Shrink time stamp and frame id arrays if necessary.
  if (upper_bound_num_frames > frame_counter) {
    frame_ids.resize(frame_counter);
    pose_timestamps.conservativeResize(Eigen::NoChange, frame_counter);
  }

  // Reserve enough space in the frame_to_pose_map to add the poses of the
  // current mission.
  total_num_frames += frame_counter;
  interpolated_frame_poses->reserve(total_num_frames);

  // Interpolate poses for all the VisualFrames.
  if (frame_counter > 0u) {
    VLOG(1) << "Interpolating the exact visual frame poses for "
            << frame_counter << " frames of mission " << mission_id;
    PoseInterpolator pose_interpolator;
    aslam::TransformationVector poses_M_I;
    pose_interpolator.getPosesAtTime(
        map, mission_id, pose_timestamps, &poses_M_I);
    CHECK_EQ(poses_M_I.size(), frame_counter);
    for (size_t frame_num = 0u; frame_num < frame_counter; ++frame_num) {
      interpolated_frame_poses->emplace(
          frame_ids[frame_num], poses_M_I.at(frame_num));
    }
    CHECK_EQ(interpolated_frame_poses->size(), total_num_frames);
  } else {
    VLOG(10) << "No frames found for mission " << mission_id
             << " that need to be interpolated.";
  }

  if (total_num_frames == 0u) {
    VLOG(2) << "No frame pose in any of the missions needs interpolation!";
  }
}

void interpolateVisualFramePoses(
    const vi_map::MissionId& mission_id, const vi_map::VIMap& map,
    FrameToPoseMap* interpolated_frame_poses) {
  CHECK(map.hasMission(mission_id));
  CHECK_NOTNULL(interpolated_frame_poses)->clear();
  const vi_map::VIMission& mission = map.getMission(mission_id);
  const pose_graph::VertexId& starting_vertex_id = mission.getRootVertexId();
  interpolateVisualFramePoses(
      mission_id, starting_vertex_id, map, interpolated_frame_poses);
}

void retriangulateLandmarksOfVertex(
    const FrameToPoseMap& interpolated_frame_poses,
    pose_graph::VertexId storing_vertex_id, vi_map::VIMap* map,
    bool only_included_ids, const vi_map::LandmarkIdSet& landmark_ids) {
  CHECK_NOTNULL(map);
  vi_map::Vertex& storing_vertex = map->getVertex(storing_vertex_id);
  vi_map::LandmarkStore& landmark_store = storing_vertex.getLandmarks();

  const aslam::Transformation& T_M_I_storing = storing_vertex.get_T_M_I();
  const aslam::Transformation& T_G_M_storing =
      const_cast<const vi_map::VIMap*>(map)
          ->getMissionBaseFrameForVertex(storing_vertex_id)
          .get_T_G_M();
  const aslam::Transformation T_G_I_storing = T_G_M_storing * T_M_I_storing;

  for (vi_map::Landmark& landmark : landmark_store) {
    if (only_included_ids) {
      if (landmark_ids.find(landmark.id()) == landmark_ids.end()) {
        continue;
      }
    }

    landmark.setQuality(vi_map::Landmark::Quality::kBad);

    // Triangulation is handled differently for LiDAR and camera landmarks
    const bool is_visual_landmark =
        !vi_map::isLidarFeature(landmark.getFeatureType());

    // The following have one entry per measurement:
    Eigen::Matrix3Xd G_bearing_vectors;
    Eigen::Matrix3Xd p_G_C_vectors;
    Eigen::Matrix3Xd p_G_fi_vectors;

    const vi_map::KeypointIdentifierList& observations =
        landmark.getObservations();

    if (is_visual_landmark) {
      // If we don't have enough observations for a 2D visual landmark we can
      // already abort here. For LiDAR landmarks one observation is enough.
      if (observations.size() < 2u) {
        continue;
      }

      G_bearing_vectors.resize(Eigen::NoChange, observations.size());
      p_G_C_vectors.resize(Eigen::NoChange, observations.size());
    } else {
      p_G_fi_vectors.resize(Eigen::NoChange, observations.size());
    }

    int num_measurements = 0;
    for (const vi_map::KeypointIdentifier& observation : observations) {
      const pose_graph::VertexId& observer_id = observation.frame_id.vertex_id;
      CHECK(map->hasVertex(observer_id))
          << "Observer " << observer_id << " of store landmark "
          << landmark.id() << " not in currently loaded map!";

      const vi_map::Vertex& observer =
          const_cast<const vi_map::VIMap*>(map)->getVertex(observer_id);
      const size_t frame_idx = observation.frame_id.frame_index;
      const aslam::VisualFrame& visual_frame =
          observer.getVisualFrame(frame_idx);
      const aslam::Transformation& T_G_M_observer =
          const_cast<const vi_map::VIMap*>(map)
              ->getMissionBaseFrameForVertex(observer_id)
              .get_T_G_M();

      aslam::Transformation T_M_I_observer;
      if (is_visual_landmark) {
        FrameToPoseMap::const_iterator it =
            interpolated_frame_poses.find(visual_frame.getId());
        if (it != interpolated_frame_poses.end()) {
          // If there are precomputed/interpolated T_M_I, use those.
          T_M_I_observer = it->second;
        } else {
          // TODO(smauq): add linear interpolation here on the fly
          T_M_I_observer = observer.get_T_M_I();
        }
      } else {
        const int64_t offset =
            visual_frame.getKeypointTimeOffset(observation.keypoint_index);

        if (offset != 0) {
          // For LiDAR landmarks we have to compensate for an arbitrary time
          // offset so it can't be precomputed. We use linear interpolation
          // instead of the IMU based one which would take too long
          const bool success =
              interpolateLinear(*map, observer, offset, &T_M_I_observer);

          if (!success) {
            continue;
          }
        }
      }

      const aslam::Transformation& T_I_C =
          observer.getNCameras()->get_T_C_B(frame_idx).inverse();
      aslam::Transformation T_G_C = T_G_M_observer * T_M_I_observer * T_I_C;

      if (is_visual_landmark) {
        Eigen::Vector2d measurement =
            visual_frame.getKeypointMeasurement(observation.keypoint_index);

        Eigen::Vector3d C_bearing_vector;
        bool projection_result = observer.getCamera(frame_idx)->backProject3(
            measurement, &C_bearing_vector);
        if (!projection_result) {
          continue;
        }

        G_bearing_vectors.col(num_measurements) =
            T_G_C.getRotationMatrix() * C_bearing_vector;
        p_G_C_vectors.col(num_measurements) = T_G_C.getPosition();
      } else {
        Eigen::Vector3d p_C_fi =
            visual_frame.getKeypoint3DPosition(observation.keypoint_index);
        p_G_fi_vectors.col(num_measurements) = T_G_C * p_C_fi;
      }

      ++num_measurements;
    }

    Eigen::Vector3d p_G_fi;
    if (is_visual_landmark) {
      if (num_measurements < 2) {
        continue;
      }

      // Resize to final number of valid measurements
      G_bearing_vectors.conservativeResize(Eigen::NoChange, num_measurements);
      p_G_C_vectors.conservativeResize(Eigen::NoChange, num_measurements);

      // Triangulate using collected bearing vectors and camera positions
      aslam::TriangulationResult triangulation_result =
          aslam::linearTriangulateFromNViews(
              G_bearing_vectors, p_G_C_vectors, &p_G_fi);

      if (!triangulation_result.wasTriangulationSuccessful()) {
        continue;
      }
    } else {
      if (num_measurements < 1) {
        continue;
      }

      // Resize to final number of valid measurements
      p_G_fi_vectors.conservativeResize(Eigen::NoChange, num_measurements);

      // Triangulate by averaging the 3D measurements in global frame
      p_G_fi = p_G_fi_vectors.rowwise().mean();
    }

    landmark.set_p_B(T_G_I_storing.inverse() * p_G_fi);
    constexpr bool kReEvaluateQuality = true;
    if (vi_map::isLandmarkWellConstrained(*map, landmark, kReEvaluateQuality)) {
      landmark.setQuality(vi_map::Landmark::Quality::kGood);
    }
  }
}

void retriangulateLandmarksOfMission(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex_id,
    const FrameToPoseMap& interpolated_frame_poses, vi_map::VIMap* map,
    bool only_included_ids = false,
    const vi_map::LandmarkIdSet& landmark_ids = vi_map::LandmarkIdSet()) {
  CHECK_NOTNULL(map);

  VLOG(1) << "Getting vertices of mission: " << mission_id;
  pose_graph::VertexIdList relevant_vertex_ids;
  map->getAllVertexIdsInMissionAlongGraph(
      mission_id, starting_vertex_id, &relevant_vertex_ids);

  const size_t num_vertices = relevant_vertex_ids.size();
  VLOG(1) << "Retriangulating landmarks of " << num_vertices << " vertices.";

  common::MultiThreadedProgressBar progress_bar;
  std::function<void(const std::vector<size_t>&)> retriangulator =
      [&relevant_vertex_ids, map, landmark_ids, only_included_ids,
       &progress_bar,
       &interpolated_frame_poses](const std::vector<size_t>& batch) {
        progress_bar.setNumElements(batch.size());
        size_t num_processed = 0u;
        for (size_t item : batch) {
          CHECK_LT(item, relevant_vertex_ids.size());
          retriangulateLandmarksOfVertex(
              interpolated_frame_poses, relevant_vertex_ids[item], map,
              only_included_ids, landmark_ids);
          progress_bar.update(++num_processed);
        }
      };

  static constexpr bool kAlwaysParallelize = false;
  constexpr size_t kMaxNumHardwareThreads = 8u;
  const size_t available_num_threads = common::getNumHardwareThreads();
  const size_t num_threads = (available_num_threads < kMaxNumHardwareThreads)
                                 ? available_num_threads
                                 : kMaxNumHardwareThreads;
  common::ParallelProcess(
      num_vertices, retriangulator, kAlwaysParallelize, num_threads);
}

void retriangulateLandmarksOfMission(
    const vi_map::MissionId& mission_id,
    const FrameToPoseMap& interpolated_frame_poses, vi_map::VIMap* map,
    bool only_included_ids, const vi_map::LandmarkIdSet& landmark_ids) {
  CHECK_NOTNULL(map);
  const vi_map::VIMission& mission = map->getMission(mission_id);
  const pose_graph::VertexId& starting_vertex_id = mission.getRootVertexId();
  retriangulateLandmarksOfMission(
      mission_id, starting_vertex_id, interpolated_frame_poses, map,
      only_included_ids, landmark_ids);
}
}  // namespace

void retriangulateLandmarks(
    const vi_map::MissionIdList& mission_ids, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);

  for (const vi_map::MissionId& mission_id : mission_ids) {
    retriangulateLandmarksOfMission(mission_id, map);
  }
}

void retriangulateLandmarksOfMission(
    const vi_map::MissionId& mission_id, vi_map::VIMap* map,
    bool only_included_ids, const vi_map::LandmarkIdSet& landmark_ids) {
  FrameToPoseMap interpolated_frame_poses;
  interpolateVisualFramePoses(mission_id, *map, &interpolated_frame_poses);
  retriangulateLandmarksOfMission(
      mission_id, interpolated_frame_poses, map, only_included_ids,
      landmark_ids);
}

void retriangulateLandmarksAlongMissionAfterVertex(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  FrameToPoseMap interpolated_frame_poses;
  interpolateVisualFramePoses(
      mission_id, starting_vertex, *map, &interpolated_frame_poses);
  retriangulateLandmarksOfMission(
      mission_id, starting_vertex, interpolated_frame_poses, map);
}

void retriangulateLandmarks(vi_map::VIMap* map) {
  vi_map::MissionIdList mission_ids;
  map->getAllMissionIds(&mission_ids);
  retriangulateLandmarks(mission_ids, map);
}

void retriangulateLandmarksOfVertex(
    const pose_graph::VertexId& storing_vertex_id, vi_map::VIMap* map,
    bool only_included_ids, const vi_map::LandmarkIdSet& landmark_ids) {
  CHECK_NOTNULL(map);
  FrameToPoseMap empty_frame_to_pose_map;
  retriangulateLandmarksOfVertex(
      empty_frame_to_pose_map, storing_vertex_id, map, only_included_ids,
      landmark_ids);
}

}  // namespace landmark_triangulation
