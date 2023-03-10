#include "vi-map-helpers/vi-map-manipulation.h"

#include <Eigen/Dense>
#include <aslam/common/memory.h>
#include <maplab-common/accessors.h>
#include <maplab-common/conversions.h>
#include <maplab-common/pose_types.h>
#include <maplab-common/progress-bar.h>
#include <sensors/external-features.h>
#include <vi-map/vi-map.h>

DEFINE_int64(
    disturb_random_generator_seed, 0,
    "The seed for the random number generator for distrubing vertices.");
DEFINE_double(
    disturb_translation_random_walk_sigma_m, 0.005,
    "Standard deviation of the normal distribution determining the change in "
    "translation for each vertex.");
DEFINE_double(
    disturb_max_translational_disturbance_norm_m, 0.05,
    "The maximum translational disturbance that is applied to a vertex.");
DEFINE_double(
    disturb_yaw_angle_stddev_rad, 0.0005,
    "Standard deviation for determining the yaw angle disturbance.");
DEFINE_double(
    disturb_max_yaw_angle_rad, 0.05, "The maximum yaw angle disturbance.");

namespace vi_map_helpers {

VIMapManipulation::VIMapManipulation(vi_map::VIMap* map)
    : map_(*CHECK_NOTNULL(map)), geometry_(map_), queries_(map_) {}

void VIMapManipulation::rotate(const size_t dimension, const double degrees) {
  pose::Transformation T_G_old_G_new(
      Eigen::Vector3d(0, 0, 0),
      Eigen::Quaterniond(
          Eigen::AngleAxisd(
              degrees * kDegToRad, Eigen::Vector3d::Unit(dimension)))
          .normalized());
  vi_map::MissionBaseFrameIdList frames;
  map_.getAllMissionBaseFrameIds(&frames);
  for (const vi_map::MissionBaseFrameId& id : frames) {
    vi_map::MissionBaseFrame& frame = map_.getMissionBaseFrame(id);
    frame.set_T_G_M(T_G_old_G_new * frame.get_T_G_M());
  }
}

void VIMapManipulation::artificiallyDisturbVertices() {
  std::mt19937 generator(FLAGS_disturb_random_generator_seed);
  std::normal_distribution<double> position_distribution_m(
      0, FLAGS_disturb_translation_random_walk_sigma_m);
  std::normal_distribution<double> yaw_angle_distribution_rad(
      0, FLAGS_disturb_yaw_angle_stddev_rad);

  const Eigen::Vector3d G_disturbance_rotation_axis(0, 0, 1);

  vi_map::MissionIdList all_missions;
  map_.getAllMissionIds(&all_missions);
  for (const vi_map::MissionId& mission_id : all_missions) {
    pose_graph::VertexIdList all_vertices_in_mission;
    map_.getAllVertexIdsInMissionAlongGraph(
        mission_id, &all_vertices_in_mission);

    aslam::Position3D disturbance_translation(
        position_distribution_m(generator), position_distribution_m(generator),
        position_distribution_m(generator));
    double disturbance_yaw_angle_rad = yaw_angle_distribution_rad(generator);

    // Total disturbed mission frame.
    aslam::Transformation T_M_M_total_disturbance;
    aslam::Transformation last_T_M_I;
    const aslam::Transformation& T_G_M =
        map_.getMissionBaseFrameForMission(mission_id).get_T_G_M();
    const aslam::Transformation T_M_G = T_G_M.inverse();

    for (const pose_graph::VertexId& vertex_id : all_vertices_in_mission) {
      vi_map::Vertex* vertex = map_.getVertexPtr(vertex_id);
      const aslam::Transformation& current_T_M_I = vertex->get_T_M_I();

      // Disturbances are scaled to the distance of the current and the previous
      // vertex ("previous" vertex pose will evaluate to identity for the first
      // vertex in the graph). This is to avoid overly large disturbances in
      // places of little or not motion and will make the results of disturbing
      // a keyframed and a non-keyframed map comparable.
      const double vertex_distance_m =
          (current_T_M_I * last_T_M_I.inverse()).getPosition().norm();

      // Generate a translation.
      for (int i = 0; i < 3; ++i) {
        disturbance_translation[i] += position_distribution_m(generator);
      }
      if (disturbance_translation.norm() >
          FLAGS_disturb_max_translational_disturbance_norm_m) {
        disturbance_translation.normalize();
        disturbance_translation *=
            FLAGS_disturb_max_translational_disturbance_norm_m;
      }

      disturbance_yaw_angle_rad += yaw_angle_distribution_rad(generator);
      if (disturbance_yaw_angle_rad > FLAGS_disturb_max_yaw_angle_rad) {
        disturbance_yaw_angle_rad = FLAGS_disturb_max_yaw_angle_rad;
      } else if (disturbance_yaw_angle_rad < -FLAGS_disturb_max_yaw_angle_rad) {
        disturbance_yaw_angle_rad = -FLAGS_disturb_max_yaw_angle_rad;
      }
      const aslam::Quaternion orientation_disturbance(aslam::AngleAxis(
          disturbance_yaw_angle_rad * vertex_distance_m,
          G_disturbance_rotation_axis));

      const aslam::Transformation T_G_G_disturbance = aslam::Transformation(
          disturbance_translation * vertex_distance_m, orientation_disturbance);

      T_M_M_total_disturbance =
          T_M_M_total_disturbance * (T_M_G * T_G_G_disturbance * T_G_M);
      const aslam::Transformation new_T_M_I =
          T_M_M_total_disturbance * current_T_M_I;

      last_T_M_I = current_T_M_I;

      vertex->set_T_M_I(new_T_M_I);
    }
  }
}

void VIMapManipulation::alignToXYPlane(const vi_map::MissionId& mission_id) {
  CHECK(map_.hasMission(mission_id));
  // Let bv_M_z be the eigenvector of lowest eigenvalue of the vertex position
  // covariance. We then first rotate along bv_G_R_A x bv_G_z (z unit vector)
  // until bv_M_z x bv_G_z is minimized. We then rotate along the new bv_G_R_A
  // until bv_M_z x bv_G_z is zero.
  vi_map::MissionBaseFrame& base_frame =
      map_.getMissionBaseFrameForMission(mission_id);
  Eigen::Vector3d bv_M_z;
  Eigen::Vector3d bv_G_R_A_normalized;
  // bv_G_R_A x bv_G_z rotation
  {
    Eigen::Vector3d eigenvalues;
    Eigen::Matrix3d eigenvectors;
    geometry_.get_p_G_I_CovarianceEigenValuesAndVectorsAscending(
        mission_id, &eigenvalues, &eigenvectors);
    bv_M_z = eigenvectors.col(0).normalized();
    if (bv_M_z(2) < 0) {
      bv_M_z *= -1.;
    }
    bv_G_R_A_normalized =
        geometry_.get_bv_G_root_average(mission_id).normalized();
    // Angle to rotate along bv_G_R_A x bv_G_z = angle between bv_G_z and the
    // projection of bv_M_z into the plane bv_G_R_A x bv_G_z.
    // atan() intended, want range +- pi/2
    const double pitch_angle_rad =
        atan(bv_M_z.head<2>().dot(bv_G_R_A_normalized.head<2>()) / bv_M_z(2));
    const Eigen::Vector3d R_G_M_old_new =
        bv_G_R_A_normalized.cross(Eigen::Vector3d::UnitZ()).normalized() *
        pitch_angle_rad;
    const pose::Quaternion q_G_M_old_new(R_G_M_old_new);
    base_frame.set_T_G_M(
        pose::Transformation(q_G_M_old_new, Eigen::Vector3d::Zero()) *
        base_frame.get_T_G_M());

    bv_M_z = q_G_M_old_new.rotate(bv_M_z);
  }
  // bv_G_R_A rotation
  {
    const double roll_angle_rad = acos(bv_M_z(2));
    bv_G_R_A_normalized(2) = 0;
    bv_G_R_A_normalized.normalize();
    const pose::Quaternion q_G_M_old_new(
        Eigen::Vector3d(bv_G_R_A_normalized * -roll_angle_rad));
    base_frame.set_T_G_M(
        pose::Transformation(q_G_M_old_new, Eigen::Vector3d::Zero()) *
        base_frame.get_T_G_M());
  }
}

void VIMapManipulation::getViwlsEdgesWithoutImuMeasurements(
    const vi_map::MissionId& mission_id,
    pose_graph::EdgeIdList* corrupt_edge_ids) const {
  CHECK(map_.hasMission(mission_id));
  CHECK_NOTNULL(corrupt_edge_ids)->clear();

  pose_graph::EdgeIdList edge_ids;
  map_.getAllEdgeIdsInMissionAlongGraph(
      mission_id, pose_graph::Edge::EdgeType::kViwls, &edge_ids);

  CHECK(!edge_ids.empty()) << "No Viwls edges found in mission "
                           << mission_id.hexString();

  for (size_t i = 0; i < edge_ids.size(); ++i) {
    const vi_map::ViwlsEdge& edge =
        map_.getEdgeAs<vi_map::ViwlsEdge>(edge_ids[i]);
    if (edge.getImuData().size() == 0) {
      corrupt_edge_ids->push_back(edge_ids[i]);

      VLOG(2) << "Found edge with no IMU data: " << edge_ids[i].hexString();
      VLOG(2) << "Edge index within mission: " << i;
    }
  }
}

void VIMapManipulation::fixViwlsEdgesWithoutImuMeasurements(
    const vi_map::MissionId& mission_id) {
  CHECK(map_.hasMission(mission_id));

  pose_graph::EdgeIdList corrupt_edge_ids;
  getViwlsEdgesWithoutImuMeasurements(mission_id, &corrupt_edge_ids);

  if (corrupt_edge_ids.empty()) {
    VLOG(2) << "No corrupt edges found in mission: " << mission_id.hexString();
    return;
  }
  pose_graph::EdgeIdList all_edge_ids;
  map_.getAllEdgeIdsInMissionAlongGraph(mission_id, &all_edge_ids);
  CHECK_LT(corrupt_edge_ids.size(), all_edge_ids.size()) << "All Viwls edges "
                                                         << "are corrupt.";

  for (const pose_graph::EdgeId& edge_id : corrupt_edge_ids) {
    VLOG(2) << "Fixing edge " << edge_id;
    const vi_map::ViwlsEdge& current_edge =
        map_.getEdgeAs<vi_map::ViwlsEdge>(edge_id);

    pose_graph::EdgeId new_edge_id;
    aslam::generateId(&new_edge_id);

    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> new_imu_timestamps;
    Eigen::Matrix<double, 6, Eigen::Dynamic> new_imu_data;

    const pose_graph::VertexId& vertex_from = current_edge.from();
    const pose_graph::VertexId& vertex_to = current_edge.to();

    pose_graph::EdgeIdSet incoming_edges_from;
    map_.getVertex(vertex_from).getIncomingEdges(&incoming_edges_from);

    // Get the IMU measurements from the previous edge.
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_from;
    Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_from;
    for (const pose_graph::EdgeId& incoming_edge_id : incoming_edges_from) {
      if (map_.getEdgeType(incoming_edge_id) ==
          pose_graph::Edge::EdgeType::kViwls) {
        const vi_map::ViwlsEdge& incoming_edge =
            map_.getEdgeAs<vi_map::ViwlsEdge>(incoming_edge_id);

        imu_timestamps_from = incoming_edge.getImuTimestamps();
        imu_data_from = incoming_edge.getImuData();
        break;
      }
    }

    // Add the last IMU measurement from the previous edge.
    if (imu_timestamps_from.cols() > 0u) {
      new_imu_timestamps.conservativeResize(
          Eigen::NoChange, new_imu_timestamps.cols() + 1);
      new_imu_timestamps.col(new_imu_timestamps.cols() - 1) =
          imu_timestamps_from.rightCols(1);
      new_imu_data.conservativeResize(Eigen::NoChange, new_imu_data.cols() + 1);
      new_imu_data.col(new_imu_data.cols() - 1) = imu_data_from.rightCols(1);
    } else {
      // Since corrupt edges are in graph traversal order, if the previous
      // edge is corrupt it must be the first edge (otherwise it already
      // crashed), so we need to remove the first vertex in the mission
      // (vertex_from) and make vertex_to the new root vertex for the mission.
      map_.removeVertex(vertex_from);
      vi_map::VIMission& mission = map_.getMission(mission_id);
      mission.setRootVertexId(vertex_to);
      map_.removeEdge(edge_id);
      continue;
    }

    pose_graph::EdgeIdSet outgoing_edges_to;
    map_.getVertex(vertex_to).getOutgoingEdges(&outgoing_edges_to);

    // Get the IMU measurements from the following edge.
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_to;
    Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_to;
    for (const pose_graph::EdgeId& outgoing_edge_id : outgoing_edges_to) {
      if (map_.getEdgeType(outgoing_edge_id) ==
          pose_graph::Edge::EdgeType::kViwls) {
        const vi_map::ViwlsEdge& outgoing_edge =
            map_.getEdgeAs<vi_map::ViwlsEdge>(outgoing_edge_id);

        imu_timestamps_to = outgoing_edge.getImuTimestamps();
        imu_data_to = outgoing_edge.getImuData();
        break;
      }
    }
    if (imu_timestamps_to.cols() > 0u) {
      constexpr unsigned int kNumInterpolationTerms = 3u;
      new_imu_timestamps.conservativeResize(
          Eigen::NoChange,
          new_imu_timestamps.cols() + kNumInterpolationTerms + 1);
      new_imu_data.conservativeResize(
          Eigen::NoChange, new_imu_data.cols() + kNumInterpolationTerms + 1);
      // Add interpolation terms for stability.
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_interp_step =
          (imu_timestamps_to.leftCols(1) - imu_timestamps_from.rightCols(1)) /
          (kNumInterpolationTerms + 1);
      Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data_interp_step =
          (imu_data_to.leftCols(1) + imu_data_from.rightCols(1)) /
          (kNumInterpolationTerms + 1);
      for (unsigned int i = 0u; i < kNumInterpolationTerms; ++i) {
        const unsigned int index =
            new_imu_timestamps.cols() - kNumInterpolationTerms - 1 + i;
        new_imu_timestamps.col(index) =
            new_imu_timestamps.col(index - 1) + imu_timestamps_interp_step;
        new_imu_data.col(index) =
            new_imu_data.col(index - 1) + imu_data_interp_step;
      }
      // Add the first IMU measurement from the following edge.
      new_imu_timestamps.col(new_imu_timestamps.cols() - 1) =
          imu_timestamps_to.leftCols(1);
      new_imu_data.col(new_imu_data.cols() - 1) = imu_data_to.leftCols(1);
    } else {
      const bool kIsVertexToLastVertexInMission = outgoing_edges_to.size() == 0;
      CHECK(kIsVertexToLastVertexInMission) << "Consecutive corrupt edges.";
      // The corrupt edge is the last edge, so it can be safely removed along
      // with the last vertex.
      map_.removeEdge(edge_id);
      map_.removeVertex(vertex_to);
      continue;
    }

    VLOG(3) << "Adding the following IMU timestamps: " << new_imu_timestamps;
    VLOG(3) << "Adding the following IMU data: " << new_imu_data;
    vi_map::Edge* new_edge_ptr(new vi_map::ViwlsEdge(
        new_edge_id, vertex_from, vertex_to, new_imu_timestamps, new_imu_data));

    // Align the vertex_from and vertex_to to avoid overextending the edge.
    map_.getVertexPtr(vertex_to)->set_T_M_I(
        map_.getVertex(vertex_from).get_T_M_I());

    map_.removeEdge(edge_id);
    map_.addEdge(vi_map::Edge::UniquePtr(new_edge_ptr));
  }
}

void VIMapManipulation::removePosegraphAfter(
    const pose_graph::VertexId& vertex_id,
    pose_graph::VertexIdList* removed_vertex_ids) {
  CHECK_NOTNULL(removed_vertex_ids)->clear();
  CHECK(vertex_id.isValid());

  constexpr bool kIncludeStartingVertex = false;
  queries_.getFollowingVertexIdsAlongGraph(
      vertex_id, kIncludeStartingVertex, removed_vertex_ids);

  removeVerticesAndIncomingEdges(*removed_vertex_ids);
}

void VIMapManipulation::removeVerticesAndIncomingEdges(
    const pose_graph::VertexIdList& vertex_ids) {
  // Get all edges to delete.
  pose_graph::EdgeIdSet edges_to_remove;
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    const vi_map::Vertex& vertex = map_.getVertex(vertex_id);

    pose_graph::EdgeIdSet incoming_edges;
    vertex.getIncomingEdges(&incoming_edges);
    edges_to_remove.insert(incoming_edges.begin(), incoming_edges.end());
  }

  // Remove the elements from the map.
  for (const pose_graph::EdgeId& edge_id : edges_to_remove) {
    map_.removeEdge(edge_id);
  }

  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    map_.removeVertex(vertex_id);
  }
}

size_t VIMapManipulation::initializeLandmarksFromUnusedFeatureTracksOfMission(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex_id,
    vi_map::LandmarkIdList* initialized_landmark_ids) {
  CHECK(mission_id.isValid());
  pose_graph::VertexIdList all_vertices_in_missions;
  map_.getAllVertexIdsInMissionAlongGraph(
      mission_id, starting_vertex_id, &all_vertices_in_missions);

  MultiTrackIndexToLandmarkIdMap track_id_to_landmark_id;
  const size_t num_landmarks_initial = map_.numLandmarks();
  initializeLandmarksFromUnusedFeatureTracksOfOrderedVertices(
      all_vertices_in_missions, &track_id_to_landmark_id);

  const size_t num_new_landmarks = map_.numLandmarks() - num_landmarks_initial;
  if (initialized_landmark_ids) {
    for (const auto& track_index_map : track_id_to_landmark_id) {
      for (const auto& track_index_with_landmark_id : track_index_map.second) {
        initialized_landmark_ids->emplace_back(
            track_index_with_landmark_id.second);
      }
    }
    CHECK_EQ(initialized_landmark_ids->size(), num_new_landmarks);
  }
  return num_new_landmarks;
}

size_t VIMapManipulation::initializeLandmarksFromUnusedFeatureTracksOfMission(
    const vi_map::MissionId& mission_id,
    vi_map::LandmarkIdList* initialized_landmark_ids) {
  const vi_map::VIMission& mission = map_.getMission(mission_id);
  const pose_graph::VertexId& starting_vertex_id = mission.getRootVertexId();
  return initializeLandmarksFromUnusedFeatureTracksOfMission(
      mission_id, starting_vertex_id, initialized_landmark_ids);
}

void VIMapManipulation::
    initializeLandmarksFromUnusedFeatureTracksOfOrderedVertices(
        const pose_graph::VertexIdList& ordered_vertex_ids,
        MultiTrackIndexToLandmarkIdMap* multitrackid_landmarkid_map) {
  for (const pose_graph::VertexId& vertex_id : ordered_vertex_ids) {
    initializeLandmarksFromUnusedFeatureTracksOfVertex(
        vertex_id, multitrackid_landmarkid_map);
  }
}

void VIMapManipulation::initializeLandmarksFromUnusedFeatureTracksOfVertex(
    const pose_graph::VertexId& vertex_id,
    MultiTrackIndexToLandmarkIdMap* multitrackid_landmarkid_map) {
  CHECK_NOTNULL(multitrackid_landmarkid_map);
  const vi_map::Vertex& vertex = map_.getVertex(vertex_id);

  vertex.forEachFrame([this, &vertex, &multitrackid_landmarkid_map](
                          const size_t frame_index,
                          const aslam::VisualFrame& frame) {
    const size_t num_keypoints = frame.getNumKeypointMeasurements();
    if (!frame.hasTrackIds()) {
      VLOG(3) << "Frame has no tracking information. Skipping frame...";
      return;
    }
    const Eigen::VectorXi& track_ids = frame.getTrackIds();

    CHECK_EQ(static_cast<int>(num_keypoints), track_ids.rows());

    vi_map::LandmarkIdList landmark_ids;
    vertex.getFrameObservedLandmarkIds(frame_index, &landmark_ids);

    // Go over all tracks of this frame and add a new landmark if it wasn't
    // observed before, otherwise add an observation backlink.
    for (size_t keypoint_i = 0u; keypoint_i < num_keypoints; ++keypoint_i) {
      const int track_id = track_ids(keypoint_i);

      // Skip non-tracked landmark observation.
      if (track_id < 0 || landmark_ids[keypoint_i].isValid()) {
        continue;
      }

      // Check whether this feature type has been encountered before
      // and insert a track index map if necessary
      const int feature_type = frame.getDescriptorType(keypoint_i);
      const MultiTrackIndexKey track_index_key =
          std::make_pair(frame_index, feature_type);
      TrackIndexToLandmarkIdMap& trackid_landmarkid_map =
          (*multitrackid_landmarkid_map)[track_index_key];

      // Check whether this track has already a global landmark id
      // associated.
      const vi_map::LandmarkId* landmark_id_ptr =
          common::getValuePtr(trackid_landmarkid_map, track_id);

      if (landmark_id_ptr != nullptr && map_.hasLandmark(*landmark_id_ptr)) {
        map_.associateKeypointWithExistingLandmark(
            vertex.id(), frame_index, keypoint_i, *landmark_id_ptr);
      } else {
        // Assign a new global landmark id to this track if it hasn't
        // been seen before and add a new landmark to the map.
        vi_map::LandmarkId landmark_id =
            aslam::createRandomId<vi_map::LandmarkId>();
        // operator[] intended as this is either overwriting an old outdated
        // entry or creating a new one.
        trackid_landmarkid_map[track_id] = landmark_id;

        vi_map::KeypointIdentifier keypoint_id;
        keypoint_id.frame_id.frame_index = frame_index;
        keypoint_id.frame_id.vertex_id = vertex.id();
        keypoint_id.keypoint_index = keypoint_i;
        map_.addNewLandmark(
            landmark_id, keypoint_id,
            static_cast<vi_map::FeatureType>(feature_type));
      }
    }
  });
}

size_t VIMapManipulation::mergeLandmarksBasedOnTrackIds(
    const vi_map::MissionId& mission_id) {
  CHECK(map_.hasMission(mission_id));
  typedef std::unordered_map<int, vi_map::LandmarkId> TrackIdToLandmarkIdMap;
  TrackIdToLandmarkIdMap track_id_to_store_landmark_id;
  size_t num_merges = 0u;

  pose_graph::VertexIdList vertex_ids;
  map_.getAllVertexIdsInMission(mission_id, &vertex_ids);

  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    const vi_map::Vertex& vertex = map_.getVertex(vertex_id);

    for (size_t frame_idx = 0u; frame_idx < vertex.numFrames(); ++frame_idx) {
      vi_map::LandmarkIdList landmark_ids;
      vertex.getFrameObservedLandmarkIds(frame_idx, &landmark_ids);

      CHECK(vertex.getVisualFrame(frame_idx).hasTrackIds());
      const Eigen::VectorXi& track_ids =
          vertex.getVisualFrame(frame_idx).getTrackIds();
      CHECK_EQ(static_cast<size_t>(track_ids.rows()), landmark_ids.size());

      for (size_t keypoint_idx = 0u; keypoint_idx < landmark_ids.size();
           ++keypoint_idx) {
        if (landmark_ids[keypoint_idx].isValid() &&
            track_ids(keypoint_idx) < 0) {
          const vi_map::LandmarkId landmark_id = landmark_ids[keypoint_idx];
          CHECK(map_.hasLandmark(landmark_id));

          if (track_id_to_store_landmark_id
                  .emplace(track_ids(keypoint_idx), landmark_id)
                  .second) {
            // Emplace succeeded so this track ID was not used before.
          } else {
            // Emplace failed so this track ID is already used, we need to
            // merge landmarks.
            TrackIdToLandmarkIdMap::const_iterator it =
                track_id_to_store_landmark_id.find(track_ids(keypoint_idx));
            CHECK(it != track_id_to_store_landmark_id.end());
            CHECK(it->second.isValid());
            CHECK(map_.hasLandmark(it->second))
                << "Landmark " << it->second.hexString()
                << " not found in the map.";

            if (it->second != landmark_id) {
              map_.mergeLandmarks(landmark_id, it->second);
              ++num_merges;

              // As the dataset could be loop-closed before (so more than
              // a single track id pointing to the same store landmark, we need
              // to update the track id to store landmark id map.
              for (TrackIdToLandmarkIdMap::value_type& track_id_to_landmark :
                   track_id_to_store_landmark_id) {
                if (track_id_to_landmark.second == landmark_id) {
                  track_id_to_landmark.second = it->second;
                }
              }
            }
          }
        }
      }
    }
  }
  VLOG(2) << "Number of merges " << num_merges;
  return num_merges;
}

void VIMapManipulation::releaseOldVisualFrameImages(
    const pose_graph::VertexId& current_vertex_id,
    const int image_removal_age_threshold) {
  CHECK_GT(image_removal_age_threshold, 0);
  CHECK(map_.hasVertex(current_vertex_id));

  const vi_map::MissionId& mission_id =
      map_.getVertex(current_vertex_id).getMissionId();

  pose_graph::VertexId vertex_id = current_vertex_id;
  for (int i = 0; i < image_removal_age_threshold; ++i) {
    map_.getPreviousVertex(vertex_id, &vertex_id);
  }

  while (map_.getPreviousVertex(vertex_id, &vertex_id)) {
    const vi_map::Vertex& vertex = map_.getVertex(vertex_id);
    for (size_t i = 0; i < vertex.numFrames(); ++i) {
      const aslam::VisualFrame& const_vframe = vertex.getVisualFrame(i);
      if (const_vframe.isValid() && const_vframe.hasRawImage()) {
        aslam::VisualFrame::Ptr visual_frame =
            map_.getVertex(vertex_id).getVisualFrameShared(i);
        visual_frame->releaseRawImage();
        CHECK(!visual_frame->hasRawImage());
      }
    }
  }
}

size_t VIMapManipulation::removeBadLandmarks() {
  vi_map::LandmarkIdList bad_landmark_ids;
  queries_.getAllNotWellConstrainedLandmarkIds(&bad_landmark_ids);

  for (const vi_map::LandmarkId& bad_landmark_id : bad_landmark_ids) {
    map_.removeLandmark(bad_landmark_id);
  }
  return bad_landmark_ids.size();
}

void VIMapManipulation::dropMapDataBeforeVertex(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& new_root_vertex,
    const bool delete_resources_from_file_system) {
  pose_graph::VertexIdList all_vertices_in_mission;
  map_.getAllVertexIdsInMissionAlongGraph(mission_id, &all_vertices_in_mission);

  const vi_map::Vertex& root_vertex = map_.getVertex(new_root_vertex);
  const int64_t root_timestamp_ns = root_vertex.getMinTimestampNanoseconds();

  for (const pose_graph::VertexId& current_vertex_id :
       all_vertices_in_mission) {
    if (current_vertex_id == new_root_vertex) {
      break;
    }

    vi_map::Vertex* vertex = map_.getVertexPtr(current_vertex_id);

    // Delete all edges.
    pose_graph::EdgeIdSet incoming_edges;
    vertex->getIncomingEdges(&incoming_edges);
    for (pose_graph::EdgeId incoming_edge_id : incoming_edges) {
      if (map_.hasEdge(incoming_edge_id)) {
        map_.removeEdge(incoming_edge_id);
      }
    }
    pose_graph::EdgeIdSet outgoing_edges;
    vertex->getOutgoingEdges(&outgoing_edges);
    for (pose_graph::EdgeId outgoing_edge_id : outgoing_edges) {
      if (map_.hasEdge(outgoing_edge_id)) {
        map_.removeEdge(outgoing_edge_id);
      }
    }

    // Delete all landmarks.
    vi_map::LandmarkIdList landmark_ids;
    vertex->getStoredLandmarkIdList(&landmark_ids);
    for (const vi_map::LandmarkId& landmark_id : landmark_ids) {
      map_.removeLandmark(landmark_id);
      CHECK(!map_.hasLandmarkIdInLandmarkIndex(landmark_id));
      CHECK(!vertex->hasStoredLandmark(landmark_id));
      CHECK_EQ(vertex->getNumLandmarkObservations(landmark_id), 0u);
    }
    CHECK_EQ(0u, vertex->getLandmarks().size());

    // Remove references to global landmarks and landmark backlinks and frame
    // resources.
    for (unsigned int frame_idx = 0u; frame_idx < vertex->numFrames();
         ++frame_idx) {
      map_.deleteAllFrameResources(
          frame_idx, delete_resources_from_file_system, vertex);

      vi_map::LandmarkIdList frame_landmark_ids;
      vertex->getFrameObservedLandmarkIds(frame_idx, &frame_landmark_ids);

      for (size_t i = 0; i < frame_landmark_ids.size(); ++i) {
        if (frame_landmark_ids[i].isValid()) {
          const vi_map::LandmarkId& landmark_id = frame_landmark_ids[i];

          vi_map::Landmark& landmark = map_.getLandmark(landmark_id);
          landmark.removeAllObservationsOfVertex(current_vertex_id);

          // Also clean up the global landmark id list in the current vertex
          // so that we feed a consistent state to the removeVertex method
          // below.
          vi_map::LandmarkId invalid_landmark_id;
          invalid_landmark_id.setInvalid();
          vertex->setObservedLandmarkId(frame_idx, i, invalid_landmark_id);

          // If the current vertex was the only observer of the landmark stored
          // in some other mission, we should remove this orphaned landmark.
          if (map_.getMissionIdForLandmark(landmark_id) != mission_id &&
              !landmark.hasObservations()) {
            map_.removeLandmark(landmark_id);
            CHECK(!map_.hasLandmarkIdInLandmarkIndex(landmark_id));
            CHECK(!vertex->hasStoredLandmark(landmark_id));
            CHECK_EQ(vertex->getNumLandmarkObservations(landmark_id), 0u);
          }
        }
      }
    }
    map_.removeVertex(current_vertex_id);
  }

  map_.deleteAllSensorResourcesBeforeTime(
      mission_id, root_timestamp_ns, delete_resources_from_file_system);

  // Set vertex as root vertex.
  map_.getMission(mission_id).setRootVertexId(new_root_vertex);
}

uint32_t VIMapManipulation::addOdometryEdgesBetweenVertices(
    const uint32_t min_number_of_common_landmarks) {
  uint32_t num_edges_added = 0u;

  vi_map::MissionIdList mission_ids;
  map_.getAllMissionIds(&mission_ids);
  for (const vi_map::MissionId& mission_id : mission_ids) {
    const vi_map::VIMission& mission = map_.getMission(mission_id);

    if (!mission.hasOdometry6DoFSensor()) {
      LOG(ERROR) << "Cannot add odometry edges in between the vertices of "
                 << "mission " << mission_id
                 << " because it does not have an odometry sensor!";
      continue;
    }

    const aslam::SensorId& sensor_id = mission.getOdometry6DoFSensor();

    const vi_map::Odometry6DoF& odometry_sensor =
        map_.getSensorManager().getSensor<vi_map::Odometry6DoF>(sensor_id);

    const aslam::Transformation& T_B_S =
        map_.getSensorManager().getSensor_T_B_S(sensor_id);

    aslam::TransformationCovariance odometry_covariance;
    if (!odometry_sensor.get_T_St_Stp1_fixed_covariance(&odometry_covariance)) {
      LOG(ERROR) << "Cannot add odometry edges in between vertices if the "
                    "provided odometry sensor ("
                 << sensor_id << ") does not have a valid covariance!";
      return num_edges_added;
    }

    pose_graph::VertexIdList all_vertices_in_mission;
    map_.getAllVertexIdsInMissionAlongGraph(
        mission_id, &all_vertices_in_mission);

    vi_map_helpers::VIMapQueries vi_map_queries(map_);

    const bool add_vertices_based_on_common_landmarks =
        min_number_of_common_landmarks > 0u;

    for (uint32_t vertex_idx = 1u; vertex_idx < all_vertices_in_mission.size();
         ++vertex_idx) {
      const pose_graph::VertexId& current_vertex_id =
          all_vertices_in_mission[vertex_idx - 1];
      const pose_graph::VertexId& next_vertex_id =
          all_vertices_in_mission[vertex_idx];
      const vi_map::Vertex& current_vertex = map_.getVertex(current_vertex_id);
      const vi_map::Vertex& next_vertex = map_.getVertex(next_vertex_id);

      if (add_vertices_based_on_common_landmarks) {
        vi_map::LandmarkIdSet common_landmarks;
        const int num_common_landmarks =
            vi_map_queries.getNumberOfCommonLandmarks(
                current_vertex_id, next_vertex_id, &common_landmarks);
        if (num_common_landmarks >
            static_cast<int>(min_number_of_common_landmarks)) {
          continue;
        }
      }
      const aslam::Transformation& T_M_B_current = current_vertex.get_T_M_I();
      const aslam::Transformation& T_M_B_next = next_vertex.get_T_M_I();
      const aslam::Transformation T_S_current_S_next =
          T_B_S.inverse() * T_M_B_current.inverse() * T_M_B_next * T_B_S;

      const pose_graph::EdgeId edge_id =
          aslam::createRandomId<pose_graph::EdgeId>();

      map_.addEdge(aligned_unique<vi_map::TransformationEdge>(
          pose_graph::Edge::EdgeType::kOdometry, edge_id, current_vertex_id,
          next_vertex_id, T_S_current_S_next, odometry_covariance, sensor_id));

      ++num_edges_added;
    }
  }
  return num_edges_added;
}

bool VIMapManipulation::constrainStationarySubmapWithLoopClosureEdge(
    const double max_translation_m, const double max_rotation_rad) {
  CHECK_GE(max_translation_m, 0.0);
  CHECK_GE(max_rotation_rad, 0.0);

  vi_map::MissionIdList mission_ids;
  map_.getAllMissionIds(&mission_ids);

  if (mission_ids.size() > 1u) {
    LOG(ERROR) << "Constraining a potentially stationary submap with a loop "
               << "closure edge does not make sense for multi-mission maps!";
    return false;
  } else if (mission_ids.empty()) {
    LOG(ERROR)
        << "Constraining a potentially stationary submap with a loop "
        << "closure edge does not make sense for map without a single mission!";
    return false;
  }

  const vi_map::MissionId& mission_id = mission_ids.front();
  const vi_map::VIMission& mission = map_.getMission(mission_id);

  // If possible use covariance of odometry sensor, otherwise use gflags
  // value.
  aslam::TransformationCovariance T_B_first_B_last_covariance;
  bool odometry_sensor_has_covariance = false;
  if (mission.hasOdometry6DoFSensor()) {
    const aslam::SensorId& sensor_id = mission.getOdometry6DoFSensor();
    const vi_map::Odometry6DoF& odometry_sensor =
        map_.getSensorManager().getSensor<vi_map::Odometry6DoF>(sensor_id);
    aslam::TransformationCovariance odometry_covariance;
    odometry_sensor_has_covariance =
        odometry_sensor.get_T_St_Stp1_fixed_covariance(
            &T_B_first_B_last_covariance);
    LOG_IF(WARNING, !odometry_sensor_has_covariance)
        << "An odometry sensor is set, but no covariance, odometry "
        << "edges will be added with a hardcoded covariance.";
  }

  if (!odometry_sensor_has_covariance) {
    const double kOdometry6DoFCovarianceScaler = 1e-4;
    T_B_first_B_last_covariance.setIdentity();
    T_B_first_B_last_covariance *= kOdometry6DoFCovarianceScaler;
  }

  pose_graph::VertexIdList all_vertices_in_mission;
  map_.getAllVertexIdsInMissionAlongGraph(mission_id, &all_vertices_in_mission);

  if (all_vertices_in_mission.size() < 2u) {
    LOG(ERROR) << "Constraining a potentially stationary submap with a loop "
               << "closure edge does not make sense for map with less than two "
               << "vertices!";
    return false;
  }

  const pose_graph::VertexId& first_vertex_id = all_vertices_in_mission.front();
  const vi_map::Vertex& first_vertex = map_.getVertex(first_vertex_id);
  const aslam::Transformation& T_M_B_first = first_vertex.get_T_M_I();

  // Vertex id and transformation to the final vertex if the map turns out to be
  // stationary.
  pose_graph::VertexId last_vertex_id;
  aslam::Transformation T_B_first_B_last;

  for (uint32_t vertex_idx = 1u; vertex_idx < all_vertices_in_mission.size();
       ++vertex_idx) {
    last_vertex_id = all_vertices_in_mission[vertex_idx];
    const vi_map::Vertex& last_vertex = map_.getVertex(last_vertex_id);
    const aslam::Transformation& T_M_B_last = last_vertex.get_T_M_I();
    T_B_first_B_last = T_M_B_first.inverse() * T_M_B_last;

    // If at any point in this map we move past the translation or rotation
    // threshold, we don't consider this map as stationary.
    if (T_B_first_B_last.getPosition().norm() > max_translation_m) {
      return false;
    }
    if (std::abs(aslam::AngleAxis(T_B_first_B_last.getRotation()).angle()) >
        max_rotation_rad) {
      return false;
    }
  }

  // If we get to this point, the map is stationary and we will add a loop
  // closure edge between first and last vertex.
  const pose_graph::EdgeId loop_closure_edge_id =
      aslam::createRandomId<pose_graph::EdgeId>();
  const double kSwitchVariable = 1.0;
  const double kSwitchVariableVariance = 1e-8;
  map_.addEdge(aligned_unique<vi_map::LoopClosureEdge>(
      loop_closure_edge_id, first_vertex_id, last_vertex_id, kSwitchVariable,
      kSwitchVariableVariance, T_B_first_B_last, T_B_first_B_last_covariance));

  return true;
}

}  // namespace vi_map_helpers
