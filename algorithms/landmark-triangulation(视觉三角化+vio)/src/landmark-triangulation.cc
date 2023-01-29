#include "landmark-triangulation/landmark-triangulation.h"

#include <aslam/triangulation/triangulation.h>
#include <functional>
#include <maplab-common/multi-threaded-progress-bar.h>
#include <maplab-common/parallel-process.h>
#include <string>
#include <unordered_map>
#include <vi-map/landmark-quality-metrics.h>
#include <vi-map/vi-map.h>

#include "landmark-triangulation/pose-interpolator.h"

namespace landmark_triangulation {
typedef AlignedUnorderedMap<aslam::FrameId, aslam::Transformation>
    FrameToPoseMap;  // 用于存储帧的位姿

namespace {

void interpolateVisualFramePoses(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex_id, const vi_map::VIMap& map,
    FrameToPoseMap*
        interpolated_frame_poses) {  // 插值视觉帧的位姿，输入mission_id为当前的mission任务，starting_vertex_id为当前的vertex，map为地图，interpolated_frame_poses为输出的帧的位姿
  CHECK(map.hasMission(mission_id));  // 检查地图中是否有当前的mission
  CHECK_NOTNULL(interpolated_frame_poses)->clear();  // 清空输出的帧的位姿
  // Loop over all missions, vertices and frames and add the interpolated poses
  // to the map.
  size_t total_num_frames = 0u;

  // Check if there is IMU data.
  VertexToTimeStampMap vertex_to_time_map;
  PoseInterpolator imu_timestamp_collector;
  imu_timestamp_collector.getVertexToTimeStampMap(
      map, mission_id,
      &vertex_to_time_map);  // 获取当前mission的所有vertex的时间戳
  if (vertex_to_time_map.empty()) {
    VLOG(2) << "Couldn't find any IMU data to interpolate exact landmark "
               "observer positions in "
            << "mission " << mission_id;
    return;
  }

  pose_graph::VertexIdList vertex_ids;
  map.getAllVertexIdsInMissionAlongGraph(
      mission_id, starting_vertex_id,
      &vertex_ids);  // 获取当前mission的所有vertex的id

  // Compute upper bound for number of VisualFrames.
  const aslam::NCamera& ncamera =
      map.getMissionNCamera(mission_id);  // 获取当前mission的相机模型
  const unsigned int upper_bound_num_frames =
      vertex_ids.size() *
      ncamera.numCameras();  // 计算当前mission的所有帧的数量

  // Extract the timestamps for all VisualFrames.
  std::vector<aslam::FrameId> frame_ids(
      upper_bound_num_frames);  // 用于存储所有帧的id
  Eigen::Matrix<int64_t, 1, Eigen::Dynamic> pose_timestamps(
      upper_bound_num_frames);  // 用于存储所有帧的时间戳
  unsigned int frame_counter = 0u;
  for (const pose_graph::VertexId& vertex_id :
       vertex_ids) {  // 遍历所有的vertex
    const vi_map::Vertex& vertex = map.getVertex(vertex_id);
    for (unsigned int frame_idx = 0u; frame_idx < vertex.numFrames();
         ++frame_idx) {                           // 遍历所有的帧
      if (vertex.isFrameIndexValid(frame_idx)) {  // 检查帧的索引是否有效
        CHECK_LT(frame_counter, upper_bound_num_frames);
        const aslam::VisualFrame& visual_frame =
            vertex.getVisualFrame(frame_idx);  // 获取帧
        const int64_t timestamp =
            visual_frame.getTimestampNanoseconds();  // 获取帧的时间戳
        // Only interpolate if the VisualFrame timestamp and the vertex
        // timestamp do not match.
        VertexToTimeStampMap::const_iterator it = vertex_to_time_map.find(
            vertex_id);  // 获取当前帧对应的vertex的时间戳
        if (it != vertex_to_time_map.end() && it->second != timestamp) {
          pose_timestamps(0, frame_counter) =
              timestamp;  // 将帧的时间戳存储到pose_timestamps中
          frame_ids[frame_counter] =
              visual_frame.getId();  // 将帧的id存储到frame_ids中
          ++frame_counter;
        }
      }
    }
  }

  // Shrink time stamp and frame id arrays if necessary.
  if (upper_bound_num_frames >
      frame_counter) {  // 如果帧的数量大于实际的帧的数量
    frame_ids.resize(frame_counter);
    pose_timestamps.conservativeResize(
        Eigen::NoChange, frame_counter);  // 调整pose_timestamps的大小
  }

  // Reserve enough space in the frame_to_pose_map to add the poses of the
  // current mission.
  total_num_frames += frame_counter;
  interpolated_frame_poses->reserve(
      total_num_frames);  // 为interpolated_frame_poses分配空间

  // Interpolate poses for all the VisualFrames.
  if (frame_counter > 0u) {
    VLOG(1) << "Interpolating the exact visual frame poses for "
            << frame_counter << " frames of mission " << mission_id;
    PoseInterpolator pose_interpolator;     // 位姿插值器
    aslam::TransformationVector poses_M_I;  // 用于存储插值后的位姿
    pose_interpolator.getPosesAtTime(
        map, mission_id, pose_timestamps, &poses_M_I);  // 获取插值后的位姿
    CHECK_EQ(poses_M_I.size(), frame_counter);
    for (size_t frame_num = 0u; frame_num < frame_counter; ++frame_num) {
      interpolated_frame_poses->emplace(
          frame_ids[frame_num],
          poses_M_I.at(
              frame_num));  // 将插值后的位姿存储到interpolated_frame_poses中
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
    FrameToPoseMap* interpolated_frame_poses) {  // 多态，用于获取插值后的位姿
  CHECK(map.hasMission(mission_id));
  CHECK_NOTNULL(interpolated_frame_poses)->clear();
  const vi_map::VIMission& mission = map.getMission(mission_id);
  const pose_graph::VertexId& starting_vertex_id = mission.getRootVertexId();
  interpolateVisualFramePoses(
      mission_id, starting_vertex_id, map,
      interpolated_frame_poses);  // 用于获取插值后的位姿
}

void retriangulateLandmarksOfVertex(
    const FrameToPoseMap& interpolated_frame_poses,
    pose_graph::VertexId storing_vertex_id, vi_map::VIMap* map,
    bool only_included_ids,
    const vi_map::LandmarkIdSet&
        landmark_ids) {  // 用于重建地图点,输入的参数有interpolated_frame_poses作为插值后的位姿，storing_vertex_id作为存储地图点的vertex，map作为地图，only_included_ids表示是否只重建landmark_ids中的地图点，landmark_ids作为需要重建的地图点的id
  CHECK_NOTNULL(map);
  vi_map::Vertex& storing_vertex =
      map->getVertex(storing_vertex_id);  // 获取存储地图点的vertex
  vi_map::LandmarkStore& landmark_store =
      storing_vertex.getLandmarks();  // 获取存储地图点的vertex中的地图点

  const aslam::Transformation& T_M_I_storing =
      storing_vertex.get_T_M_I();  // 获取存储地图点的vertex的位姿
  const aslam::Transformation& T_G_M_storing =
      const_cast<const vi_map::VIMap*>(map)
          ->getMissionBaseFrameForVertex(storing_vertex_id)
          .get_T_G_M();  // 获取存储地图点的vertex所在的mission的位姿
  const aslam::Transformation T_G_I_storing =
      T_G_M_storing * T_M_I_storing;  // 获取存储地图点的vertex的位姿

  for (vi_map::Landmark& landmark :
       landmark_store) {      // 遍历存储地图点的vertex中的地图点
    if (only_included_ids) {  // 如果只重建landmark_ids中的地图点
      if (landmark_ids.find(landmark.id()) == landmark_ids.end()) {
        continue;
      }
    }

    landmark.setQuality(
        vi_map::Landmark::Quality::kBad);  // 将地图点的质量设置为kBad

    // Triangulation is handled differently for LiDAR and camera landmarks
    const bool is_visual_landmark = !vi_map::isLidarFeature(
        landmark.getFeatureType());  // 判断地图点是否为视觉地图点

    // The following have one entry per measurement:
    Eigen::Matrix3Xd G_bearing_vectors;
    Eigen::Matrix3Xd p_G_C_vectors;
    Eigen::Matrix3Xd p_G_fi_vectors;

    const vi_map::KeypointIdentifierList& observations =
        landmark.getObservations();  // 获取地图点的观测

    if (is_visual_landmark) {  // 如果地图点为视觉地图点
      // If we don't have enough observations for a 2D visual landmark we can
      // already abort here. For LiDAR landmarks one observation is enough.
      if (observations.size() <
          2u) {  // 如果地图点的观测数量小于2，说明地图点的观测数量不够，直接跳过
        continue;
      }

      G_bearing_vectors.resize(
          Eigen::NoChange, observations.size());  // 初始化G_bearing_vectors
      p_G_C_vectors.resize(Eigen::NoChange, observations.size());
    } else {
      p_G_fi_vectors.resize(Eigen::NoChange, observations.size());
    }

    int num_measurements = 0;
    for (const vi_map::KeypointIdentifier& observation :
         observations) {  // 遍历地图点的观测
      const pose_graph::VertexId& observer_id = observation.frame_id.vertex_id;
      CHECK(map->hasVertex(observer_id))
          << "Observer " << observer_id << " of store landmark "
          << landmark.id() << " not in currently loaded map!";

      const vi_map::Vertex& observer =
          const_cast<const vi_map::VIMap*>(map)->getVertex(
              observer_id);  // 获取观测地图点的vertex
      const size_t frame_idx = observation.frame_id.frame_index;
      const aslam::VisualFrame& visual_frame = observer.getVisualFrame(
          frame_idx);  // 获取观测地图点的vertex中的frame
      const aslam::Transformation& T_G_M_observer =
          const_cast<const vi_map::VIMap*>(map)
              ->getMissionBaseFrameForVertex(observer_id)
              .get_T_G_M();  // 获取观测地图点的vertex所在的mission的位姿

      aslam::Transformation T_M_I_observer;
      if (is_visual_landmark) {  // 如果地图点为视觉地图点
        FrameToPoseMap::const_iterator it = interpolated_frame_poses.find(
            visual_frame.getId());                   // 获取视觉帧的位姿
        if (it != interpolated_frame_poses.end()) {  // 如果还有视觉帧的位姿
          // If there are precomputed/interpolated T_M_I, use those.
          T_M_I_observer = it->second;  // 获取视觉帧的位姿
        } else {
          // TODO(smauq): add linear interpolation here on the fly
          T_M_I_observer =
              observer.get_T_M_I();  // 获取观测地图点的vertex的位姿
        }
      } else {
        const int64_t offset = visual_frame.getKeypointTimeOffset(
            observation.keypoint_index);  // 获取视觉帧的时间偏移量

        if (offset != 0) {
          // For LiDAR landmarks we have to compensate for an arbitrary time
          // offset so it can't be precomputed. We use linear interpolation
          // instead of the IMU based one which would take too long
          const bool success = interpolateLinear(
              *map, observer, offset,
              &T_M_I_observer);  // 根据offset获取对应时刻的位姿

          if (!success) {
            continue;
          }
        }
      }

      const aslam::Transformation& T_I_C =
          observer.getNCameras()
              ->get_T_C_B(frame_idx)
              .inverse();  // 获取观测地图点的vertex中的frame的位姿
      aslam::Transformation T_G_C =
          T_G_M_observer * T_M_I_observer *
          T_I_C;  // 获取观测地图点的vertex中的frame的位姿

      if (is_visual_landmark) {
        Eigen::Vector2d measurement = visual_frame.getKeypointMeasurement(
            observation.keypoint_index);  // 获取视觉帧的特征点

        Eigen::Vector3d C_bearing_vector;
        bool projection_result = observer.getCamera(frame_idx)->backProject3(
            measurement, &C_bearing_vector);  // 将特征点转换为3D坐标
        if (!projection_result) {
          continue;
        }

        G_bearing_vectors.col(num_measurements) =
            T_G_C.getRotationMatrix() *
            C_bearing_vector;  // 将3D坐标转换到世界坐标系下
        p_G_C_vectors.col(num_measurements) =
            T_G_C.getPosition();  // 获取观测地图点的vertex中的frame的位姿
      } else {
        Eigen::Vector3d p_C_fi = visual_frame.getKeypoint3DPosition(
            observation.keypoint_index);  // 获取视觉帧的特征点的3D坐标
        p_G_fi_vectors.col(num_measurements) =
            T_G_C * p_C_fi;  // 将3D坐标转换到世界坐标系下
      }

      ++num_measurements;
    }

    Eigen::Vector3d p_G_fi;
    if (is_visual_landmark) {
      if (num_measurements < 2) {
        continue;
      }

      // Resize to final number of valid measurements
      G_bearing_vectors.conservativeResize(
          Eigen::NoChange,
          num_measurements);  // 将G_bearing_vectors的列数调整为num_measurements
      p_G_C_vectors.conservativeResize(
          Eigen::NoChange,
          num_measurements);  // 将p_G_C_vectors的列数调整为num_measurements

      // Triangulate using collected bearing vectors and camera positions
      aslam::TriangulationResult triangulation_result =
          aslam::linearTriangulateFromNViews(
              G_bearing_vectors, p_G_C_vectors,
              &p_G_fi);  // 通过多视角三角化获取地图点的3D坐标

      if (!triangulation_result
               .wasTriangulationSuccessful()) {  // 如果三角化失败
        continue;
      }
    } else {
      if (num_measurements < 1) {  // 如果视觉帧的特征点的数量小于1
        continue;
      }

      // Resize to final number of valid measurements
      p_G_fi_vectors.conservativeResize(
          Eigen::NoChange,
          num_measurements);  // 将p_G_fi_vectors的列数调整为num_measurements

      // Triangulate by averaging the 3D measurements in global frame
      p_G_fi = p_G_fi_vectors.rowwise()
                   .mean();  // 通过多视角三角化获取地图点的3D坐标
    }

    landmark.set_p_B(
        T_G_I_storing.inverse() *
        p_G_fi);  // 将地图点的3D坐标转换到存储地图点的vertex的坐标系下
    constexpr bool kReEvaluateQuality = true;
    if (vi_map::isLandmarkWellConstrained(
            *map, landmark, kReEvaluateQuality)) {  // 判断地图点是否被充分观测
      landmark.setQuality(
          vi_map::Landmark::Quality::kGood);  // 设置地图点的质量为kGood
    }
  }
}

void retriangulateLandmarksOfMission(
    const vi_map::MissionId& mission_id,
    const pose_graph::VertexId& starting_vertex_id,
    const FrameToPoseMap& interpolated_frame_poses, vi_map::VIMap* map,
    bool only_included_ids = false,
    const vi_map::LandmarkIdSet& landmark_ids = vi_map::
        LandmarkIdSet()) {  // 判断重新三角化地图点的质量，输入为任务id--mission_id，起始顶点id--starting_vertex_id，插值后的帧位姿--interpolated_frame_poses，地图--map，是否只包含包含的id--only_included_ids，地图点id集合--landmark_ids
  CHECK_NOTNULL(map);

  VLOG(1) << "Getting vertices of mission: " << mission_id;
  pose_graph::VertexIdList relevant_vertex_ids;
  map->getAllVertexIdsInMissionAlongGraph(
      mission_id, starting_vertex_id,
      &relevant_vertex_ids);  // 获取任务id为mission_id的所有顶点id

  const size_t num_vertices = relevant_vertex_ids.size();  // 获取顶点的数量
  VLOG(1) << "Retriangulating landmarks of " << num_vertices << " vertices.";

  common::MultiThreadedProgressBar progress_bar;
  std::function<void(const std::vector<size_t>&)> retriangulator =
      [&relevant_vertex_ids, map, landmark_ids, only_included_ids,
       &progress_bar,
       &interpolated_frame_poses](
          const std::vector<size_t>& batch) {  // 重新三角化地图点的函数
        progress_bar.setNumElements(batch.size());  // 设置进度条的元素数量
        size_t num_processed = 0u;
        for (size_t item : batch) {  // 遍历batch中的元素
          CHECK_LT(
              item, relevant_vertex_ids.size());  // 检查item是否小于顶点的数量
          retriangulateLandmarksOfVertex(
              interpolated_frame_poses, relevant_vertex_ids[item], map,
              only_included_ids, landmark_ids);  // 重新三角化地图点
          progress_bar.update(++num_processed);  // 更新进度条
        }
      };

  static constexpr bool kAlwaysParallelize = false;
  constexpr size_t kMaxNumHardwareThreads = 8u;
  const size_t available_num_threads =
      common::getNumHardwareThreads();  // 获取硬件线程的数量
  const size_t num_threads = (available_num_threads < kMaxNumHardwareThreads)
                                 ? available_num_threads
                                 : kMaxNumHardwareThreads;  // 获取线程的数量
  common::ParallelProcess(
      num_vertices, retriangulator, kAlwaysParallelize,
      num_threads);  // 并行处理
}

void retriangulateLandmarksOfMission(
    const vi_map::MissionId& mission_id,
    const FrameToPoseMap& interpolated_frame_poses, vi_map::VIMap* map,
    bool only_included_ids, const vi_map::LandmarkIdSet& landmark_ids) {
  CHECK_NOTNULL(map);
  const vi_map::VIMission& mission = map->getMission(mission_id);  // 获取任务
  const pose_graph::VertexId& starting_vertex_id =
      mission.getRootVertexId();  // 获取任务的根顶点id
  retriangulateLandmarksOfMission(
      mission_id, starting_vertex_id, interpolated_frame_poses, map,
      only_included_ids, landmark_ids);  // 重新三角化地图点
}
}  // namespace

void retriangulateLandmarks(
    const vi_map::MissionIdList& mission_ids,
    vi_map::VIMap* map) {  // 重新三角化地图点
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
    const pose_graph::VertexId& starting_vertex,
    vi_map::VIMap* map) {  // 只在任务中的顶点执行地图点重新三角化
  CHECK_NOTNULL(map);
  FrameToPoseMap interpolated_frame_poses;
  interpolateVisualFramePoses(
      mission_id, starting_vertex, *map, &interpolated_frame_poses);  // 插值
  retriangulateLandmarksOfMission(
      mission_id, starting_vertex, interpolated_frame_poses,
      map);  // 重新三角化地图点
}

void retriangulateLandmarks(vi_map::VIMap* map) {  // 重新三角化地图点
  vi_map::MissionIdList mission_ids;
  map->getAllMissionIds(&mission_ids);
  retriangulateLandmarks(mission_ids, map);
}

void retriangulateLandmarksOfVertex(
    const pose_graph::VertexId& storing_vertex_id, vi_map::VIMap* map,
    bool only_included_ids,
    const vi_map::LandmarkIdSet& landmark_ids) {  // 重新三角化地图点根据顶点id
  CHECK_NOTNULL(map);
  FrameToPoseMap empty_frame_to_pose_map;
  retriangulateLandmarksOfVertex(
      empty_frame_to_pose_map, storing_vertex_id, map, only_included_ids,
      landmark_ids);
}

}  // namespace landmark_triangulation
