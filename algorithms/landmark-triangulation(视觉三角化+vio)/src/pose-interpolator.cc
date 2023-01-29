#include "landmark-triangulation/pose-interpolator.h"

#include <aslam/common/time.h>
#include <glog/logging.h>
#include <imu-integrator/imu-integrator.h>
#include <limits>
#include <maplab-common/macros.h>

namespace landmark_triangulation {

bool interpolateLinear(
    const vi_map::VIMap& map, const vi_map::Vertex& vertex, int64_t offset_ns,
    aslam::Transformation*
        T_inter) {  // 通过线性插值计算位姿，输入为地图--map，位姿--vertex，时间偏移量--offset_ns，输出为插值后的位姿--T_inter
  CHECK_NOTNULL(T_inter);
  // TODO(smauq): Implement interpolation also backwards
  CHECK_GT(offset_ns, 0);

  // We can not interpolate for the last vertex
  pose_graph::VertexId next_vertex_id;
  if (!map.getNextVertex(vertex.id(), &next_vertex_id)) {  // 获取下一个顶点的id
    return false;
  }

  const vi_map::Vertex& next_vertex =
      map.getVertex(next_vertex_id);  // 获取下一个顶点

  const int64_t t1 =
      vertex.getMinTimestampNanoseconds();  // 获取当前顶点的时间戳
  const int64_t t2 =
      next_vertex.getMinTimestampNanoseconds();  // 获取下一个顶点的时间戳
  const double lambda = static_cast<double>(offset_ns) /
                        static_cast<double>(t2 - t1);  // 计算插值系数

  *T_inter = kindr::minimal::interpolateComponentwise(
      vertex.get_T_M_I(), next_vertex.get_T_M_I(),
      lambda);  // 通过线性插值计算位姿

  return true;
}

void PoseInterpolator::buildListOfAllRequiredIMUMeasurements(
    const vi_map::VIMap& map, const std::vector<int64_t>& timestamps,
    const pose_graph::EdgeId& imu_edge_id, int start_index, int end_index,
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
    Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_data)
    const {  // 通过IMU边缘计算位姿，输入为地图--map，时间戳--timestamps，IMU边缘--imu_edge_id，起始索引--start_index，结束索引--end_index，输出为IMU时间戳--imu_timestamps，IMU数据--imu_data
  CHECK_NOTNULL(imu_timestamps);
  CHECK_NOTNULL(imu_data);
  CHECK_LT(start_index, static_cast<int>(timestamps.size()));
  CHECK_LT(end_index, static_cast<int>(timestamps.size()));
  CHECK_GE(start_index, 0);
  CHECK_GE(end_index, 0);

  // First add all imu measurements from this vertex to the buffer.
  typedef std::pair<const int64_t, IMUMeasurement>
      buffer_value_type;  // 定义一个pair类型，第一个元素为int64_t类型，第二个元素为IMUMeasurement类型
  using common::TemporalBuffer;
  typedef TemporalBuffer<
      IMUMeasurement, Eigen::aligned_allocator<buffer_value_type>>
      ImuMeasurementBuffer;  // 定义一个TemporalBuffer类型，该类型的元素为buffer_value_type类型
  ImuMeasurementBuffer imu_buffer;
  {
    const vi_map::ViwlsEdge& imu_edge =
        map.getEdgeAs<vi_map::ViwlsEdge>(imu_edge_id);  // 获取IMU边缘
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps =
        imu_edge.getImuTimestamps();  // 获取IMU时间戳
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data =
        imu_edge.getImuData();  // 3x1 accel, 3x1 gyro, 获取IMU数据
    CHECK_EQ(
        imu_timestamps.cols(),
        imu_data.cols());  // 检查IMU时间戳和IMU数据的列数是否相等
    for (int i = 0; i < imu_data.cols(); ++i) {
      IMUMeasurement measurement;
      measurement.imu_measurement = imu_data.col(i);  // 获取IMU数据的第i列
      measurement.timestamp = imu_timestamps(0, i);  // 获取IMU时间戳的第i列
      imu_buffer.addValue(
          measurement.timestamp,
          measurement);  // 将IMU时间戳和IMU数据添加到imu_buffer中
    }
  }

  // Now compute the interpolated values for the requested items in case
  // we don't have IMU measurements at the particular time.
  for (int i = start_index; i <= end_index; ++i) {
    int64_t requested_time = timestamps[i];
    IMUMeasurement measurement;
    bool have_value_at_time = imu_buffer.getValueAtTime(
        requested_time,
        &measurement);  // 获取IMU时间戳为requested_time的IMU数据
    if (have_value_at_time) {
      // No need for interpolation of IMU measurements.
      continue;
    }
    // Get the IMU measurement closest before and after the requested
    // time.
    IMUMeasurement measurement_before;
    int64_t timestamp_before = 0;
    CHECK(imu_buffer.getValueAtOrBeforeTime(
        requested_time, &timestamp_before,
        &measurement_before));  // 获取IMU时间戳小于等于requested_time的最大值
    IMUMeasurement
        measurement_after;  // 获取IMU时间戳大于等于requested_time的最小值
    int64_t timestamp_after = 0;
    CHECK(imu_buffer.getValueAtOrAfterTime(
        requested_time, &timestamp_after,
        &measurement_after));  // 获取IMU时间戳大于等于requested_time的最小值

    CHECK_NE(
        timestamp_after,
        timestamp_before);  // 检查IMU时间戳的最大值和最小值是否相等

    // Interpolate the IMU measurement.
    IMUMeasurement interpolated_measurement;
    interpolated_measurement.timestamp = requested_time;  // 获取IMU时间戳
    double alpha =
        static_cast<double>(requested_time - timestamp_before) /
        static_cast<double>(timestamp_after - timestamp_before);  // 计算alpha
    CHECK_GT(alpha, 0.0);
    CHECK_LT(alpha, 1.0);
    interpolated_measurement.imu_measurement =
        (1 - alpha) * measurement_before.imu_measurement +
        alpha * measurement_after.imu_measurement;  // 计算插值后的IMU数据
    imu_buffer.addValue(
        interpolated_measurement.timestamp,
        interpolated_measurement);  // 将插值后的IMU数据添加到imu_buffer中
  }

  imu_timestamps->resize(
      Eigen::NoChange, imu_buffer.size());  // 设置IMU时间戳的大小
  imu_data->resize(Eigen::NoChange, imu_buffer.size());  // 设置IMU数据的大小
  int index = 0;
  for (const buffer_value_type& value : imu_buffer) {
    (*imu_timestamps)(0, index) = value.second.timestamp;  // 获取IMU时间戳
    imu_data->col(index) = value.second.imu_measurement;   // 获取IMU数据
    ++index;
  }
}

void PoseInterpolator::computeRequestedPosesInRange(
    const vi_map::VIMap& map, const vi_map::VIMission& mission,
    const pose_graph::VertexId& vertex_begin_id,
    const pose_graph::EdgeId& /*imu_edge_id*/,
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data,
    StateBuffer* state_buffer)
    const {  // 计算请求范围内的位姿,输入参数为地图--map，任务--mission，起始顶点--vertex_begin_id，IMU时间戳--imu_timestamps，IMU数据--imu_data，状态缓冲区--state_buffer
  CHECK_NOTNULL(state_buffer);
  CHECK_EQ(imu_timestamps.cols(), imu_data.cols());
  if (imu_data.cols() == 0) {
    return;
  }

  using imu_integrator::ImuIntegratorRK4;
  const vi_map::MissionId& mission_id = mission.id();  // 获取任务ID
  CHECK(mission_id.isValid());  // 检查任务ID是否有效
  const vi_map::Imu& imu_sensor =
      map.getMissionImu(mission_id);  // 获取任务对应的IMU传感器
  const vi_map::ImuSigmas& imu_sigmas =
      imu_sensor.getImuSigmas();  // 获取IMU传感器的IMU噪声参数

  ImuIntegratorRK4 integrator(
      imu_sigmas.gyro_noise_density,
      imu_sigmas.gyro_bias_random_walk_noise_density,
      imu_sigmas.acc_noise_density,
      imu_sigmas.acc_bias_random_walk_noise_density,
      imu_sensor.getGravityMagnitudeMps2());  // 构造IMU积分器

  using imu_integrator::kAccelBiasBlockSize;    // 加速度偏置块大小
  using imu_integrator::kAccelReadingOffset;    // 加速度读数偏移量
  using imu_integrator::kErrorStateSize;        // 误差状态大小
  using imu_integrator::kGyroBiasBlockSize;     // 陀螺仪偏置块大小
  using imu_integrator::kGyroReadingOffset;     // 陀螺仪读数偏移量
  using imu_integrator::kImuReadingSize;        // IMU读数大小
  using imu_integrator::kNanoSecondsToSeconds;  // 纳秒转秒
  using imu_integrator::kPositionBlockSize;     // 位置块大小
  using imu_integrator::kStateAccelBiasOffset;  // 加速度偏置状态偏移量
  using imu_integrator::kStateGyroBiasOffset;  // 陀螺仪偏置状态偏移量
  using imu_integrator::kStateOrientationBlockSize;  // 状态方向块大小
  using imu_integrator::kStatePositionOffset;        // 位置状态偏移量
  using imu_integrator::kStateSize;                  // 状态大小
  using imu_integrator::kStateVelocityOffset;        // 速度状态偏移量
  using imu_integrator::kVelocityBlockSize;          // 速度块大小

  Eigen::Matrix<double, 2 * kImuReadingSize, 1>
      debiased_imu_readings;  // 去偏置的IMU读数
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize> phi;  // 状态转移矩阵
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>
      new_phi_accum;  // 新的状态转移矩阵累加
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>
      Q;  // 状态噪声协方差矩阵
  Eigen::Matrix<double, kErrorStateSize, kErrorStateSize>
      new_Q_accum;  // 新的状态噪声协方差矩阵累加
  Eigen::Matrix<double, kStateSize, 1> current_state;  // 当前状态
  Eigen::Matrix<double, kStateSize, 1> next_state;     // 下一个状态

  const vi_map::Vertex& vertex_from =
      map.getVertex(vertex_begin_id);  // 获取起始顶点
  const aslam::Transformation T_M_I =
      vertex_from.get_T_M_I();  // 获取起始顶点的位姿

  // Active to passive and direction switch, so no inversion.
  const Eigen::Matrix<double, 4, 1>& q_I_M_from =
      T_M_I.getRotation().toImplementation().coeffs();  // 获取起始顶点的四元数
  const Eigen::Matrix<double, 3, 1>& p_M_I_from =
      T_M_I.getPosition();  // 获取起始顶点的位置
  const Eigen::Matrix<double, 3, 1>& v_M_I_from =
      vertex_from.get_v_M();  // 获取起始顶点的速度
  const Eigen::Matrix<double, 3, 1>& b_g_from =
      vertex_from.getGyroBias();  // 获取起始顶点的陀螺仪偏置
  const Eigen::Matrix<double, 3, 1>& b_a_from =
      vertex_from.getAccelBias();  // 获取起始顶点的加速度偏置

  current_state << q_I_M_from, b_g_from, v_M_I_from, b_a_from,
      p_M_I_from;  // 当前状态

  // Store the value where we start integration.
  StateLinearizationPoint state_linearization_point_begin;
  state_linearization_point_begin.timestamp = imu_timestamps(0, 0);
  state_linearization_point_begin.q_M_I.coeffs() =
      current_state.head<kStateOrientationBlockSize>();  // 获取当前状态的四元数
  state_linearization_point_begin.p_M_I =
      current_state.segment<kPositionBlockSize>(
          kStatePositionOffset);  // 获取当前状态的位置
  current_state.head<kStateOrientationBlockSize>();  // 获取当前状态的四元数
  state_linearization_point_begin.v_M =
      current_state.segment<kVelocityBlockSize>(
          kStateVelocityOffset);  // 获取当前状态的速度
  state_linearization_point_begin.accel_bias =
      current_state.segment<kAccelBiasBlockSize>(
          kStateAccelBiasOffset);  // 获取当前状态的加速度偏置
  state_linearization_point_begin.gyro_bias =
      current_state.segment<kGyroBiasBlockSize>(
          kStateGyroBiasOffset);  // 获取当前状态的陀螺仪偏置
  state_buffer->addValue(
      state_linearization_point_begin.timestamp,
      state_linearization_point_begin);  // 添加到状态缓冲区

  // Now compute all the integrated values.
  for (int i = 0; i < imu_data.cols() - 1; ++i) {
    CHECK_GE(
        imu_timestamps(0, i + 1),
        imu_timestamps(0, i))  // 检查IMU数据是否按时间顺序排列
        << "IMU measurements not properly ordered";

    Eigen::Vector3d current_gyro_bias =
        current_state.segment<kGyroBiasBlockSize>(
            kStateGyroBiasOffset);  // 获取当前状态的陀螺仪偏置
    Eigen::Vector3d current_accel_bias =
        current_state.segment<kAccelBiasBlockSize>(
            kStateAccelBiasOffset);  // 获取当前状态的加速度偏置

    debiased_imu_readings << imu_data.col(i).segment<3>(kAccelReadingOffset) -
                                 current_accel_bias,
        imu_data.col(i).segment<3>(kGyroReadingOffset) - current_gyro_bias,
        imu_data.col(i + 1).segment<3>(kAccelReadingOffset) -
            current_accel_bias,
        imu_data.col(i + 1).segment<3>(kGyroReadingOffset) -
            current_gyro_bias;  // 去除偏置后的IMU数据

    double delta_time_seconds =
        (imu_timestamps(0, i + 1) - imu_timestamps(0, i)) *
        kNanoSecondsToSeconds;  // 两个IMU数据之间的时间间隔
    integrator.integrate(
        current_state, debiased_imu_readings, delta_time_seconds, &next_state,
        &phi, &Q);  // 使用IMU数据对当前状态进行积分

    StateLinearizationPoint state_linearization_point;
    state_linearization_point.timestamp =
        imu_timestamps(0, i + 1);  // 获取IMU数据的时间戳
    state_linearization_point.q_M_I.coeffs() =
        next_state.head<kStateOrientationBlockSize>();  // 获取积分后的四元数
    state_linearization_point.p_M_I = next_state.segment<kPositionBlockSize>(
        kStatePositionOffset);  // 获取积分后的位置
    state_linearization_point.v_M = current_state.segment<kVelocityBlockSize>(
        kStateVelocityOffset);  // 获取积分前的速度
    state_linearization_point.accel_bias =
        current_state.segment<kAccelBiasBlockSize>(
            kStateAccelBiasOffset);  // 获取积分前的加速度偏置
    state_linearization_point.gyro_bias =
        current_state.segment<kGyroBiasBlockSize>(
            kStateGyroBiasOffset);  // 获取积分前的陀螺仪偏置
    state_buffer->addValue(
        state_linearization_point.timestamp,
        state_linearization_point);  // 添加到状态缓冲区

    current_state = next_state;
  }
}

void PoseInterpolator::getVertexToTimeStampMap(
    const vi_map::VIMap& map, const vi_map::MissionId& mission_id,
    VertexToTimeStampMap* vertex_to_time_map, int64_t* min_timestamp_ns,
    int64_t* max_timestamp_ns)
    const {  // 获取顶点到时间戳的映射,输入为地图--map,任务--mission_id,顶点到时间戳的映射--vertex_to_time_map,最小时间戳--min_timestamp_ns,最大时间戳--max_timestamp_ns
  CHECK_NOTNULL(vertex_to_time_map)->clear();

  if (min_timestamp_ns != nullptr) {
    *min_timestamp_ns = std::numeric_limits<int64_t>::max();  // 最小时间戳
  }
  if (max_timestamp_ns != nullptr) {
    *max_timestamp_ns = std::numeric_limits<int64_t>::min();  // 最大时间戳
  }

  // Get the outgoing edge of the vertex and its IMU data.
  pose_graph::VertexIdList all_mission_vertices;
  map.getAllVertexIdsInMissionAlongGraph(
      mission_id, &all_mission_vertices);  // 获取任务中的所有顶点

  pose_graph::Edge::EdgeType backbone_type =
      map.getGraphTraversalEdgeType(mission_id);  // 获取任务中的边类型
  switch (backbone_type) {
    case pose_graph::Edge::EdgeType::kViwls:  // 如果是视觉里程计
      // Everything ok.
      break;
    case pose_graph::Edge::EdgeType::kOdometry:  // 如果是里程计
      LOG(ERROR) << "Interpolation of poses in between vertices is not "
                 << "implemented for odometry type pose graph edges!";
      return;
    default:
      LOG(ERROR) << "Cannot interpolate poses in between vertices based on "
                 << "this posegraph edge type: "
                 << static_cast<int>(backbone_type);
      return;
  }

  vertex_to_time_map->reserve(
      all_mission_vertices.size());  // 为顶点到时间戳的映射分配空间
  for (const pose_graph::VertexId& vertex_id :
       all_mission_vertices) {  // 遍历所有顶点
    const vi_map::Vertex& vertex = map.getVertex(vertex_id);  // 获取顶点
    pose_graph::EdgeIdSet outgoing_edges;      // 获取顶点的出边
    vertex.getOutgoingEdges(&outgoing_edges);  // 获取顶点的出边
    pose_graph::EdgeId outgoing_imu_edge_id;
    for (const pose_graph::EdgeId& edge_id :
         outgoing_edges) {  // 遍历顶点的出边
      if (map.getEdgeType(edge_id) ==
          backbone_type) {  // 如果边类型与任务中的边类型相同
        outgoing_imu_edge_id = edge_id;
        break;
      }
    }
    // We must have reached the end of the graph.
    if (!outgoing_imu_edge_id.isValid()) {  // 如果没有找到边
      break;
    }

    switch (backbone_type) {
      case pose_graph::Edge::EdgeType::kViwls: {  // 如果是视觉里程计
        const vi_map::ViwlsEdge& imu_edge = map.getEdgeAs<vi_map::ViwlsEdge>(
            outgoing_imu_edge_id);  // 获取视觉里程计边
        const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps =
            imu_edge.getImuTimestamps();  // 获取视觉里程计边的IMU时间戳
        if (imu_timestamps.cols() > 0) {
          (*vertex_to_time_map)[vertex_id] =
              imu_timestamps(0, 0);  // 顶点到时间戳的映射
          if (min_timestamp_ns != nullptr) {
            *min_timestamp_ns = std::min(
                *min_timestamp_ns, imu_timestamps(0, 0));  // 最小时间戳
          }
          if (max_timestamp_ns != nullptr) {
            *max_timestamp_ns =
                std::max(*max_timestamp_ns, imu_timestamps(0, 0));
          }
        }
      }
        continue;
      case pose_graph::Edge::EdgeType::kOdometry: {
        LOG(FATAL) << "Should not have reached this!";
      }
        continue;
      default: {
        LOG(FATAL) << "Cannot interpolate poses in between vertices based on "
                   << "this posegraph edge type: "
                   << static_cast<int>(backbone_type);
      }
        continue;
    }
  }
}

void PoseInterpolator::getVertexTimeStampVector(
    const vi_map::VIMap& map, const vi_map::MissionId& mission_id,
    std::vector<int64_t>* vertex_timestamps_nanoseconds)
    const {  // 获取顶点时间戳向量
  CHECK_NOTNULL(vertex_timestamps_nanoseconds)->clear();

  VLOG(1) << "Extract time stamps for mission.";
  pose_graph::VertexIdList all_mission_vertices;
  map.getAllVertexIdsInMissionAlongGraph(mission_id, &all_mission_vertices);
  vertex_timestamps_nanoseconds->reserve(all_mission_vertices.size());

  for (const pose_graph::VertexId& vertex_id :
       all_mission_vertices) {  // 遍历所有顶点
    const int64_t vertex_timestamp =
        map.getVertex(vertex_id).getMinTimestampNanoseconds();
    vertex_timestamps_nanoseconds->emplace_back(
        vertex_timestamp);  // 顶点时间戳向量
  }
}

void PoseInterpolator::buildVertexToTimeList(
    const vi_map::VIMap& map, const vi_map::MissionId& mission_id,
    std::vector<VertexInformation>* vertices_and_time)
    const {  // 构建顶点到时间戳的映射
  CHECK_NOTNULL(vertices_and_time)->clear();
  // Get the outgoing edge of the vertex and its IMU data.
  pose_graph::VertexIdList all_mission_vertices;
  map.getAllVertexIdsInMissionAlongGraph(
      mission_id, &all_mission_vertices);  // 获取所有顶点

  vertices_and_time->reserve(
      all_mission_vertices.size());  // 顶点到时间戳的映射
  for (const pose_graph::VertexId& vertex_id : all_mission_vertices) {
    const vi_map::Vertex& vertex = map.getVertex(vertex_id);  // 获取顶点
    pose_graph::EdgeIdSet outgoing_edges;
    vertex.getOutgoingEdges(&outgoing_edges);  // 获取顶点的出边
    pose_graph::EdgeId outgoing_imu_edge_id;
    for (const pose_graph::EdgeId& edge_id : outgoing_edges) {  // 遍历出边
      if (map.getEdgeType(edge_id) ==
          pose_graph::Edge::EdgeType::kViwls) {  // 如果是视觉里程计
        outgoing_imu_edge_id = edge_id;          // 获取视觉里程计边
        break;
      }
    }
    // We must have reached the end of the graph.
    if (!outgoing_imu_edge_id.isValid()) {
      break;
    }
    const vi_map::ViwlsEdge& imu_edge = map.getEdgeAs<vi_map::ViwlsEdge>(
        outgoing_imu_edge_id);  // 获取视觉里程计边
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps =
        imu_edge.getImuTimestamps();  // 获取视觉里程计边的IMU时间戳
    if (imu_timestamps.cols() > 0) {
      VertexInformation vertex_information;  // 顶点到时间戳的映射
      vertex_information.timestamp_ns = imu_timestamps(0, 0);  // 顶点时间戳
      vertex_information.timestamp_ns_end =
          imu_timestamps(0, imu_timestamps.cols() - 1);  // 顶点时间戳
      vertex_information.vertex_id = vertex_id;
      vertex_information.outgoing_imu_edge_id =
          outgoing_imu_edge_id;  // 视觉里程计边
      vertices_and_time->push_back(vertex_information);
    }
  }
}

void PoseInterpolator::getImuDataInRange(
    const vi_map::VIMap& map, pose_graph::EdgeId imu_edge_id,
    const std::vector<int64_t>& sorted_timestamps, int64_t range_time_start,
    int64_t range_time_end,
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
    Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_data) const {  // 获取IMU数据
  CHECK_NOTNULL(imu_timestamps);
  CHECK_NOTNULL(imu_data);

  CHECK_LE(range_time_start, range_time_end);

  typedef std::vector<int64_t>::const_iterator
      TimestampIterator;  // 时间戳迭代器

  // Search if we have timestamp requests in this range.
  TimestampIterator it_range_start = std::lower_bound(
      sorted_timestamps.begin(), sorted_timestamps.end(), range_time_start,
      std::less<int64_t>());  // 二分查找
  TimestampIterator it_range_end = std::lower_bound(
      sorted_timestamps.begin(), sorted_timestamps.end(), range_time_end,
      std::less<int64_t>());  // 二分查找

  int start_index = -1;
  int end_index = -1;
  if (it_range_start == it_range_end &&
      it_range_start !=
          sorted_timestamps
              .end()) {  // 范围内无请求。Range包含在两个连续时间戳之间的间隔中。
    // No request in range. Range is enclosed in the interval between two
    // consecutive timestamps.
  } else if (
      it_range_start == sorted_timestamps.begin() &&
      it_range_end != sorted_timestamps.end()) {
    // Some requests in range.
    start_index = 0;
    --it_range_end;
    end_index = std::distance(
        sorted_timestamps.cbegin(), it_range_end);  // 获取迭代器距离
  } else if (
      it_range_end == sorted_timestamps.end() &&
      it_range_start != sorted_timestamps.end()) {
    // Some requests in range.
    start_index = std::distance(
        sorted_timestamps.cbegin(), it_range_start);  // 获取迭代器距离
    end_index = sorted_timestamps.size() - 1;
  } else if (
      it_range_end != sorted_timestamps.end() &&
      it_range_start != sorted_timestamps.end()) {
    start_index = std::distance(
        sorted_timestamps.cbegin(), it_range_start);  // 获取迭代器距离
    --it_range_end;                                   // 迭代器减一
    end_index = std::distance(sorted_timestamps.cbegin(), it_range_end);
  }

  // Do we have pose requests in this range?
  if (start_index != -1 && end_index != -1) {
    CHECK_GE(start_index, 0);
    CHECK_GE(end_index, 0);
    CHECK_LT(start_index, static_cast<int>(sorted_timestamps.size()));
    CHECK_LT(end_index, static_cast<int>(sorted_timestamps.size()));

    CHECK_GE(sorted_timestamps[start_index], range_time_start)
        << start_index << "<->" << end_index;
    CHECK_LE(sorted_timestamps[end_index], range_time_end)
        << start_index << "<->" << end_index;

    buildListOfAllRequiredIMUMeasurements(
        map, sorted_timestamps, imu_edge_id, start_index, end_index,
        imu_timestamps, imu_data);  // 获取IMU数据
  } else {
    // Both indices should be set to -1.
    CHECK_EQ(start_index, end_index);
  }
}

void PoseInterpolator::getPosesAtTime(
    const vi_map::VIMap& map, vi_map::MissionId mission_id,
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& pose_timestamps,
    aslam::TransformationVector* poses_M_I,
    std::vector<Eigen::Vector3d>* velocities_M_I,
    std::vector<Eigen::Vector3d>* gyro_biases,
    std::vector<Eigen::Vector3d>* accel_biases) const {  // 获取位姿,主要函数
  CHECK_NOTNULL(poses_M_I)->clear();
  CHECK_GT(pose_timestamps.rows(), 0);

  CHECK(
      map.getGraphTraversalEdgeType(mission_id) ==
      pose_graph::Edge::EdgeType::kViwls);

  // Remember the initial ordering of the timestamps before we sort them.
  std::vector<int64_t> timestamps;
  timestamps.reserve(pose_timestamps.rows());  // 保留空间
  for (int i = 0; i < pose_timestamps.size(); ++i) {
    timestamps.emplace_back(pose_timestamps(0, i));
  }

  std::sort(timestamps.begin(), timestamps.end(), std::less<int64_t>());

  // Build up a list of vertex-information and the timestamps of the first imu
  // measurement on a vertex' outgoing IMU edge.
  std::vector<VertexInformation> vertices_and_time;            // 顶点信息
  buildVertexToTimeList(map, mission_id, &vertices_and_time);  // 获取顶点信息
  CHECK_GT(vertices_and_time.size(), 1u)
      << "The Viwls edges of mission " << mission_id
      << " include none at all or only a single IMU "
      << "measurement. Interpolation is not possible!";

  int64_t smallest_time = vertices_and_time.front().timestamp_ns;
  int64_t largest_time = vertices_and_time.back().timestamp_ns_end;
  CHECK_GE(timestamps.front(), smallest_time)
      << "Requested sample out of bounds! First available time is "
      << smallest_time << " but " << timestamps.front() << " was requested.";
  CHECK_LE(timestamps.back(), largest_time)
      << "Requested sample out of bounds! Last available time is "
      << largest_time << " but " << timestamps.back() << " was requested.";
  VLOGF(4) << "Interpolation range is valid: (" << timestamps.front()
           << " >= " << smallest_time << ") and (" << timestamps.back()
           << " <= " << largest_time << ")";

  std::vector<VertexInformation>::const_iterator vertex_it =
      vertices_and_time.begin();

  const vi_map::VIMission& mission = map.getMission(mission_id);  // 获取任务

  StateBuffer state_buffer;

  while (vertex_it != vertices_and_time.end()) {
    int64_t range_time_start = vertex_it->timestamp_ns;    // 获取时间戳
    int64_t range_time_end = vertex_it->timestamp_ns_end;  // 获取时间戳

    const pose_graph::EdgeId& imu_edge_id =
        vertex_it->outgoing_imu_edge_id;  // 获取边ID
    const pose_graph::VertexId& vertex_id = vertex_it->vertex_id;  // 获取顶点ID

    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps;
    Eigen::Matrix<double, 6, Eigen::Dynamic> imu_data;

    getImuDataInRange(
        map, imu_edge_id, timestamps, range_time_start, range_time_end,
        &imu_timestamps, &imu_data);  // 获取IMU数据

    computeRequestedPosesInRange(
        map, mission, vertex_id, imu_edge_id, imu_timestamps, imu_data,
        &state_buffer);  // 计算位姿

    ++vertex_it;
  }

  // Copy the interpolated data to the output buffer.
  poses_M_I->clear();
  poses_M_I->reserve(pose_timestamps.cols());
  for (int i = 0; i < pose_timestamps.cols(); ++i) {
    StateLinearizationPoint state_linearization_point;
    const bool buffer_has_state = state_buffer.getValueAtTime(
        pose_timestamps(0, i), &state_linearization_point);  // 获取位姿
    CHECK(buffer_has_state)
        << ": No value in state_buffer at time: " << pose_timestamps(0, i);
    poses_M_I->emplace_back(
        state_linearization_point.q_M_I,
        state_linearization_point.p_M_I);  // 位姿压入
    if (velocities_M_I) {
      velocities_M_I->emplace_back(state_linearization_point.v_M);  // 速度压入
    }
    if (accel_biases) {
      accel_biases->emplace_back(
          state_linearization_point.accel_bias);  // 加速度压入
    }
    if (gyro_biases) {
      gyro_biases->emplace_back(
          state_linearization_point.gyro_bias);  // 陀螺仪压入
    }
  }
}

void PoseInterpolator::getMissionTimeRange(
    const vi_map::VIMap& vi_map, const vi_map::MissionId mission_id,
    int64_t* mission_start_ns_ptr,
    int64_t* mission_end_ns_ptr) const {  // 获取任务时间范围
  *CHECK_NOTNULL(mission_start_ns_ptr) = -1;
  *CHECK_NOTNULL(mission_end_ns_ptr) = -1;
  pose_graph::VertexIdList vertex_ids;  // 顶点ID
  pose_graph::EdgeIdSet edge_ids;       // 边ID
  vi_map.getAllVertexIdsInMissionAlongGraph(
      mission_id, &vertex_ids);  // 获取任务顶点ID
  CHECK(!vertex_ids.empty());

  // Interpolation is based on IMU measurements, so select the first IMU
  // timestamp along the viwls edge leaving the root vertex.
  const vi_map::Vertex& root_vertex =
      vi_map.getVertex(vertex_ids.front());  // 获取顶点
  VLOGF(2) << "Earliest timestamp on root vertex: "
           << root_vertex.getVisualNFrame()
                  .getMinTimestampNanoseconds();  // 获取时间戳
  root_vertex.getOutgoingEdges(&edge_ids);
  for (const pose_graph::EdgeId& edge_id : edge_ids) {
    if (vi_map.getEdgeType(edge_id) ==
        pose_graph::Edge::EdgeType::kViwls) {  // 获取边类型,如果是视觉边
      *mission_start_ns_ptr =
          vi_map.getEdgeAs<vi_map::ViwlsEdge>(edge_id).getImuTimestamps()(
              0, 0);  // 获取时间戳
      break;
    }
  }
  CHECK_GE(*mission_start_ns_ptr, 0)
      << "Did not find timestamp on viwls edge leaving root vertex.";
  VLOGF(2) << "First IMU timestamp after root: " << *mission_start_ns_ptr;

  // Likewise, use the last IMU measurement before the final vertex.
  const vi_map::Vertex& last_vertex =
      vi_map.getVertex(vertex_ids.back());  // 获取顶点
  VLOGF(2) << "Latest timestamp on final vertex: "
           << last_vertex.getVisualNFrame().getMaxTimestampNanoseconds();
  last_vertex.getIncomingEdges(&edge_ids);  // 获取边
  for (const pose_graph::EdgeId& edge_id : edge_ids) {
    if (vi_map.getEdgeType(edge_id) == pose_graph::Edge::EdgeType::kViwls) {
      *mission_end_ns_ptr =
          vi_map.getEdgeAs<vi_map::ViwlsEdge>(edge_id)
              .getImuTimestamps()
              .topRightCorner<1, 1>()(0, 0);  // 获取视觉end时间戳
      break;
    }
  }
  CHECK_GE(*mission_end_ns_ptr, 0)
      << "Did not find timestamp on viwls edge entering final vertex.";
  CHECK_LT(*mission_start_ns_ptr, *mission_end_ns_ptr);
}

void PoseInterpolator::getPosesEveryNSeconds(
    const vi_map::VIMap& vi_map, const vi_map::MissionId mission_id,
    const double timestep_seconds,
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* pose_times,
    aslam::TransformationVector* poses) const {
  CHECK_GT(timestep_seconds, 0) << "Interpolation timestep must be positive.";
  CHECK_NOTNULL(pose_times);
  CHECK_NOTNULL(poses);
  int64_t mission_start_ns = -1;
  int64_t mission_end_ns = -1;
  getMissionTimeRange(vi_map, mission_id, &mission_start_ns, &mission_end_ns);

  const int64_t timestep_ns =
      aslam::time::secondsToNanoSeconds(timestep_seconds);
  CHECK_GT(timestep_ns, 0);
  const int64_t pose_count =
      ((mission_end_ns - mission_start_ns) / timestep_ns) + 1;
  VLOGF(1) << "Interpolating between " << mission_start_ns << " and "
           << mission_end_ns;
  VLOGF(1) << "Interpolating " << pose_count << " poses, last will be "
           << mission_start_ns + (pose_count - 1) * timestep_ns;

  pose_times->resize(Eigen::NoChange, pose_count);
  for (int64_t i = 0; i < pose_count; ++i) {
    (*pose_times)(0, i) = mission_start_ns + i * timestep_ns;
  }
  CHECK_LE((*pose_times)(0, pose_count - 1), mission_end_ns);
  return getPosesAtTime(vi_map, mission_id, *pose_times, poses);
}

}  // namespace landmark_triangulation
