#include "maplab-node/datasource-rostopic.h"

#include <aslam/common/time.h>
#include <boost/bind.hpp>
#include <map-resources/resource-conversion.h>
#include <maplab-common/accessors.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensors/lidar.h>
#include <string>
#include <vi-map/sensor-utils.h>
#include <vio-common/rostopic-settings.h>

#include "maplab-node/ros-helpers.h"

DECLARE_bool(zero_initial_timestamps);

namespace maplab {

DataSourceRostopic::DataSourceRostopic(
    const vio_common::RosTopicSettings& ros_topics,
    const vi_map::SensorManager& sensor_manager)
    : DataSource(sensor_manager),
      shutdown_requested_(false),
      ros_topics_(ros_topics),
      image_transport_(node_handle_),
      last_imu_timestamp_ns_(aslam::time::getInvalidTime()),
      last_imu_dispatch_timestamp_ns_(aslam::time::getInvalidTime()),
      imu_batch_period_ns_(
          1e9 / FLAGS_maplab_batch_imu_measurements_at_frequency),
      last_wheel_odometry_timestamp_ns_(aslam::time::getInvalidTime()),
      last_odometry_timestamp_ns_(aslam::time::getInvalidTime()),
      odometry_min_period_ns_(1e9 / FLAGS_maplab_throttle_frequency_odometry) {
  const uint8_t num_cameras = ros_topics_.camera_topic_cam_index_map.size();
  if (num_cameras > 0u) {
    last_image_timestamp_ns_.resize(num_cameras, aslam::time::getInvalidTime());
  }

  CHECK_GT(odometry_min_period_ns_, 0);
}

DataSourceRostopic::~DataSourceRostopic() {}

void DataSourceRostopic::startStreaming() {
  registerSubscribers(ros_topics_);
}

void DataSourceRostopic::shutdown() {
  shutdown_requested_ = true;
}

void DataSourceRostopic::registerSubscribers(
    const vio_common::RosTopicSettings& ros_topics) {
  // Camera subscriber.
  const size_t num_cameras = ros_topics.camera_topic_cam_index_map.size();
  sub_images_.reserve(num_cameras);

  for (const std::pair<const std::string, size_t>& topic_camidx :
       ros_topics.camera_topic_cam_index_map) {
    CHECK(!topic_camidx.first.empty()) << "Camera " << topic_camidx.second
                                       << " is subscribed to an empty topic!";

    boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback =
        boost::bind(
            &DataSourceRostopic::imageCallback, this, _1, topic_camidx.second);

    constexpr size_t kRosSubscriberQueueSizeImage = 20u;
    image_transport::Subscriber image_sub = image_transport_.subscribe(
        topic_camidx.first, kRosSubscriberQueueSizeImage, image_callback);
    sub_images_.push_back(image_sub);
    VLOG(1) << "[MaplabNode-DataSource] Camera " << topic_camidx.second
            << " is subscribed to topic: '" << topic_camidx.first << "'";
  }

  // IMU subscriber.
  CHECK(!ros_topics.imu_topic.empty())
      << "IMU is subscribed to an empty topic!";
  constexpr size_t kRosSubscriberQueueSizeImu = 1000u;
  boost::function<void(const sensor_msgs::ImuConstPtr&)> imu_callback =
      boost::bind(&DataSourceRostopic::imuMeasurementCallback, this, _1);
  sub_imu_ = node_handle_.subscribe(
      ros_topics.imu_topic, kRosSubscriberQueueSizeImu, imu_callback);

  VLOG(1) << "[MaplabNode-DataSource] IMU is subscribed to topic: '"
          << ros_topics.imu_topic << "'";

  // Lidar subscriber.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.lidar_topic_sensor_id_map) {
    CHECK(topic_sensorid.second.isValid())
        << "The ROS-topic to Lidar sensor id association contains an invalid "
        << "sensor id! topic: " << topic_sensorid.first;
    CHECK(!topic_sensorid.first.empty())
        << "Lidar(" << topic_sensorid.second
        << ") is subscribed to an empty topic!";
    boost::function<void(const sensor_msgs::PointCloud2ConstPtr&)>
        lidar_callback = boost::bind(
            &DataSourceRostopic::lidarMeasurementCallback, this, _1,
            topic_sensorid.second);
    constexpr size_t kRosSubscriberQueueSizeLidar = 20u;

    sub_lidars_.emplace_back(node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeLidar, lidar_callback));

    VLOG(1) << "[MaplabNode-DataSource] Lidar(" << topic_sensorid.second
            << ") is subscribed to topic: '" << topic_sensorid.first << "'";
  }

  // Odometry subscriber.
  const std::pair<std::string, aslam::SensorId>& topic_sensorid =
      ros_topics.odometry_6dof_topic;
  if (!topic_sensorid.first.empty()) {
    CHECK(topic_sensorid.second.isValid());
    constexpr size_t kRosSubscriberQueueSizeOdometry = 1000u;
    boost::function<void(const maplab_msgs::OdometryWithImuBiasesConstPtr&)>
        odometry_callback = boost::bind(
            &DataSourceRostopic::odometryEstimateCallback, this, _1,
            topic_sensorid.second);
    sub_odom_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeOdometry,
        odometry_callback);

    VLOG(1) << "[MaplabNode-DataSource] External odometry sensor with id "
            << topic_sensorid.second << " is "
            << "subscribed to topic: '" << topic_sensorid.first << "'";
  } else {
    LOG(FATAL) << "[MaplabNode-DataSource] Subscribing to the odometry sensor "
               << "failed, because the topic is empty!";
  }

  // Absolute pose constraint subscribers.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.absolute_6dof_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeAbsoluteConstraints = 1000u;
    boost::function<void(
        const geometry_msgs::PoseWithCovarianceStampedConstPtr&)>
        absolute_callback = boost::bind(
            &DataSourceRostopic::absolute6DoFConstraintCallback, this, _1,
            topic_sensorid.second);
    sub_absolute_6dof_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeAbsoluteConstraints,
        absolute_callback);

    VLOG(1)
        << "[MaplabNode-DataSource] External absolute 6DoF pose sensor with id "
        << topic_sensorid.second << " is "
        << "subscribed to topic: '" << topic_sensorid.first << "'";
  }

  // Wheel odometry constraint subscribers.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.wheel_odometry_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeWheelOdometryConstraints = 1000u;
    boost::function<void(const nav_msgs::OdometryConstPtr&)>
        wheel_odometry_callback = boost::bind(
            &DataSourceRostopic::wheelOdometryConstraintCallback, this, _1,
            topic_sensorid.second);
    sub_wheel_odometry_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeWheelOdometryConstraints,
        wheel_odometry_callback);

    VLOG(1) << "[MaplabNode-DataSource] External wheel odometry "
            << "sensor with id " << topic_sensorid.second
            << " is subscribed to topic: '" << topic_sensorid.first << "'";
  }

  // External features subscribers.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.external_features_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeExternalFeatures = 1000u;

    boost::function<void(const maplab_msgs::FeaturesConstPtr&)>
        external_features_callback = boost::bind(
            &DataSourceRostopic::externalFeaturesCallback, this, _1,
            topic_sensorid.second);

    sub_external_features_.emplace_back(node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeExternalFeatures,
        external_features_callback));

    VLOG(1) << "[MaplabNode-DataSource] External features "
            << "sensor with id " << topic_sensorid.second
            << " is subscribed to topic: '" << topic_sensorid.first << "'";
  }

#ifdef VOXGRAPH
  // Loop closure constraint subscribers.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.loop_closure_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeLoopClosureConstraints = 1000u;
    boost::function<void(const voxgraph_msgs::LoopClosureEdgeListConstPtr&)>
        loop_closure_callback = boost::bind(
            &DataSourceRostopic::voxgraphLoopClosureConstraintCallback, this,
            _1, topic_sensorid.second);
    sub_loop_closure_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeLoopClosureConstraints,
        loop_closure_callback);

    VLOG(1) << "[MaplabNode-DataSource] External loop closure constraint "
            << "sensor with id " << topic_sensorid.second
            << " is subscribed to topic: '" << topic_sensorid.first << "'";
  }
#endif  // VOXGRAPH

#ifdef VOXGRAPH
  // Point cloud submap subscriber, based on voxgraph types.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.pointcloud_map_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeRelativeConstraints = 1000u;
    boost::function<void(const voxgraph_msgs::MapSurfaceConstPtr&)>
        submap_callback = boost::bind(
            &DataSourceRostopic::voxgraphPointCloudMapCallback, this, _1,
            topic_sensorid.second);
    sub_pointcloud_map_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeRelativeConstraints,
        submap_callback);

    VLOG(1) << "[MaplabNode-DataSource] External point cloud (sub-)map "
            << "sensor with id " << topic_sensorid.second
            << " is subscribed to topic: '" << topic_sensorid.first << "'";
  }
#else
  // Point cloud submap subscriber, based on PointCloud2 types.
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.pointcloud_map_topic_map) {
    constexpr size_t kRosSubscriberQueueSizeRelativeConstraints = 1000u;
    boost::function<void(const sensor_msgs::PointCloud2ConstPtr&)>
        submap_callback = boost::bind(
            &DataSourceRostopic::pointCloudMapCallback, this, _1,
            topic_sensorid.second);
    sub_pointcloud_map_ = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeRelativeConstraints,
        submap_callback);

    VLOG(1) << "[MaplabNode-DataSource] External point cloud (sub-)map "
            << "sensor with id " << topic_sensorid.second
            << " is subscribed to topic: '" << topic_sensorid.first << "'";
  }
#endif  // VOXGRAPH
}

void DataSourceRostopic::imageCallback(
    const sensor_msgs::ImageConstPtr& image_message, size_t camera_idx) {
  if (shutdown_requested_) {
    return;
  }

  vio::ImageMeasurement::Ptr image_measurement =
      convertRosImageToMaplabImage(image_message, camera_idx);
  CHECK(image_measurement);

  // Apply the IMU to camera time shift.
  if (FLAGS_imu_to_camera_time_offset_ns != 0) {
    image_measurement->timestamp += FLAGS_imu_to_camera_time_offset_ns;
  }

  // Check for strictly increasing image timestamps.
  CHECK_LT(camera_idx, last_image_timestamp_ns_.size());
  if (aslam::time::isValidTime(last_image_timestamp_ns_[camera_idx]) &&
      last_image_timestamp_ns_[camera_idx] >= image_measurement->timestamp) {
    LOG(WARNING) << "[MaplabNode-DataSource] Image message (cam " << camera_idx
                 << ") is not strictly "
                 << "increasing! Current timestamp: "
                 << image_measurement->timestamp << "ns vs last timestamp: "
                 << last_image_timestamp_ns_[camera_idx] << "ns.";
    return;
  } else {
    last_image_timestamp_ns_[camera_idx] = image_measurement->timestamp;
  }

  invokeImageCallbacks(image_measurement);
}

void DataSourceRostopic::imuMeasurementCallback(
    const sensor_msgs::ImuConstPtr& msg) {
  if (shutdown_requested_) {
    return;
  }

  const int64_t timestamp_ns = rosTimeToNanoseconds(msg->header.stamp);

  // Check for strictly increasing imu timestamps.
  if (aslam::time::isValidTime(last_imu_timestamp_ns_) &&
      last_imu_timestamp_ns_ >= timestamp_ns) {
    LOG(WARNING) << "[MaplabNode-DataSource] IMU message is not strictly "
                 << "increasing! Current timestamp: " << timestamp_ns
                 << "ns vs last timestamp: " << last_imu_timestamp_ns_ << "ns.";
    return;
  }
  // This IMU measurement was accepted.
  last_imu_timestamp_ns_ = timestamp_ns;

  // Initialize the dispatch timer, if we get here for the first time.
  if (!aslam::time::isValidTime(last_imu_dispatch_timestamp_ns_)) {
    last_imu_dispatch_timestamp_ns_ = timestamp_ns;
  }
  // Initialize a new batch if this is the first time or if in the previous
  // call we just released a batch.
  if (!current_imu_batch_) {
    current_imu_batch_.reset(new vio::BatchedImuMeasurements);
  }
  CHECK(current_imu_batch_);

  addRosImuMeasurementToImuMeasurementBatch(*msg, current_imu_batch_.get());

  // To batch or not to batch.
  if (timestamp_ns - last_imu_dispatch_timestamp_ns_ > imu_batch_period_ns_) {
    // Should release the current batch and initialize a new one.
    vio::BatchedImuMeasurements::ConstPtr const_batch_ptr =
        std::const_pointer_cast<const vio::BatchedImuMeasurements>(
            current_imu_batch_);
    invokeImuCallbacks(const_batch_ptr);

    // Reset current batch.
    current_imu_batch_.reset();

    // Update time of last dispatch.
    last_imu_dispatch_timestamp_ns_ = timestamp_ns;
  }
}

void DataSourceRostopic::lidarMeasurementCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  vi_map::RosLidarMeasurement::Ptr lidar_measurement =
      convertRosCloudToMaplabCloud(msg, sensor_id);
  CHECK(lidar_measurement);

  // Apply the IMU to lidar time shift.
  if (FLAGS_imu_to_lidar_time_offset_ns != 0) {
    *lidar_measurement->getTimestampNanosecondsMutable() +=
        FLAGS_imu_to_lidar_time_offset_ns;
  }

  invokeLidarCallbacks(lidar_measurement);
}

void DataSourceRostopic::odometryEstimateCallback(
    const maplab_msgs::OdometryWithImuBiasesConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  static constexpr int64_t kAcceptanceTime = -10000;
  const int64_t timestamp_ns = rosTimeToNanoseconds(msg->header.stamp);
  if (aslam::time::isValidTime(last_odometry_timestamp_ns_)) {
    const int64_t odometry_period_ns =
        timestamp_ns - last_odometry_timestamp_ns_;
    if (odometry_period_ns <= 0) {
      LOG(WARNING)
          << "[MaplabNode-DataSource] Odometry message is not strictly "
          << "increasing! Current timestamp: " << timestamp_ns
          << "ns vs last timestamp: " << last_odometry_timestamp_ns_ << "ns.";
      return;
    } else {
      const int64_t odometry_diff_ns =
          odometry_period_ns - odometry_min_period_ns_;
      if (odometry_diff_ns < 0 && odometry_diff_ns < kAcceptanceTime) {
        // Skip this odometry message, since it arrives at a higher
        // frequency than desired.
        return;
      }
    }
  }

  // This odometry measurement was accepted.
  last_odometry_timestamp_ns_ = timestamp_ns;

  // Convert message.
  const vi_map::Odometry6DoF& sensor =
      sensor_manager_.getSensor<vi_map::Odometry6DoF>(sensor_id);
  const aslam::Transformation& T_B_S =
      sensor_manager_.getSensor_T_B_S(sensor_id);
  maplab::OdometryEstimate::Ptr odometry_measurement =
      convertRosOdometryMsgToOdometryEstimate(msg, T_B_S, sensor);
  CHECK(odometry_measurement);

  invokeOdometryCallbacks(odometry_measurement);
}

void DataSourceRostopic::absolute6DoFConstraintCallback(
    const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  const vi_map::Absolute6DoF& sensor =
      sensor_manager_.getSensor<vi_map::Absolute6DoF>(sensor_id);
  vi_map::Absolute6DoFMeasurement::Ptr absolute_constraint =
      convertPoseWithCovarianceToAbsolute6DoFConstraint(msg, sensor);
  if (!absolute_constraint) {
    LOG(ERROR) << "Received invalid Absolute6DoF constraint!";
    return;
  }

  invokeAbsolute6DoFConstraintCallbacks(absolute_constraint);
}

#ifdef VOXGRAPH
void DataSourceRostopic::voxgraphLoopClosureConstraintCallback(
    const voxgraph_msgs::LoopClosureEdgeListConstPtr& lc_edges_msg,
    const aslam::SensorId& sensor_id) {
  CHECK(lc_edges_msg);
  if (shutdown_requested_) {
    return;
  }

  const vi_map::LoopClosureSensor& sensor =
      sensor_manager_.getSensor<vi_map::LoopClosureSensor>(sensor_id);

  std::vector<vi_map::LoopClosureMeasurement::Ptr> lc_edges;
  convertVoxgraphEdgeListToLoopClosureConstraint(
      lc_edges_msg, sensor, &lc_edges);

  VLOG(3) << "[DataSourceRostopic] Received a list of " << lc_edges.size()
          << " loop closure constraints.";

  for (const vi_map::LoopClosureMeasurement::Ptr& lc_edge : lc_edges) {
    CHECK(lc_edge);
    invokeLoopClosureConstraintCallbacks(lc_edge);
  }
}
#endif  // VOXGRAPH

#ifdef VOXGRAPH
void DataSourceRostopic::voxgraphPointCloudMapCallback(
    const voxgraph_msgs::MapSurfaceConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  vi_map::RosPointCloudMapSensorMeasurement::Ptr pointcloud_map =
      convertVoxgraphMapToPointCloudMap(msg, sensor_id);
  CHECK(pointcloud_map);

  invokePointCloudMapCallbacks(pointcloud_map);
}
#else
void DataSourceRostopic::pointCloudMapCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  vi_map::RosPointCloudMapSensorMeasurement::Ptr pointcloud_map =
      convertRosPointCloudToPointCloudMap(msg, sensor_id);
  CHECK(pointcloud_map);

  invokePointCloudMapCallbacks(pointcloud_map);
}
#endif  // VOXGRAPH

void DataSourceRostopic::wheelOdometryConstraintCallback(
    const nav_msgs::OdometryConstPtr& wheel_odometry_msg,
    const aslam::SensorId& sensor_id) {
  CHECK(wheel_odometry_msg);
  if (shutdown_requested_) {
    return;
  }

  vi_map::WheelOdometryMeasurement::Ptr wheel_odometry_measurement;
  wheel_odometry_measurement =
      convertRosOdometryToMaplabWheelOdometry(wheel_odometry_msg, sensor_id);
  CHECK(wheel_odometry_measurement);

  // Check for strictly increasing wheel odometry timestamps.
  if (aslam::time::isValidTime(last_wheel_odometry_timestamp_ns_) &&
      last_wheel_odometry_timestamp_ns_ >=
          wheel_odometry_measurement->getTimestampNanoseconds()) {
    LOG(WARNING) << "[MaplabNode-DataSource] Wheel odometry message is "
                 << "not strictly increasing! Current timestamp: "
                 << wheel_odometry_measurement->getTimestampNanoseconds()
                 << "ns vs last timestamp: "
                 << last_wheel_odometry_timestamp_ns_ << "ns.";
    return;
  } else {
    last_wheel_odometry_timestamp_ns_ =
        wheel_odometry_measurement->getTimestampNanoseconds();
  }

  invokeWheelOdometryConstraintCallbacks(wheel_odometry_measurement);
}

void DataSourceRostopic::externalFeaturesCallback(
    const maplab_msgs::FeaturesConstPtr& msg,
    const aslam::SensorId& sensor_id) {
  CHECK(msg);
  if (shutdown_requested_) {
    return;
  }

  vi_map::ExternalFeaturesMeasurement::Ptr external_features_measurement;
  external_features_measurement =
      convertRosFeatureMsgToMaplabExternalFeatures(msg, sensor_id);

  // Apply the IMU to camera time shift.
  if (FLAGS_imu_to_camera_time_offset_ns != 0) {
    *(external_features_measurement->getTimestampNanosecondsMutable()) +=
        FLAGS_imu_to_camera_time_offset_ns;
  }

  invokeExternalFeaturesCallbacks(external_features_measurement);
}

}  // namespace maplab
