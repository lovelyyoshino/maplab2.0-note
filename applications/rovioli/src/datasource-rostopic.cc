#include "rovioli/datasource-rostopic.h"

#include <string>

#include <aslam/common/time.h>
#include <boost/bind.hpp>
#include <maplab-common/accessors.h>
#include <vio-common/rostopic-settings.h>

#include "rovioli/ros-helpers.h"

DECLARE_bool(rovioli_zero_initial_timestamps);

namespace rovioli {

DataSourceRostopic::DataSourceRostopic(
    const vio_common::RosTopicSettings& ros_topics)
    : shutdown_requested_(false),
      ros_topics_(ros_topics),
      image_transport_(node_handle_),
      last_imu_timestamp_ns_(aslam::time::getInvalidTime()),
      last_odometry_timestamp_ns_(aslam::time::getInvalidTime()) {
  const uint8_t num_cameras = ros_topics_.camera_topic_cam_index_map.size();
  if (num_cameras > 0u) {
    last_image_timestamp_ns_.resize(num_cameras, aslam::time::getInvalidTime());
  }
  if (FLAGS_imu_to_camera_time_offset_ns != 0) {
    LOG(WARNING) << "You are applying a time offset between IMU and camera, be "
                 << "aware that this will shift the image timestamps, which "
                 << "means the published pose estimates will now correspond "
                 << "these shifted timestamps!";
  }
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
    boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback =
        boost::bind(
            &DataSourceRostopic::imageCallback, this, _1, topic_camidx.second);

    constexpr size_t kRosSubscriberQueueSizeImage = 20u;
    image_transport::Subscriber image_sub = image_transport_.subscribe(
        topic_camidx.first, kRosSubscriberQueueSizeImage, image_callback);
    sub_images_.push_back(image_sub);
  }

  // IMU subscriber.
  constexpr size_t kRosSubscriberQueueSizeImu = 1000u;
  boost::function<void(const sensor_msgs::ImuConstPtr&)> imu_callback =
      boost::bind(&DataSourceRostopic::imuMeasurementCallback, this, _1);
  sub_imu_ = node_handle_.subscribe(
      ros_topics.imu_topic, kRosSubscriberQueueSizeImu, imu_callback);

  // Wheel odometry subscriber
  constexpr size_t kRosSubscriberQueueSizeWheelOdometry = 1000u;
  for (const std::pair<const std::string, aslam::SensorId>& topic_sensorid :
       ros_topics.wheel_odometry_topic_map) {
    boost::function<void(const nav_msgs::OdometryConstPtr&)>
        wheel_odometry_callback = boost::bind(
            &DataSourceRostopic::odometryMeasurementCallback, this, _1);
    ros::Subscriber sub_wheel_odometry = node_handle_.subscribe(
        topic_sensorid.first, kRosSubscriberQueueSizeWheelOdometry,
        wheel_odometry_callback);
    sub_odometry_.push_back(sub_wheel_odometry);
  }
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

  // Shift timestamps to start at 0.
  if (!FLAGS_rovioli_zero_initial_timestamps ||
      shiftByFirstTimestamp(&(image_measurement->timestamp))) {
    // Check for strictly increasing image timestamps.
    CHECK_LT(camera_idx, last_image_timestamp_ns_.size());
    if (aslam::time::isValidTime(last_image_timestamp_ns_[camera_idx]) &&
        last_image_timestamp_ns_[camera_idx] >= image_measurement->timestamp) {
      LOG(WARNING) << "[ROVIOLI-DataSource] Image message (cam " << camera_idx
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
}

void DataSourceRostopic::imuMeasurementCallback(
    const sensor_msgs::ImuConstPtr& msg) {
  if (shutdown_requested_) {
    return;
  }

  vio::ImuMeasurement::Ptr imu_measurement = convertRosImuToMaplabImu(msg);

  // Shift timestamps to start at 0.
  if (!FLAGS_rovioli_zero_initial_timestamps ||
      shiftByFirstTimestamp(&(imu_measurement->timestamp))) {
    // Check for strictly increasing imu timestamps.
    if (aslam::time::isValidTime(last_imu_timestamp_ns_) &&
        last_imu_timestamp_ns_ >= imu_measurement->timestamp) {
      LOG(WARNING) << "[ROVIOLI-DataSource] IMU message is not strictly "
                   << "increasing! Current timestamp: "
                   << imu_measurement->timestamp
                   << "ns vs last timestamp: " << last_imu_timestamp_ns_
                   << "ns.";
      return;
    } else {
      last_imu_timestamp_ns_ = imu_measurement->timestamp;
    }

    invokeImuCallbacks(imu_measurement);
  }
}

void DataSourceRostopic::odometryMeasurementCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  if (shutdown_requested_) {
    return;
  }

  vio::OdometryMeasurement::Ptr odometry_measurement =
      convertRosOdometryToOdometry(msg);

  // Shift timestamps to start at 0.
  if (!FLAGS_rovioli_zero_initial_timestamps ||
      shiftByFirstTimestamp(&(odometry_measurement->timestamp))) {
    // Check for strictly increasing wheel odometry timestamps.
    if (aslam::time::isValidTime(last_odometry_timestamp_ns_) &&
        last_odometry_timestamp_ns_ >= odometry_measurement->timestamp) {
      LOG(WARNING) << "[MaplabNode-DataSource] Wheel odometry message is "
                   << "not strictly increasing! Current timestamp: "
                   << odometry_measurement->timestamp
                   << "ns vs last timestamp: " << last_odometry_timestamp_ns_
                   << "ns.";
    } else {
      last_odometry_timestamp_ns_ = odometry_measurement->timestamp;
    }

    invokeOdometryCallbacks(odometry_measurement);
  }
}

}  // namespace rovioli
