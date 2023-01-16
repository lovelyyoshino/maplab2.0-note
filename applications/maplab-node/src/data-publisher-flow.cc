#include "maplab-node/data-publisher-flow.h"
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <maplab-common/conversions.h>
#include <maplab-common/file-logger.h>
#include <maplab-common/localization-result.h>
#include <minkindr_conversions/kindr_msg.h>
#include "maplab-node/ros-helpers.h"

DEFINE_double(
    map_publish_interval_s, 2.0,
    "Interval of publishing the visual-inertial map to ROS [seconds].");

DEFINE_bool(
    publish_only_on_keyframes, false,
    "Publish frames only on keyframes instead of the IMU measurements. This "
    "means a lower frequency.");

DEFINE_bool(
    publish_debug_markers, true,
    "Publish debug sphere markers for T_M_B, T_G_I and localization frames.");

DEFINE_string(
    export_estimated_poses_to_csv, "",
    "If not empty, the map builder will export the estimated poses to a CSV "
    "file.");

DEFINE_bool(
    visualize_map, true,
    "Set to false to disable map visualization. Note: map building needs to be "
    "active for the visualization.");

DECLARE_bool(run_map_builder);

#include "maplab-node/vi-map-with-mutex.h"

namespace maplab {
namespace {
inline ros::Time createRosTimestamp(int64_t timestamp_nanoseconds) {
  static constexpr uint32_t kNanosecondsPerSecond = 1e9;
  const uint64_t timestamp_u64 = static_cast<uint64_t>(timestamp_nanoseconds);
  const uint32_t ros_timestamp_sec = timestamp_u64 / kNanosecondsPerSecond;
  const uint32_t ros_timestamp_nsec =
      timestamp_u64 - (ros_timestamp_sec * kNanosecondsPerSecond);
  return ros::Time(ros_timestamp_sec, ros_timestamp_nsec);
}
}  // namespace

DataPublisherFlow::DataPublisherFlow(
    const vi_map::SensorManager& sensor_manager)
    : sensor_manager_(sensor_manager),
      map_publisher_timeout_(common::TimeoutCounter(
          FLAGS_map_publish_interval_s * kSecondsToNanoSeconds)) {
  visualization::RVizVisualizationSink::init();
  plotter_.reset(new visualization::ViwlsGraphRvizPlotter);
}

void DataPublisherFlow::registerPublishers() {
  pub_pose_T_M_B_ =
      node_handle_.advertise<geometry_msgs::PoseStamped>(kTopicPoseMission, 1);
  pub_pose_T_G_I_ =
      node_handle_.advertise<geometry_msgs::PoseStamped>(kTopicPoseGlobal, 1);
  pub_baseframe_T_G_M_ =
      node_handle_.advertise<geometry_msgs::PoseStamped>(kTopicBaseframe, 1);
  pub_velocity_I_ =
      node_handle_.advertise<geometry_msgs::Vector3Stamped>(kTopicVelocity, 1);
  pub_imu_acc_bias_ =
      node_handle_.advertise<geometry_msgs::Vector3Stamped>(kTopicBiasAcc, 1);
  pub_imu_gyro_bias_ =
      node_handle_.advertise<geometry_msgs::Vector3Stamped>(kTopicBiasGyro, 1);
  pub_extrinsics_T_C_Bs_ = node_handle_.advertise<geometry_msgs::PoseArray>(
      kCameraExtrinsicTopic, 1);
  pub_localization_result_T_G_B_ =
      node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
          kTopicLocalizationResult, 1);
  pub_fused_localization_result_T_G_B_ =
      node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
          kTopicLocalizationResultFused, 1);
}

void DataPublisherFlow::attachToMessageFlow(message_flow::MessageFlow* flow) {
  CHECK_NOTNULL(flow);
  registerPublishers();
  static constexpr char kSubscriberNodeName[] = "DataPublisher";

  if (FLAGS_run_map_builder && FLAGS_visualize_map) {
    flow->registerSubscriber<message_flow_topics::RAW_VIMAP>(
        kSubscriberNodeName, message_flow::DeliveryOptions(),
        [this](const VIMapWithMutex::ConstPtr& map_with_mutex) {
          if (map_publisher_timeout_.reached()) {
            std::lock_guard<std::mutex> lock(map_with_mutex->mutex);
            visualizeMap(map_with_mutex->vi_map);
            map_publisher_timeout_.reset();
          }
        });
  }

  // Publish localization results.
  flow->registerSubscriber<message_flow_topics::LOCALIZATION_RESULT>(
      kSubscriberNodeName, message_flow::DeliveryOptions(),
      [this](const common::LocalizationResult::ConstPtr& localization) {
        CHECK(localization);
        localizationCallback(
            *localization, false /* is NOT a fused localization result */);
      });

  // Publish fused localization results.
  flow->registerSubscriber<message_flow_topics::FUSED_LOCALIZATION_RESULT>(
      kSubscriberNodeName, message_flow::DeliveryOptions(),
      [this](const common::LocalizationResult::ConstPtr& fused_localization) {
        CHECK(fused_localization);
        localizationCallback(
            *fused_localization, true /* is a fused localization result */);
      });

  flow->registerSubscriber<message_flow_topics::MAP_UPDATES>(
      kSubscriberNodeName, message_flow::DeliveryOptions(),
      [this](const vio::MapUpdate::ConstPtr& vio_update) {
        CHECK(vio_update != nullptr);
        bool has_T_G_M =
            (vio_update->localization_state ==
                 common::LocalizationState::kLocalized ||
             vio_update->localization_state ==
                 common::LocalizationState::kMapTracking);
        publishVinsState(
            vio_update->timestamp_ns, vio_update->vinode, has_T_G_M,
            vio_update->T_G_M);
      });

  flow->registerSubscriber<message_flow_topics::ODOMETRY_ESTIMATES>(
      kSubscriberNodeName, message_flow::DeliveryOptions(),
      [this](const OdometryEstimate::ConstPtr& state) {
        CHECK(state != nullptr);
        publishOdometryState(state->timestamp_ns, state->vinode);

        // Publish estimated camera-extrinsics.
        geometry_msgs::PoseArray T_C_Bs_message;
        T_C_Bs_message.header.frame_id = FLAGS_tf_imu_frame;
        T_C_Bs_message.header.stamp = createRosTimestamp(state->timestamp_ns);
        for (const auto& cam_idx_T_C_B : state->maplab_camera_index_to_T_C_B) {
          geometry_msgs::Pose T_C_B_message;
          tf::poseKindrToMsg(cam_idx_T_C_B.second, &T_C_B_message);
          T_C_Bs_message.poses.emplace_back(T_C_B_message);
        }
        pub_extrinsics_T_C_Bs_.publish(T_C_Bs_message);
      });

  // CSV export for end-to-end test.
  if (!FLAGS_export_estimated_poses_to_csv.empty()) {
    // Lambda function will take ownership.
    std::shared_ptr<common::FileLogger> file_logger =
        std::make_shared<common::FileLogger>(
            FLAGS_export_estimated_poses_to_csv);
    constexpr char kDelimiter[] = " ";
    file_logger->writeDataWithDelimiterAndNewLine(
        kDelimiter, "# Timestamp [s]", "p_G_Ix", "p_G_Iy", "p_G_Iz", "q_G_Ix",
        "q_G_Iy", "q_G_Iz", "q_G_Iw");
    CHECK(file_logger != nullptr);
    flow->registerSubscriber<message_flow_topics::MAP_UPDATES>(
        kSubscriberNodeName, message_flow::DeliveryOptions(),
        [file_logger, kDelimiter,
         this](const vio::MapUpdate::ConstPtr& vio_update) {
          CHECK(vio_update != nullptr);
          const aslam::Transformation& T_M_I = vio_update->vinode.get_T_M_I();

          file_logger->writeDataWithDelimiterAndNewLine(
              kDelimiter,
              aslam::time::nanoSecondsToSeconds(vio_update->timestamp_ns),
              T_M_I.getPosition(), T_M_I.getEigenQuaternion());
        });
  }
}

void DataPublisherFlow::visualizeMap(const vi_map::VIMap& vi_map) const {
  static constexpr bool kPublishBaseframes = true;
  static constexpr bool kPublishVertices = true;
  static constexpr bool kPublishEdges = true;
  static constexpr bool kPublishLandmarks = true;
  static constexpr bool kPublishAbsolute6DofConstraints = true;
  plotter_->visualizeMap(
      vi_map, kPublishBaseframes, kPublishVertices, kPublishEdges,
      kPublishLandmarks, kPublishAbsolute6DofConstraints);
}

void DataPublisherFlow::publishOdometryState(
    int64_t timestamp_ns, const vio::ViNodeState& vinode) {
  ros::Time timestamp_ros = createRosTimestamp(timestamp_ns);
  // Publish pose in mission frame.
  const aslam::Transformation& T_M_B = vinode.get_T_M_I();
  geometry_msgs::PoseStamped T_M_I_message;
  tf::poseStampedKindrToMsg(
      T_M_B, timestamp_ros, FLAGS_tf_mission_frame, &T_M_I_message);
  pub_pose_T_M_B_.publish(T_M_I_message);

  aslam::SensorId base_sensor_id;
  if (sensor_manager_.getBaseSensorIdIfUnique(&base_sensor_id)) {
    const vi_map::SensorType base_sensor_type =
        sensor_manager_.getSensorType(base_sensor_id);
    const std::string base_sensor_tf_frame_id =
        visualization::convertSensorTypeToTfFrameId(base_sensor_type) +
        "_0_BASE";

    visualization::publishTF(
        T_M_B, FLAGS_tf_mission_frame, base_sensor_tf_frame_id, timestamp_ros);

    visualization::publishSensorTFs(sensor_manager_, timestamp_ros);
  } else {
    LOG(ERROR) << "There is more than one base sensor, cannot publish base to "
                  "tf tree!";
  }
}

void DataPublisherFlow::publishVinsState(
    int64_t timestamp_ns, const vio::ViNodeState& vinode, const bool has_T_G_M,
    const aslam::Transformation& T_G_M) {
  ros::Time timestamp_ros = createRosTimestamp(timestamp_ns);

  // Publish pose in mission frame.
  const aslam::Transformation& T_M_B = vinode.get_T_M_I();
  geometry_msgs::PoseStamped T_M_I_message;
  tf::poseStampedKindrToMsg(
      T_M_B, timestamp_ros, FLAGS_tf_mission_frame, &T_M_I_message);
  pub_pose_T_M_B_.publish(T_M_I_message);

  aslam::SensorId base_sensor_id;
  if (sensor_manager_.getBaseSensorIdIfUnique(&base_sensor_id)) {
    const vi_map::SensorType base_sensor_type =
        sensor_manager_.getSensorType(base_sensor_id);
    const std::string localized_base_sensor_tf_frame_id =
        visualization::convertSensorTypeToTfFrameId(base_sensor_type) +
        "_0_BASE_LOCALIZED";

    visualization::publishTF(
        T_M_B, FLAGS_tf_mission_frame, localized_base_sensor_tf_frame_id,
        timestamp_ros);

  } else {
    LOG(ERROR) << "There is more than one base sensor, cannot publish base to "
                  "tf tree!";
  }

  // Publish pose in global frame, but only as marker, not into the tf tree,
  // since it is composing this transformation on its own.
  aslam::Transformation T_G_I = T_G_M * T_M_B;
  geometry_msgs::PoseStamped T_G_I_message;
  tf::poseStampedKindrToMsg(
      T_G_I, timestamp_ros, FLAGS_tf_map_frame, &T_G_I_message);
  pub_pose_T_G_I_.publish(T_G_I_message);

  // Publish baseframe transformation.
  geometry_msgs::PoseStamped T_G_M_message;
  tf::poseStampedKindrToMsg(
      T_G_M, timestamp_ros, FLAGS_tf_map_frame, &T_G_M_message);
  pub_baseframe_T_G_M_.publish(T_G_M_message);
  visualization::publishTF(
      T_G_M, FLAGS_tf_map_frame, FLAGS_tf_mission_frame, timestamp_ros);

  // Publish velocity.
  const Eigen::Vector3d& v_M_I = vinode.get_v_M_I();
  geometry_msgs::Vector3Stamped velocity_msg;
  velocity_msg.header.stamp = timestamp_ros;
  velocity_msg.vector.x = v_M_I[0];
  velocity_msg.vector.y = v_M_I[1];
  velocity_msg.vector.z = v_M_I[2];
  pub_velocity_I_.publish(velocity_msg);

  // Publish IMU bias.
  geometry_msgs::Vector3Stamped bias_msg;
  bias_msg.header.stamp = timestamp_ros;
  Eigen::Matrix<double, 6, 1> imu_bias_acc_gyro = vinode.getImuBias();
  bias_msg.vector.x = imu_bias_acc_gyro[0];
  bias_msg.vector.y = imu_bias_acc_gyro[1];
  bias_msg.vector.z = imu_bias_acc_gyro[2];
  pub_imu_acc_bias_.publish(bias_msg);

  bias_msg.vector.x = imu_bias_acc_gyro[3];
  bias_msg.vector.y = imu_bias_acc_gyro[4];
  bias_msg.vector.z = imu_bias_acc_gyro[5];
  pub_imu_gyro_bias_.publish(bias_msg);

  if (FLAGS_publish_debug_markers) {
    stateDebugCallback(vinode, has_T_G_M, T_G_M);
  }
}

void DataPublisherFlow::stateDebugCallback(
    const vio::ViNodeState& vinode, const bool has_T_G_M,
    const aslam::Transformation& T_G_M) {
  constexpr size_t kMarkerId = 0u;
  visualization::Sphere sphere;
  const aslam::Transformation& T_M_B = vinode.get_T_M_I();
  sphere.position = T_M_B.getPosition();
  sphere.radius = 0.2;
  sphere.color = visualization::kCommonGreen;
  sphere.alpha = 0.8;
  T_M_B_spheres_.push_back(sphere);
  visualization::publishSpheres(
      T_M_B_spheres_, kMarkerId, FLAGS_tf_map_frame, "debug", "debug_T_M_B");

  // Publish estimated velocity
  const aslam::Position3D& M_p_M_B = T_M_B.getPosition();
  const Eigen::Vector3d& v_M_I = vinode.get_v_M_I();
  visualization::Arrow arrow;
  arrow.from = M_p_M_B;
  arrow.to = M_p_M_B + v_M_I;
  arrow.scale = 0.3;
  arrow.color = visualization::kCommonRed;
  arrow.alpha = 0.8;
  visualization::publishArrow(
      arrow, kMarkerId, FLAGS_tf_map_frame, "debug", "debug_v_M_I_");

  // Publish global frame if it is available.
  if (has_T_G_M) {
    aslam::Transformation T_G_I = T_G_M * T_M_B;
    visualization::Sphere sphere;
    sphere.position = T_G_I.getPosition();
    sphere.radius = 0.2;
    sphere.color = visualization::kCommonWhite;
    sphere.alpha = 0.8;
    T_G_I_spheres_.push_back(sphere);

    visualization::publishSpheres(
        T_G_I_spheres_, kMarkerId, FLAGS_tf_map_frame, "debug", "debug_T_G_B");
  }
}

void DataPublisherFlow::localizationCallback(
    const common::LocalizationResult& localization_result,
    const bool is_fused_localization_result) {
  const Eigen::Vector3d& p_G_B = localization_result.T_G_B.getPosition();

  visualization::Sphere sphere;
  sphere.position = p_G_B;
  sphere.radius = 0.2;
  if (is_fused_localization_result) {
    sphere.color = visualization::kCommonRed;
  } else {
    switch (localization_result.localization_type) {
      case common::LocalizationType::kVisualFeatureBased:
        sphere.color = visualization::kCommonPink;
        break;
      default:
        LOG(FATAL) << "Unknown localization result type: "
                   << static_cast<int>(localization_result.localization_type);
        break;
    }
  }
  sphere.alpha = 0.8;

  geometry_msgs::PoseWithCovarianceStamped pose_msg;
  pose_msg.header.stamp = createRosTimestamp(localization_result.timestamp_ns);
  pose_msg.header.frame_id = FLAGS_tf_map_frame;
  tf::poseKindrToMsg(localization_result.T_G_B, &pose_msg.pose.pose);
  eigenMatrixToOdometryCovariance(
      localization_result.T_G_B_covariance, pose_msg.pose.covariance.data());

  constexpr size_t kMarkerId = 0u;
  if (is_fused_localization_result) {
    T_G_B_fused_loc_spheres_.push_back(sphere);
    visualization::publishSpheres(
        T_G_B_fused_loc_spheres_, kMarkerId, FLAGS_tf_map_frame, "debug",
        "debug_T_G_B_fused_localizations");
    pub_fused_localization_result_T_G_B_.publish(pose_msg);

  } else {
    T_G_B_raw_loc_spheres_.push_back(sphere);
    visualization::publishSpheres(
        T_G_B_raw_loc_spheres_, kMarkerId, FLAGS_tf_map_frame, "debug",
        "debug_T_G_B_raw_localizations");
    pub_localization_result_T_G_B_.publish(pose_msg);
  }
}

}  //  namespace maplab
