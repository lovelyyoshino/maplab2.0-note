#include "visualization/common-rviz-visualization.h"

#include <algorithm>

#include <Eigen/Eigenvalues>
#include <aslam/common/covariance-helpers.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <glog/logging.h>
#include <minkindr_conversions/kindr_msg.h>
#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>

#include "visualization/color.h"
#include "visualization/eigen-visualization.h"

DEFINE_string(tf_map_frame, "map", "map tf frame name");
DEFINE_string(tf_mission_frame, "mission", "mission tf frame name");
DEFINE_string(
    tf_abs_6dof_sensor_frame, "abs_6dof_sensor",
    "abs_6dof_sensor tf frame name");
DEFINE_string(
    tf_odometry_6dof_sensor_frame, "odometry_6dof_sensor",
    "odometry_6dof_sensor tf frame name");
DEFINE_string(
    tf_wheel_odometry_sensor_frame, "wheel_odometry_sensor",
    "wheel_odometry_sensor tf frame name");
DEFINE_string(tf_lc_sensor_frame, "lc_sensor", "lc_sensor tf frame name");
DEFINE_string(tf_lidar_sensor_frame, "lidar_sensor", "lidar tf frame name");
DEFINE_string(
    tf_pointcloud_map_frame, "pointcloud_map", "pointcloud_map tf frame name");
DEFINE_string(
    tf_external_features_frame, "external_features", "external_features tf frame name");
DEFINE_string(
    tf_gps_wgs_sensor_frame, "gps_wgs_sensor", "gps_wgs_sensor tf frame name");
DEFINE_string(
    tf_gps_utm_sensor_frame, "gps_utm_sensor", "gps_utm_sensor tf frame name");
DEFINE_string(tf_imu_frame, "imu", "imu tf frame name");
DEFINE_string(tf_camera_frame, "camera", "camera tf frame name");
DEFINE_string(tf_ncamera_frame, "ncamera", "ncamera tf frame name");
DEFINE_string(tf_imu_refined_frame, "imu_refined", "refined imu tf frame name");
DEFINE_string(vis_default_namespace, "maplab_rviz_namespace", "RVIZ namespace");

namespace visualization {
void setPoseToIdentity(visualization_msgs::Marker* marker) {
  CHECK_NOTNULL(marker);

  marker->pose.position.x = 0.0;
  marker->pose.position.y = 0.0;
  marker->pose.position.z = 0.0;
  marker->pose.orientation.x = 0.0;
  marker->pose.orientation.y = 0.0;
  marker->pose.orientation.z = 0.0;
  marker->pose.orientation.w = 1.0;
}

std_msgs::ColorRGBA commonColorToRosColor(
    const visualization::Color& color, double alpha) {
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);

  std_msgs::ColorRGBA ros_color;
  ros_color.r = (static_cast<double>(color.red) / 255.0);
  ros_color.g = (static_cast<double>(color.green) / 255.0);
  ros_color.b = (static_cast<double>(color.blue) / 255.0);
  ros_color.a = alpha;

  return ros_color;
}

void createEmptySphereListMarker(
    size_t marker_id, const std::string& frame, const std::string& name_space,
    double scale, double alpha, visualization_msgs::Marker* sphere_marker) {
  CHECK_NOTNULL(sphere_marker);
  CHECK_GE(scale, 0.0);
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  sphere_marker->points.clear();
  sphere_marker->colors.clear();

  sphere_marker->id = marker_id;
  sphere_marker->type = visualization_msgs::Marker::SPHERE_LIST;
  sphere_marker->scale.x = scale;
  sphere_marker->scale.y = scale;
  sphere_marker->scale.z = scale;

  visualization::setPoseToIdentity(sphere_marker);

  sphere_marker->color.a = alpha;
  sphere_marker->header.frame_id = frame;
  sphere_marker->header.stamp = ros::Time::now();
  sphere_marker->ns = name_space;
}

void eigen3XdMatrixToSpheres(
    const Eigen::Matrix3Xd& G_points, visualization_msgs::Marker* spheres) {
  CHECK_NOTNULL(spheres);

  spheres->type = visualization_msgs::Marker::SPHERE_LIST;

  setPoseToIdentity(spheres);

  const size_t num_points = G_points.cols();
  VLOG(5) << "Converting " << num_points << " points to spheres.";

  spheres->points.resize(num_points);

  for (size_t idx = 0u; idx < num_points; ++idx) {
    (spheres->points)[idx].x = G_points(0, idx);
    (spheres->points)[idx].y = G_points(1, idx);
    (spheres->points)[idx].z = G_points(2, idx);
  }
}

void eigen3XdMatrixToPointCloud(
    const Eigen::Matrix3Xd& points, const visualization::Color& color,
    unsigned char alpha, sensor_msgs::PointCloud2* point_cloud) {
  CHECK_NOTNULL(point_cloud);

  const size_t num_points = static_cast<size_t>(points.cols());
  VLOG(5) << "Converting " << num_points << " points to point cloud.";

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.reserve(num_points);
  for (size_t idx = 0u; idx < num_points; ++idx) {
    pcl::PointXYZRGB point;
    point.x = points(0, idx);
    point.y = points(1, idx);
    point.z = points(2, idx);
    point.r = color.red;
    point.g = color.green;
    point.b = color.blue;
    point.a = alpha;
    cloud.push_back(point);
  }

  pcl::toROSMsg(cloud, *point_cloud);
}

void eigen3XfMatrixWithIntensitiesToPointCloud(
    const Eigen::Matrix3Xf& points, const Eigen::VectorXf& intensities,
    sensor_msgs::PointCloud2* point_cloud) {
  CHECK_NOTNULL(point_cloud);

  const size_t num_points = static_cast<size_t>(points.cols());
  CHECK_EQ(num_points, static_cast<size_t>(intensities.rows()));

  point_cloud->height = 3;
  point_cloud->width = num_points;
  point_cloud->fields.resize(4);

  point_cloud->fields[0].name = "x";
  point_cloud->fields[0].offset = 0;
  point_cloud->fields[0].count = 1;
  point_cloud->fields[0].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud->fields[1].name = "y";
  point_cloud->fields[1].offset = 4;
  point_cloud->fields[1].count = 1;
  point_cloud->fields[1].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud->fields[2].name = "z";
  point_cloud->fields[2].offset = 8;
  point_cloud->fields[2].count = 1;
  point_cloud->fields[2].datatype = sensor_msgs::PointField::FLOAT32;

  point_cloud->fields[3].name = "rgb";
  point_cloud->fields[3].offset = 12;
  point_cloud->fields[3].count = 1;
  point_cloud->fields[3].datatype = sensor_msgs::PointField::UINT32;

  point_cloud->point_step = 16;
  point_cloud->row_step = point_cloud->point_step * point_cloud->width;
  point_cloud->data.resize(point_cloud->row_step * point_cloud->height);
  point_cloud->is_dense = false;

  int offset = 0;
  for (size_t point_idx = 0u; point_idx < num_points; ++point_idx) {
    const Eigen::Vector3f& point = points.col(point_idx);
    memcpy(&point_cloud->data[offset + 0], &point.x(), sizeof(point.x()));
    memcpy(
        &point_cloud->data[offset + sizeof(point.x())], &point.y(),
        sizeof(point.y()));
    memcpy(
        &point_cloud->data[offset + sizeof(point.x()) + sizeof(point.y())],
        &point.z(), sizeof(point.z()));

    const uint8_t gray = intensities[point_idx];
    const uint32_t rgb = (gray << 16) | (gray << 8) | gray;
    memcpy(&point_cloud->data[offset + 12], &rgb, sizeof(uint32_t));
    offset += point_cloud->point_step;
  }
}

void eigen3XdMatrixToPointCloud(
    const Eigen::Matrix3Xd& points, const visualization::Color& color,
    sensor_msgs::PointCloud2* point_cloud) {
  const unsigned char kFullAlpha = 255u;
  eigen3XdMatrixToPointCloud(points, color, kFullAlpha, point_cloud);
}

void eigen3XdMatrixToPointCloud(
    const Eigen::Matrix3Xd& points, sensor_msgs::PointCloud2* point_cloud) {
  const unsigned char kFullAlpha = 255u;
  eigen3XdMatrixToPointCloud(points, kCommonWhite, kFullAlpha, point_cloud);
}

void spheresToPointCloud(
    const SphereVector& spheres, sensor_msgs::PointCloud2* point_cloud) {
  CHECK_NOTNULL(point_cloud);
  if (spheres.empty()) {
    return;
  }

  const size_t num_spheres = spheres.size();
  VLOG(5) << "Converting " << num_spheres << " points to point cloud.";

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.reserve(num_spheres);
  for (size_t idx = 0u; idx < num_spheres; ++idx) {
    const Sphere& sphere = spheres[idx];

    pcl::PointXYZRGB point;
    point.x = sphere.position(0);
    point.y = sphere.position(1);
    point.z = sphere.position(2);
    point.r = sphere.color.red;
    point.g = sphere.color.green;
    point.b = sphere.color.blue;
    point.a = sphere.alpha;
    cloud.push_back(point);
  }

  pcl::toROSMsg(cloud, *point_cloud);
}

void publishCoordinateFrame(
    const aslam::Transformation& T_G_fi, const std::string& label, size_t id,
    const std::string& topic) {
  CHECK(!topic.empty());
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  Pose pose;
  pose.G_p_B = T_G_fi.getPosition();
  pose.G_q_B = T_G_fi.getRotation().toImplementation();

  pose.id = id;
  pose.scale = 0.15;
  pose.line_width = 0.01;
  pose.alpha = 0.6;

  visualization_msgs::Marker pose_msg;
  drawAxes(
      pose.G_p_B, pose.G_q_B, pose.scale, pose.line_width, pose.alpha,
      pose.color, &pose_msg);

  pose_msg.header.frame_id = "map";
  pose_msg.header.stamp = ros::Time::now();
  pose_msg.id = pose.id;

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, pose_msg);

  if (!label.empty()) {
    visualization_msgs::Marker text_msg;
    text_msg.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    text_msg.id = id;
    text_msg.header.frame_id = "map";
    text_msg.header.stamp = ros::Time::now();

    text_msg.pose.position = eigenToPoint(T_G_fi.getPosition());
    text_msg.pose.orientation =
        eigenToQuaternion(T_G_fi.getRotation().toImplementation());

    text_msg.scale.z = 0.04;
    text_msg.color.a = 1.0;
    text_msg.color.r = 0.8;
    text_msg.color.g = 0.8;
    text_msg.color.b = 1.0;

    text_msg.text = label;

    const std::string text_topic = topic + "_text";
    RVizVisualizationSink::publish<visualization_msgs::Marker>(
        text_topic, text_msg);
  }
}

void publishLines(
    const visualization::LineSegmentVector& line_segments, size_t marker_id,
    const std::string& frame, const std::string& name_space,
    const std::string& topic, const bool wait_for_subscriber) {
  CHECK(!topic.empty());
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = marker_id;

  double alpha = 1.0;
  if (!line_segments.empty()) {
    marker.scale.x = line_segments[0].scale;  // y and z are not used.
    alpha = line_segments[0].alpha;
  }

  setPoseToIdentity(&marker);

  marker.color.a = alpha;
  marker.header.frame_id = frame;
  marker.header.stamp = ros::Time();
  marker.ns = name_space;

  const size_t num_line_segments = line_segments.size();

  marker.points.resize(2u * num_line_segments);
  marker.colors.resize(2u * num_line_segments);

  for (size_t idx = 0u; idx < num_line_segments; ++idx) {
    geometry_msgs::Point vertex0;
    vertex0.x = line_segments[idx].from[0];
    vertex0.y = line_segments[idx].from[1];
    vertex0.z = line_segments[idx].from[2];
    geometry_msgs::Point vertex1;
    vertex1.x = line_segments[idx].to[0];
    vertex1.y = line_segments[idx].to[1];
    vertex1.z = line_segments[idx].to[2];

    CHECK_EQ(line_segments[idx].alpha, alpha)
        << "All line segments must have "
           "identical alpha. Use a marker array instead if you want "
           "individual "
           "alpha values.";
    std_msgs::ColorRGBA color =
        commonColorToRosColor(line_segments[idx].color, alpha);

    marker.colors[2u * idx] = color;
    marker.colors[(2u * idx) + 1u] = color;

    marker.points[2u * idx] = vertex0;
    marker.points[(2u * idx) + 1u] = vertex1;
  }

  RVizVisualizationSink::publish<visualization_msgs::Marker>(
      topic, marker, wait_for_subscriber);
  ros::spinOnce();
}

void publishVerticesFromPoseVector(
    const PoseVector& poses, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_poses = poses.size();
  if (num_poses == 0u) {
    return;
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::MarkerArray pose_array;

  for (size_t pose_idx = 0u; pose_idx < num_poses; ++pose_idx) {
    visualization_msgs::Marker pose_msg;

    drawAxes(
        poses[pose_idx].G_p_B, poses[pose_idx].G_q_B, poses[pose_idx].scale,
        poses[pose_idx].line_width, poses[pose_idx].alpha,
        poses[pose_idx].color, &pose_msg);

    pose_msg.header.frame_id = frame;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.id = poses[pose_idx].id;
    pose_msg.action = poses[pose_idx].action;
    pose_msg.ns = name_space;
    pose_array.markers.push_back(pose_msg);
  }
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, pose_array);
}

void publish3DPointsAsPointCloud(
    const Eigen::Matrix3Xd& points_G, const visualization::Color& color,
    double alpha, const std::string& frame, const std::string& topic) {
  CHECK(!topic.empty());
  if (points_G.cols() == 0u) {
    return;
  }
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The point cloud will be invisible.";
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  sensor_msgs::PointCloud2 point_cloud;
  eigen3XdMatrixToPointCloud(
      points_G, color, std::floor(255.0 * alpha), &point_cloud);

  point_cloud.header.frame_id = frame;
  point_cloud.header.stamp = ros::Time::now();

  RVizVisualizationSink::publish<sensor_msgs::PointCloud2>(topic, point_cloud);
}

void publish3DPointsAsPointCloud(
    const Eigen::Matrix3Xf& points, const Eigen::VectorXf& intensities,
    const std::string& frame, const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_points = static_cast<size_t>(points.cols());
  if (num_points == 0u) {
    return;
  }
  CHECK_EQ(num_points, static_cast<size_t>(intensities.rows()));
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  sensor_msgs::PointCloud2 point_cloud;
  eigen3XfMatrixWithIntensitiesToPointCloud(points, intensities, &point_cloud);

  point_cloud.header.frame_id = frame;
  point_cloud.header.stamp = ros::Time::now();

  RVizVisualizationSink::publish<sensor_msgs::PointCloud2>(topic, point_cloud);
}

void publishSpheresAsPointCloud(
    const SphereVector& spheres, const std::string& frame,
    const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_sphers = spheres.size();
  if (num_sphers == 0u) {
    return;
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  sensor_msgs::PointCloud2 point_cloud;
  spheresToPointCloud(spheres, &point_cloud);

  point_cloud.header.frame_id = frame;
  point_cloud.header.stamp = ros::Time::now();

  RVizVisualizationSink::publish<sensor_msgs::PointCloud2>(topic, point_cloud);
}

void publish3DPointsAsSpheres(
    const Eigen::Matrix3Xd& points, const visualization::Color& color,
    double alpha, double scale, size_t marker_id, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The spheres will be invisible.";
  }
  const size_t num_points = static_cast<size_t>(points.cols());
  if (num_points == 0u) {
    LOG(WARNING) << "Zero points to visualize as a spheres. Returning.";
    return;
  }

  visualization_msgs::Marker marker;
  createEmptySphereListMarker(
      marker_id, frame, name_space, scale, alpha, &marker);

  marker.points.resize(num_points);
  marker.colors.resize(num_points);

  std_msgs::ColorRGBA color_rgba = commonColorToRosColor(color, alpha);

  for (size_t point_idx = 0u; point_idx < num_points; ++point_idx) {
    const Eigen::Vector3d& eigen_point = points.col(point_idx);

    geometry_msgs::Point point;
    point.x = eigen_point(0);
    point.y = eigen_point(1);
    point.z = eigen_point(2);

    marker.colors[point_idx] = color_rgba;
    marker.points[point_idx] = point;
  }

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishLines(
    const Eigen::Matrix3Xd& points_from, const Eigen::Matrix3Xd& points_to,
    const std::vector<visualization::Color>& color_list, double alpha,
    double scale, size_t marker_id, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_lines = points_from.cols();
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  CHECK_GE(scale, 0.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The lines will be invisible.";
  }
  if (scale < 1e-6) {
    LOG(WARNING) << "Scale is 0.0. The lines will be invisible.";
  }

  CHECK_EQ(num_lines, static_cast<size_t>(points_to.cols()));
  CHECK_EQ(num_lines, color_list.size());

  LineSegmentVector line_segments(num_lines);

  for (size_t line_idx = 0u; line_idx < num_lines; ++line_idx) {
    LineSegment& line_segment = line_segments[line_idx];

    line_segment.from = points_from.col(line_idx);
    line_segment.to = points_to.col(line_idx);

    line_segment.alpha = alpha;
    line_segment.color = color_list[line_idx];
    line_segment.scale = scale;
  }

  publishLines(line_segments, marker_id, frame, name_space, topic);
}

void publishLines(
    const Eigen::Vector3d& common_line_start_point,
    const Vector3dList& line_end_points,
    const std::vector<visualization::Color>& colors, double alpha, double scale,
    size_t marker_id, const std::string& frame, const std::string& name_space,
    const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_lines = line_end_points.size();
  if (num_lines == 0u) {
    return;
  }
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The lines will be invisible.";
  }

  Eigen::Matrix3Xd points_from = Eigen::Matrix3Xd::Zero(3, num_lines);
  Eigen::Matrix3Xd points_to = Eigen::Matrix3Xd::Zero(3, num_lines);
  for (size_t line_idx = 0u; line_idx < num_lines; ++line_idx) {
    points_from.col(line_idx) = common_line_start_point;
    points_to.col(line_idx) = line_end_points[line_idx];
  }

  publishLines(
      points_from, points_to, colors, alpha, scale, marker_id, frame,
      name_space, topic);
}

void publishArrow(
    const Arrow& arrow, size_t marker_id, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());

  CHECK_GE(arrow.alpha, 0.0);
  CHECK_LE(arrow.alpha, 1.0);
  CHECK_GE(arrow.scale, 0.0);
  if (arrow.alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The arrow will be invisible.";
  }
  if (arrow.scale < 1e-6) {
    LOG(WARNING) << "Scale is 0.0. The arrow will be invisible.";
  }

  visualization_msgs::Marker marker;
  marker.id = marker_id;
  marker.ns = name_space;

  std_msgs::ColorRGBA color = commonColorToRosColor(arrow.color, arrow.alpha);
  drawArrow(
      arrow.from, arrow.to, color, arrow.scale * 0.1, arrow.scale * 0.2, 0,
      &marker);

  marker.header.frame_id = frame;
  marker.header.stamp = ros::Time();

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishArrows(
    const ArrowVector& arrows, size_t marker_id, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());

  visualization_msgs::MarkerArray marker_array;
  for (uint16_t i = 0; i < arrows.size(); ++i) {
    const Arrow& arrow = arrows[i];
    visualization_msgs::Marker marker;

    // Can't draw 0 length transformation edges so we skip them
    double norm = (arrow.to - arrow.from).norm();
    if (norm < 1e-6) {
      continue;
    }

    CHECK_GE(arrow.alpha, 0.0);
    CHECK_LE(arrow.alpha, 1.0);
    CHECK_GE(arrow.scale, 0.0);
    if (arrow.alpha < 1e-6) {
      LOG(WARNING) << "Alpha is 0.0. The arrow will be invisible.";
    }
    if (arrow.scale < 1e-6) {
      LOG(WARNING) << "Scale is 0.0. The arrow will be invisible.";
    }

    marker.id = marker_id + i;
    marker.ns = name_space;

    std_msgs::ColorRGBA color = commonColorToRosColor(arrow.color, arrow.alpha);
    drawArrow(
        arrow.from, arrow.to, color, norm * 0.05, norm * 0.1, norm * 0.2,
        &marker);

    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time();
    marker_array.markers.emplace_back(marker);
  }

  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, marker_array);
}

void publishNormals(
    const Eigen::Matrix3Xd& p_G_p0, const Eigen::Matrix3Xd& bearings,
    const std::string& frame, const std::string& name_space,
    const std::string& topic) {
  const size_t num_normals = p_G_p0.cols();
  CHECK_EQ(num_normals, static_cast<size_t>(bearings.cols()));
  if (num_normals == 0u) {
    return;
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  constexpr double kArrowLength = 0.2;
  constexpr double kArrowDiameter = 0.1;

  const Eigen::Matrix3Xd p_G_p1 = p_G_p0 + kArrowLength * bearings;

  visualization_msgs::MarkerArray array;
  for (size_t normal_idx = 0u; normal_idx < num_normals; ++normal_idx) {
    visualization_msgs::Marker marker;
    marker.id = normal_idx;
    marker.ns = name_space;

    std_msgs::ColorRGBA rgba;
    rgba.r = 1.0;
    rgba.g = 0.0;
    rgba.b = 0.0;
    rgba.a = 1.0;

    drawArrow(
        p_G_p0.col(normal_idx), p_G_p1.col(normal_idx), rgba,
        kArrowDiameter * 0.1, kArrowDiameter * 0.2, 0, &marker);

    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time::now();

    array.markers.push_back(marker);
  }
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(topic, array);
}

void publishFilledBoxes(
    const FilledBoxVector& boxes, const std::vector<size_t>& box_marker_ids,
    const std::string& frame, const std::string& wireframe_namespace,
    const std::string& filling_namespace, const std::string& topic) {
  const size_t num_boxes = boxes.size();
  CHECK_EQ(num_boxes, box_marker_ids.size());
  if (num_boxes == 0u) {
    return;
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::MarkerArray marker_wireframes;
  visualization_msgs::MarkerArray marker_fills;

  for (size_t box_idx = 0u; box_idx < num_boxes; ++box_idx) {
    const FilledBox& box = boxes[box_idx];

    if (box.wireframe_alpha > std::numeric_limits<double>::epsilon()) {
      visualization_msgs::Marker marker_wireframe;
      marker_wireframe.type = visualization_msgs::Marker::LINE_LIST;
      marker_wireframe.action = visualization_msgs::Marker::ADD;
      marker_wireframe.id = box_marker_ids[box_idx];
      marker_wireframe.ns = wireframe_namespace;
      marker_wireframe.header.frame_id = frame;
      marker_wireframe.header.stamp = ros::Time();

      marker_wireframe.color =
          commonColorToRosColor(box.wireframe_color, box.wireframe_alpha);

      marker_wireframe.scale.x = box.wireframe_width;  // y and z are not used.

      geometry_msgs::Point vertex0;
      geometry_msgs::Point vertex1;
      geometry_msgs::Point vertex2;
      geometry_msgs::Point vertex3;
      geometry_msgs::Point vertex4;
      geometry_msgs::Point vertex5;
      geometry_msgs::Point vertex6;
      geometry_msgs::Point vertex7;

      vertex0.x = box.from.x();
      vertex0.y = box.from.y();
      vertex0.z = box.from.z();

      vertex1.x = box.from.x();
      vertex1.y = box.to.y();
      vertex1.z = box.from.z();

      vertex2.x = box.to.x();
      vertex2.y = box.to.y();
      vertex2.z = box.from.z();

      vertex3.x = box.to.x();
      vertex3.y = box.from.y();
      vertex3.z = box.from.z();

      vertex4.x = box.from.x();
      vertex4.y = box.from.y();
      vertex4.z = box.to.z();

      vertex5.x = box.from.x();
      vertex5.y = box.to.y();
      vertex5.z = box.to.z();

      vertex6.x = box.to.x();
      vertex6.y = box.to.y();
      vertex6.z = box.to.z();

      vertex7.x = box.to.x();
      vertex7.y = box.from.y();
      vertex7.z = box.to.z();

      // Lower square.
      marker_wireframe.points.push_back(vertex0);
      marker_wireframe.points.push_back(vertex1);
      marker_wireframe.points.push_back(vertex1);
      marker_wireframe.points.push_back(vertex2);
      marker_wireframe.points.push_back(vertex2);
      marker_wireframe.points.push_back(vertex3);
      marker_wireframe.points.push_back(vertex3);
      marker_wireframe.points.push_back(vertex0);

      // Link to upper square.
      marker_wireframe.points.push_back(vertex0);
      marker_wireframe.points.push_back(vertex4);
      marker_wireframe.points.push_back(vertex1);
      marker_wireframe.points.push_back(vertex5);
      marker_wireframe.points.push_back(vertex2);
      marker_wireframe.points.push_back(vertex6);
      marker_wireframe.points.push_back(vertex3);
      marker_wireframe.points.push_back(vertex7);

      // Upper square.
      marker_wireframe.points.push_back(vertex4);
      marker_wireframe.points.push_back(vertex5);
      marker_wireframe.points.push_back(vertex5);
      marker_wireframe.points.push_back(vertex6);
      marker_wireframe.points.push_back(vertex6);
      marker_wireframe.points.push_back(vertex7);
      marker_wireframe.points.push_back(vertex7);
      marker_wireframe.points.push_back(vertex4);

      marker_wireframes.markers.push_back(marker_wireframe);
    }

    visualization_msgs::Marker marker_fill;
    marker_fill.type = visualization_msgs::Marker::CUBE;
    marker_fill.action = visualization_msgs::Marker::ADD;
    marker_fill.id = box_marker_ids[box_idx];
    marker_fill.ns = filling_namespace;
    marker_fill.header.frame_id = frame;
    marker_fill.header.stamp = ros::Time::now();

    marker_fill.pose.position.x = (box.to.x() + box.from.x()) / 2.0;
    marker_fill.pose.position.y = (box.to.y() + box.from.y()) / 2.0;
    marker_fill.pose.position.z = (box.to.z() + box.from.z()) / 2.0;
    marker_fill.scale.x = box.to.x() - box.from.x();
    marker_fill.scale.y = box.to.y() - box.from.y();
    marker_fill.scale.z = box.to.z() - box.from.z();

    marker_fill.color = commonColorToRosColor(box.fill_color, box.fill_alpha);

    marker_fills.markers.push_back(marker_fill);
  }
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, marker_fills);
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, marker_wireframes);
}

void publishFilledBox(
    const FilledBox& box, size_t marker_id, const std::string& frame,
    const std::string& wireframe_namespace,
    const std::string& filling_namespace, const std::string& topic) {
  CHECK(!topic.empty());
  FilledBoxVector boxes;
  boxes.push_back(box);

  std::vector<size_t> box_marker_ids;
  box_marker_ids.push_back(marker_id);

  publishFilledBoxes(
      boxes, box_marker_ids, frame, wireframe_namespace, filling_namespace,
      topic);
}

void publishSpheres(
    const SphereVector& spheres, size_t marker_id, const std::string& frame,
    const std::string& name_space, const std::string& topic) {
  CHECK(!topic.empty());
  const size_t num_spheres = spheres.size();
  if (num_spheres == 0u) {
    return;
  }

  const double alpha = spheres[0].alpha;
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "0 alpha used for publishing a list of spheres. They will "
                    "all be invisible!";
  }

  const double scale = spheres[0].radius;
  visualization_msgs::Marker marker;
  createEmptySphereListMarker(
      marker_id, frame, name_space, scale, alpha, &marker);

  marker.points.resize(num_spheres);
  marker.colors.resize(num_spheres);

  for (size_t sphere_idx = 0u; sphere_idx < num_spheres; ++sphere_idx) {
    const Sphere& sphere = spheres[sphere_idx];

    const Eigen::Vector3d& eigen_point = sphere.position;

    geometry_msgs::Point point;
    point.x = eigen_point(0);
    point.y = eigen_point(1);
    point.z = eigen_point(2);

    CHECK_EQ(sphere.alpha, alpha)
        << "All spheres in the sphere"
           " list must have the same alpha value. Use plotting with a marker "
           "array if you want each sphere to have an individual alpha value.";

    CHECK_EQ(sphere.radius, scale)
        << "All spheres in the sphere list must have the same radius. Use "
        << "plotting with a marker array if you want each sphere to have an "
        << "individual radius.";

    marker.colors[sphere_idx] = commonColorToRosColor(sphere.color, alpha);
    marker.points[sphere_idx] = point;
  }

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishMesh(
    const Aligned<std::vector, Eigen::Matrix3d>& triangles, size_t marker_id,
    double scale, const std::string& frame, const std::string name_space,
    const std::string& topic) {
  CHECK(!topic.empty());
  CHECK_GE(scale, 0.0);
  if (scale < 1e-6) {
    LOG(WARNING) << "Scale is 0.0. The mesh will be invisible.";
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = marker_id;

  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;

  marker.header.frame_id = frame;
  marker.header.stamp = ros::Time::now();
  marker.ns = name_space;

  const size_t num_triangles = triangles.size();
  marker.points.resize(3u * num_triangles);
  marker.colors.resize(3u * num_triangles);
  for (size_t idx = 0u; idx < num_triangles; ++idx) {
    // Shade by inclination.
    const Eigen::Matrix<double, 3, 2> diff =
        triangles[idx].rightCols<2>() - triangles[idx].leftCols<2>();
    const Eigen::Vector3d normal = diff.col(0).cross(diff.col(1)).normalized();
    const double shade = 0.6 - Eigen::Vector3d::UnitZ().dot(normal) * 0.3;

    std_msgs::ColorRGBA color;
    color.r = shade;
    color.g = shade;
    color.b = shade;
    color.a = 1.0;

    for (size_t pdx = 0u; pdx < 3u; ++pdx) {
      marker.points[(3u * idx) + pdx] = eigenToPoint(triangles[idx].col(pdx));
      marker.colors[(3u * idx) + pdx] = color;
    }
  }
  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishMesh(
    const std::string& mesh_filename, const aslam::Transformation& T_G_fi,
    const aslam::Transformation& T_fi_mesh, const double scale,
    const visualization::Color color, size_t marker_id,
    const std::string& frame, const std::string& name_space,
    const std::string& topic) {
  CHECK(!topic.empty());
  CHECK_GE(scale, 0.0);
  if (scale < 1e-6) {
    LOG(WARNING) << "Scale is 0.0. The mesh will be invisible.";
  }
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = marker_id;
  marker.mesh_resource = mesh_filename;
  marker.mesh_use_embedded_materials = false;

  marker.header.frame_id = frame;
  marker.header.stamp = ros::Time::now();
  marker.ns = name_space;

  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;

  marker.color = commonColorToRosColor(color, 1.0);

  const aslam::Transformation T_G_mesh = T_G_fi * T_fi_mesh;

  marker.pose.position = eigenToPoint(T_G_mesh.getPosition());
  marker.pose.orientation =
      eigenToQuaternion(T_G_mesh.getRotation().toImplementation());

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishTransformations(
    const aslam::TransformationVector& Ts,
    const std::vector<visualization::Color>& colors, double alpha,
    const std::string& frame, const std::string& name_space,
    const std::string& topic) {
  CHECK(!topic.empty());
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  if (alpha < 1e-6) {
    LOG(WARNING) << "Alpha is 0.0. The transformations will be invisible.";
  }
  const size_t num_transformations = Ts.size();
  if (num_transformations == 0u) {
    return;
  }
  CHECK_EQ(num_transformations, colors.size());
  CHECK(ros::isInitialized()) << "ROS hasn't been initialized. Call "
                              << "RVizVisualizationSink::init() in your "
                                 "application code if you intend"
                              << " to use RViz visualizations.";

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.resize(num_transformations);

  constexpr double kArrowLength = 0.2;
  constexpr double kArrowDiameter = 0.1;

  for (size_t transformation_idx = 0u; transformation_idx < num_transformations;
       ++transformation_idx) {
    const aslam::Transformation& T = Ts[transformation_idx];
    const visualization::Color& color = colors[transformation_idx];

    visualization_msgs::Marker& marker =
        marker_array.markers[transformation_idx];
    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time::now();
    marker.id = transformation_idx;
    marker.ns = name_space;

    drawArrow(
        T.getPosition(), T * (kArrowLength * Eigen::Vector3d::UnitZ()),
        commonColorToRosColor(color, alpha), kArrowDiameter * 0.1,
        kArrowDiameter * 0.2, 0, &marker);
  }
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, marker_array);
}

void deleteMarker(const std::string& topic, size_t marker_id) {
  CHECK(!topic.empty());

  visualization_msgs::Marker marker;
  marker.id = marker_id;
  marker.action = visualization_msgs::Marker::DELETE;

  RVizVisualizationSink::publish<visualization_msgs::Marker>(topic, marker);
}

void publishTF(
    const aslam::Transformation& T, const std::string& frame_id,
    const std::string& child_frame_id) {
  publishTF(T, frame_id, child_frame_id, ros::Time::now());
}

void publishTF(
    const aslam::Transformation& T, const std::string& frame_id,
    const std::string& child_frame_id, const ros::Time& ros_time) {
  CHECK(!frame_id.empty());
  CHECK(!child_frame_id.empty());
  const Eigen::Vector3d& p = T.getPosition();

  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(p(0), p(1), p(2)));

  tf::Quaternion tf_quaternion(
      T.getRotation().x(), T.getRotation().y(), T.getRotation().z(),
      T.getRotation().w());
  tf_transform.setRotation(tf_quaternion);

  static tf::TransformBroadcaster tf_br;
  tf_br.sendTransform(
      tf::StampedTransform(tf_transform, ros_time, frame_id, child_frame_id));
}

const std::string convertSensorTypeToTfFrameId(
    const vi_map::SensorType sensor_type) {
  switch (sensor_type) {
    case vi_map::SensorType::kNCamera:
      return FLAGS_tf_ncamera_frame;
    case vi_map::SensorType::kCamera:
      return FLAGS_tf_camera_frame;
    case vi_map::SensorType::kImu:
      return FLAGS_tf_imu_frame;
    case vi_map::SensorType::kLoopClosureSensor:
      return FLAGS_tf_lc_sensor_frame;
    case vi_map::SensorType::kGpsWgs:
      return FLAGS_tf_gps_wgs_sensor_frame;
    case vi_map::SensorType::kGpsUtm:
      return FLAGS_tf_gps_utm_sensor_frame;
    case vi_map::SensorType::kLidar:
      return FLAGS_tf_lidar_sensor_frame;
    case vi_map::SensorType::kPointCloudMapSensor:
      return FLAGS_tf_pointcloud_map_frame;
    case vi_map::SensorType::kOdometry6DoF:
      return FLAGS_tf_odometry_6dof_sensor_frame;
    case vi_map::SensorType::kAbsolute6DoF:
      return FLAGS_tf_abs_6dof_sensor_frame;
    case vi_map::SensorType::kWheelOdometry:
      return FLAGS_tf_wheel_odometry_sensor_frame;
    case vi_map::SensorType::kExternalFeatures:
      return FLAGS_tf_external_features_frame;
    default:
      LOG(FATAL) << "Unknown sensor type: " << static_cast<int>(sensor_type);
  }
}

void publishSensorTFs(const vi_map::SensorManager& sensor_manager,
                      const ros::Time& ros_time) {
  aslam::SensorIdSet all_sensor_ids;
  sensor_manager.getAllSensorIds(&all_sensor_ids);

  const uint32_t num_sensor_types =
      static_cast<int>(vi_map::SensorType::kInvalid);
  std::vector<uint32_t> sensor_number_map(num_sensor_types, 0u);

  std::unordered_map<aslam::SensorId, uint32_t> sensor_to_number_map;

  for (const aslam::SensorId& sensor_id : all_sensor_ids) {
    const aslam::SensorId& base_sensor_id =
        sensor_manager.getBaseSensorId(sensor_id);
    const vi_map::SensorType base_sensor_type =
        sensor_manager.getSensorType(base_sensor_id);
    uint32_t base_sensor_number;
    if (sensor_to_number_map.count(base_sensor_id) == 0) {
      base_sensor_number =
          sensor_number_map[static_cast<int>(base_sensor_type)]++;
      sensor_to_number_map[base_sensor_id] = base_sensor_number;
    } else {
      base_sensor_number = sensor_to_number_map[base_sensor_id];
    }

    const std::string base_sensor_tf_frame_id =
        convertSensorTypeToTfFrameId(base_sensor_type) + "_" +
        std::to_string(base_sensor_number) + "_BASE";

    const vi_map::SensorType sensor_type =
        sensor_manager.getSensorType(sensor_id);
    const aslam::Transformation& T_B_S =
        sensor_manager.getSensor_T_B_S(sensor_id);
    uint32_t sensor_number;
    if (sensor_to_number_map.count(sensor_id) == 0) {
      sensor_number = sensor_number_map[static_cast<int>(sensor_type)]++;
      sensor_to_number_map[sensor_id] = sensor_number;
    } else {
      sensor_number = sensor_to_number_map[sensor_id];
    }
    const std::string sensor_tf_frame_id =
        convertSensorTypeToTfFrameId(sensor_type) + "_" +
        std::to_string(sensor_number);

    if (sensor_id != base_sensor_id) {
      visualization::publishTF(
          T_B_S, base_sensor_tf_frame_id, sensor_tf_frame_id, ros_time);
    }

    if (sensor_type == vi_map::SensorType::kNCamera) {
      const auto& ncamera = sensor_manager.getSensor<aslam::NCamera>(sensor_id);
      for (size_t camera_index = 0; camera_index < ncamera.getNumCameras(); ++camera_index) {
        const aslam::Transformation& T_C_B = ncamera.get_T_C_B(camera_index);
        const std::string camera_tf_frame_id =
            FLAGS_tf_camera_frame + "_" + std::to_string(sensor_number)
            + "." + std::to_string(camera_index);
        visualization::publishTF(
            T_C_B.inverse(), sensor_tf_frame_id, camera_tf_frame_id, ros_time);
      }
    }
  }
}

void makeRightHandedCoordinateSystem(
    Eigen::Matrix3d* eigenvectors, Eigen::Vector3d* eigenvalues) {
  CHECK_NOTNULL(eigenvectors);
  CHECK_NOTNULL(eigenvalues);
  // Note that sorting of eigenvalues may end up with left-hand coordinate
  // system. So here we correctly sort it so that it does end up being
  // righ-handed and normalised.
  Eigen::Vector3d c0 = eigenvectors->block<3, 1>(0, 0);
  c0.normalize();
  Eigen::Vector3d c1 = eigenvectors->block<3, 1>(0, 1);
  c1.normalize();
  Eigen::Vector3d c2 = eigenvectors->block<3, 1>(0, 2);
  c2.normalize();
  Eigen::Vector3d cc = c0.cross(c1);
  if (cc.dot(c2) < 0) {
    (*eigenvectors) << c1, c0, c2;
    double e = (*eigenvalues)[0];
    (*eigenvalues)[0] = (*eigenvalues)[1];
    (*eigenvalues)[1] = e;
  } else {
    (*eigenvectors) << c0, c1, c2;
  }
}

void publishPoseCovariances(
    const std::vector<aslam::Transformation>& T_G_B_vec,
    const std::vector<aslam::TransformationCovariance>& B_cov_vec,
    const Color& color, const std::string& frame, const std::string& name_space,
    const std::string& topic) {
  CHECK_EQ(T_G_B_vec.size(), B_cov_vec.size());
  const size_t num_covariances = B_cov_vec.size();

  visualization_msgs::MarkerArray marker_array;

  for (size_t idx = 0u; idx < num_covariances; ++idx) {
    const aslam::Transformation& T_G_B = T_G_B_vec[idx];
    const aslam::TransformationCovariance& B_cov = B_cov_vec[idx];

    aslam::TransformationCovariance G_cov = aslam::TransformationCovariance();
    aslam::common::rotateCovariance(T_G_B, B_cov, &G_cov);

    Eigen::Vector3d eigenvalues(Eigen::Vector3d::Identity());
    Eigen::Matrix3d eigenvectors(Eigen::Matrix3d::Zero());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(
        G_cov.topLeftCorner<3, 3>());
    // Compute eigenvectors and eigenvalues
    if (eigensolver.info() == Eigen::Success) {
      eigenvalues = eigensolver.eigenvalues();
      eigenvectors = eigensolver.eigenvectors();
    } else {
      LOG(WARNING) << "Computing eigen decomposition of covariance matrix "
                   << "failed. B_cov:\n"
                   << B_cov << "\nG_cov:\n"
                   << G_cov;
      continue;
    }

    makeRightHandedCoordinateSystem(&eigenvectors, &eigenvalues);

    Eigen::Vector3d scale;
    scale << 2 * std::sqrt(eigenvalues[0]), 2 * std::sqrt(eigenvalues[1]),
        2 * std::sqrt(eigenvalues[2]);

    Eigen::Matrix3d R_B_E;
    R_B_E << eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2),
        eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2),
        eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2);

    const bool is_valid_rotation_matrix =
        aslam::Transformation::Rotation::isValidRotationMatrix(
            R_B_E, 1e-4 /*threshold*/);
    const bool scale_has_nans = scale.hasNaN();

    if (!is_valid_rotation_matrix || scale_has_nans) {
      LOG(WARNING) << "Extracting the eigen basis from the covariance matrix "
                   << "failed. B_Covariance(position): \n"
                   << B_cov.topLeftCorner<3, 3>()
                   << "\nG_Covariance(position): \n"
                   << G_cov.topLeftCorner<3, 3>() << "\neigen values:\n"
                   << eigenvalues << "\neigenvectors:\n"
                   << eigenvectors << "\nReason:\n\tinvalid rotation: "
                   << !is_valid_rotation_matrix
                   << "\n\tscale has nans: " << scale_has_nans;
      continue;
    }

    aslam::Transformation::Rotation q_B_E =
        aslam::Transformation::Rotation::fromApproximateRotationMatrix(R_B_E);
    aslam::Transformation::Position B_t_B_E;
    B_t_B_E.setZero();
    const aslam::Transformation T_B_E(q_B_E, B_t_B_E);

    const aslam::Transformation T_G_E = T_G_B * T_B_E;

    // create visualization marker
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time();
    marker.ns = name_space;
    marker.id = idx;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    tf::poseKindrToMsg(T_G_E, &marker.pose);

    marker.scale.x = scale.x() * 2;
    marker.scale.y = scale.y() * 2;
    marker.scale.z = scale.z() * 2;

    marker.color = commonColorToRosColor(color, 0.3 /*alpha*/);

    marker_array.markers.push_back(marker);
  }
  RVizVisualizationSink::publish<visualization_msgs::MarkerArray>(
      topic, marker_array);
}

}  // namespace visualization
