#include "feature-tracking/feature-tracking-pipeline.h"

DEFINE_bool(
    feature_tracker_visualize_keypoint_matches, false,
    "Flag indicating whether keypoint matches are visualized.");  // 判断是否可视化

namespace feature_tracking {

FeatureTrackingPipeline::FeatureTrackingPipeline()
    : feature_tracking_ros_base_topic_("tracking/"),
      visualize_keypoint_matches_(
          FLAGS_feature_tracker_visualize_keypoint_matches) {}

}  // namespace feature_tracking
