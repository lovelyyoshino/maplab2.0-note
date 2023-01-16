#include "feature-tracking/feature-detection-extraction.h"

#include <array>
#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/tracking-helpers.h>
#include <brisk/brisk.h>
#include <gflags/gflags.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "feature-tracking/gridded-detector.h"

namespace feature_tracking {
// 用于检测特征点的类
FeatureDetectorExtractor::FeatureDetectorExtractor(
    const aslam::Camera& camera,
    const FeatureTrackingExtractorSettings& extractor_settings,
    const FeatureTrackingDetectorSettings& detector_settings)
    : camera_(camera),
      extractor_settings_(extractor_settings),
      detector_settings_(detector_settings) {
  initialize();
}

// 初始化
void FeatureDetectorExtractor::initialize() {
  CHECK_LT(
      2 * detector_settings_.min_tracking_distance_to_image_border_px,
      camera_.imageWidth());  // 检测特征点的最小距离不能超过图像边界
  CHECK_LT(
      2 * detector_settings_.min_tracking_distance_to_image_border_px,
      camera_.imageHeight());

  // No distance to the edges is required for the gridded detector.
  const size_t orb_detector_edge_threshold =
      detector_settings_.gridded_detector_use_gridded
          ? 0u
          : detector_settings_
                .orb_detector_edge_threshold;  // orb检测器的边缘阈值

  detector_ = cv::ORB::create(
      detector_settings_.gridded_detector_use_gridded
          ? detector_settings_.gridded_detector_cell_num_features
          : detector_settings_.orb_detector_number_features,
      detector_settings_.orb_detector_scale_factor,
      detector_settings_.orb_detector_pyramid_levels,
      orb_detector_edge_threshold, detector_settings_.orb_detector_first_level,
      detector_settings_.orb_detector_WTA_K,
      detector_settings_.orb_detector_score_type,
      detector_settings_.orb_detector_patch_size,
      detector_settings_.orb_detector_fast_threshold);  // 通过Opencv创建ORB

  switch (extractor_settings_.descriptor_type) {
    case FeatureTrackingExtractorSettings::DescriptorType::kBrisk:
      extractor_ = new brisk::BriskDescriptorExtractor(
          extractor_settings_.rotation_invariant,
          extractor_settings_.scale_invariant);  // 通过Brisk创建描述子
      break;
    case FeatureTrackingExtractorSettings::DescriptorType::kOcvFreak:
      extractor_ = cv::xfeatures2d::FREAK::create(
          extractor_settings_.rotation_invariant,
          extractor_settings_.scale_invariant,
          extractor_settings_.freak_pattern_scale,
          detector_settings_
              .orb_detector_pyramid_levels);  // 通过Freak创建描述子
      break;
    default:
      LOG(FATAL) << "Unknown descriptor type.";
      break;
  }
}

cv::Ptr<cv::DescriptorExtractor>
FeatureDetectorExtractor::getExtractorPtr()  // 获取描述子
    const {
  return extractor_;
}

// 检测特征点
void FeatureDetectorExtractor::detectAndExtractFeatures(
    aslam::VisualFrame* frame) const {
  CHECK_NOTNULL(frame);  // 检测特征点的帧不能为空
  CHECK(frame->hasRawImage()) << "Can only detect keypoints if the frame has a "
                                 "raw image";  // 检测特征点的帧必须有原始图像
  CHECK_EQ(
      camera_.getId(),
      CHECK_NOTNULL(frame->getCameraGeometry().get())
          ->getId());  // 检测特征点的帧的相机必须和当前相机一致

  timing::Timer timer_detection("keypoint detection");  // 计时器

  std::vector<cv::KeyPoint> keypoints_cv;
  const cv::Mat& image = frame->getRawImage();  // 获取原始图像

  if (detector_settings_.gridded_detector_use_gridded) {
    // gridded detection to ensure a certain distribution of keypoints across
    // the image.
    detectKeypointsGridded(
        detector_, image, /*detection_mask=*/cv::Mat(),
        detector_settings_.detector_use_nonmaxsuppression,
        detector_settings_.detector_nonmaxsuppression_radius,
        detector_settings_.detector_nonmaxsuppression_ratio_threshold,
        detector_settings_.orb_detector_number_features,
        detector_settings_.max_feature_count,
        detector_settings_.gridded_detector_cell_num_features,
        detector_settings_.gridded_detector_num_grid_cols,
        detector_settings_.gridded_detector_num_grid_rows,
        detector_settings_.gridded_detector_num_threads_per_image,
        &keypoints_cv);  // 通过网格检测特征点
  } else {
    detector_->detect(image, keypoints_cv);  // 通过ORB检测特征点

    if (detector_settings_.detector_use_nonmaxsuppression) {
      timing::Timer timer_nms("non-maximum suppression");
      localNonMaximumSuppression(
          camera_.imageHeight(),
          detector_settings_.detector_nonmaxsuppression_radius,
          detector_settings_.detector_nonmaxsuppression_ratio_threshold,
          &keypoints_cv);  // 通过非极大值抑制

      statistics::StatsCollector stat_nms(
          "non-maximum suppression (1 image) in ms");  // 统计
      stat_nms.AddSample(timer_nms.Stop() * 1000);     // 计时器
    }
    cv::KeyPointsFilter::retainBest(
        keypoints_cv,
        detector_settings_.max_feature_count);  // 保留最好的特征点
  }

  // The ORB detector tries to always return a constant number of keypoints.
  // If we get into an environment with very few good keypoint candidates
  // the detector adapts it's score such that it even detects keypoints that
  // are dust on the camera. Therefore, we put a lower bound on the threshold.
  size_t num_removed_keypoints = keypoints_cv.size();

  std::vector<cv::KeyPoint>::iterator it_erase_from = std::remove_if(
      keypoints_cv.begin(), keypoints_cv.end(), [this](const cv::KeyPoint& kp) {
        return kp.response <= detector_settings_.orb_detector_score_lower_bound;
      });  // 移除低分数的特征点
  keypoints_cv.erase(it_erase_from, keypoints_cv.end());

  num_removed_keypoints -=
      keypoints_cv.size();  // 根据keypoints_cv所得到的移除的特征点数量
  VLOG(4) << "Number of removed low score keypoints: " << num_removed_keypoints;

  if (extractor_settings_.flip_descriptor) {
    for (cv::KeyPoint& keypoint : keypoints_cv) {
      keypoint.angle = 180.0f;
    }
  } else {
    // We are doing this because:
    // - Stefan Leutenegger's BRISK implementation uses orientation information
    //   (in case of enabled rotational invariance) of OpenCV keypoints if the
    //   angles are not -1 and the ORB detector assigns orientations to
    //   keypoints.
    //   I don't know which orientation assignment is more robust so feel free
    //   to remove this if you know better. I think there is no significant
    //   performance difference.
    // - Also remove this if you use descriptors (e.g. OpenCV BRISK or ORB)
    //   that rely on orientation information provided by some detectors
    //   (e.g. ORB detector).
    for (cv::KeyPoint& keypoint : keypoints_cv) {
      keypoint.angle = -1.0f;  // 将特征点的角度设为-1
    }
  }
  timer_detection.Stop();

  timing::Timer timer_extraction("descriptor extraction");

  // Compute the descriptors.
  cv::Mat descriptors_cv;
  if (!keypoints_cv.empty()) {
    extractor_->compute(
        frame->getRawImage(), keypoints_cv,
        descriptors_cv);  // 计算特征点的描述子
  } else {
    descriptors_cv = cv::Mat(0, 0, CV_8UC1);
  }

  timer_extraction.Stop();

  // Note: It is important that the values are set even if there are no
  // keypoints as downstream code may rely on the keypoints being set.
  aslam::insertCvKeypointsAndDescriptorsIntoEmptyVisualFrame(
      keypoints_cv, descriptors_cv, detector_settings_.keypoint_uncertainty_px,
      frame);  // 将特征点和描述子插入到视觉帧中

  CHECK(frame->hasKeypointMeasurements());  // 检查视觉帧是否有特征点测量
}
}  // namespace feature_tracking
