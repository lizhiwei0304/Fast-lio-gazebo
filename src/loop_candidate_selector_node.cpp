#include "keyframe_cloud_scorer.h"

#include <dislam_msgs/LoopCandidate.h>
#include <dislam_msgs/LoopCandidates.h>
#include <dislam_msgs/SubMap.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Geometry>

#include <boost/array.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::Pose& pose)
{
  Eigen::Quaterniond q(pose.orientation.w,
                       pose.orientation.x,
                       pose.orientation.y,
                       pose.orientation.z);
  if (q.norm() < 1e-12)
  {
    q = Eigen::Quaterniond::Identity();
  }
  else
  {
    q.normalize();
  }

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = Eigen::Vector3d(pose.position.x,
                                    pose.position.y,
                                    pose.position.z);
  return T;
}

Eigen::Matrix<double, 6, 6> covarianceMsgToEigen(
    const boost::array<double, 36>& covariance)
{
  Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
  for (int r = 0; r < 6; ++r)
  {
    for (int c = 0; c < 6; ++c)
    {
      cov(r, c) = covariance[r * 6 + c];
    }
  }
  return cov;
}

template <typename T>
T getParam(const ros::NodeHandle& nh, const std::string& name, const T& default_value)
{
  T value = default_value;
  nh.param<T>(name, value, default_value);
  return value;
}

std::string toLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

KeyframeCloudScorer::SelectionMode parseSelectionMode(const std::string& mode)
{
  const std::string m = toLower(mode);
  if (m == "geo" || m == "geometry")
  {
    return KeyframeCloudScorer::SelectionMode::GEO;
  }
  if (m == "pose" || m == "geo_pose" || m == "geo-pose" ||
      m == "geometry_pose")
  {
    return KeyframeCloudScorer::SelectionMode::GEO_POSE;
  }
  return KeyframeCloudScorer::SelectionMode::PROPOSED;
}

double elapsedMilliseconds(const std::chrono::steady_clock::time_point& start,
                           const std::chrono::steady_clock::time_point& end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

ros::Time subtractDurationClampedToZero(const ros::Time& stamp, double seconds)
{
  if (stamp.toSec() <= seconds)
  {
    return ros::Time(0);
  }

  return stamp - ros::Duration(seconds);
}
}  // namespace

class LoopCandidateSelector
{
public:
  LoopCandidateSelector(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh),
        pnh_(pnh),
        scorer_(loadScorerParams(pnh))
  {
    submap_topic_ = getParam<std::string>(pnh_, "submap_topic", "submap");
    state_estimation_topic_ =
        getParam<std::string>(pnh_, "state_estimation_topic", "state_estimation");
    candidate_topic_ =
        getParam<std::string>(pnh_, "candidate_topic", "loop_candidate");
    candidate_array_topic_ =
        getParam<std::string>(pnh_, "candidate_array_topic", "loop_candidates");
    candidate_pose_topic_ =
        getParam<std::string>(pnh_, "candidate_pose_topic", "loop_candidate_poses");
    candidate_score_cloud_topic_ =
        getParam<std::string>(pnh_, "candidate_score_cloud_topic", "loop_candidate_score_cloud");

    frame_id_ = getParam<std::string>(pnh_, "frame_id", "map");
    default_robot_id_ = getParam<int>(pnh_, "default_robot_id", 0);
    nms_distance_ = getParam<double>(pnh_, "nms_distance", 3.0);
    max_candidate_num_ = getParam<int>(pnh_, "max_candidate_num", 0);
    min_keyframes_before_selection_ =
        getParam<int>(pnh_, "min_keyframes_before_selection", 1);
    state_estimation_cache_duration_ =
        std::max(0.1, getParam<double>(pnh_, "state_estimation_cache_duration", 30.0));
    state_estimation_max_time_diff_ =
        std::max(0.0, getParam<double>(pnh_, "state_estimation_max_time_diff", 0.05));
    require_state_estimation_pose_ =
        getParam<bool>(pnh_, "require_state_estimation_pose", true);
    publish_all_candidates_each_update_ =
        getParam<bool>(pnh_, "publish_all_candidates_each_update", false);
    enable_timing_log_ =
        getParam<bool>(pnh_, "enable_timing_log", true);
    timing_log_path_ =
        getParam<std::string>(pnh_, "timing_log_path",
                              "loop_candidate_selector_timing.csv");

    selection_mode_name_ =
        getParam<std::string>(pnh_, "selection_mode", "proposed");
    selection_mode_ = parseSelectionMode(selection_mode_name_);

    submap_sub_ = nh_.subscribe(submap_topic_, 20,
                                &LoopCandidateSelector::submapCallback, this);
    state_estimation_sub_ =
        nh_.subscribe(state_estimation_topic_, 200,
                      &LoopCandidateSelector::stateEstimationCallback, this);
    candidate_pub_ =
        nh_.advertise<dislam_msgs::LoopCandidate>(candidate_topic_, 20);
    candidate_array_pub_ =
        nh_.advertise<dislam_msgs::LoopCandidates>(candidate_array_topic_, 1, true);
    candidate_pose_pub_ =
        nh_.advertise<geometry_msgs::PoseArray>(candidate_pose_topic_, 1, true);
    candidate_score_cloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>(candidate_score_cloud_topic_, 1, true);

    ROS_INFO_STREAM("loop_candidate_selector subscribe: " << submap_topic_);
    ROS_INFO_STREAM("loop_candidate_selector state estimation: "
                    << state_estimation_topic_
                    << " max_dt=" << state_estimation_max_time_diff_ << "s");
    ROS_INFO_STREAM("loop_candidate_selector candidate: " << candidate_topic_);
    ROS_INFO_STREAM("loop_candidate_selector candidate array: "
                    << candidate_array_topic_);

    initializeTimingLog();
  }

private:
  struct TimedStatePose
  {
    ros::Time stamp;
    geometry_msgs::Pose pose;
  };

  static KeyframeCloudScorer::Params loadScorerParams(const ros::NodeHandle& pnh)
  {
    KeyframeCloudScorer::Params params;

    // 与新版 KeyframeCloudScorer_latest_no_2_3.hpp 保持一致。
    // 原独立 2.3 空间分布评分已经删除，空间相关信息已并入 2.1 q_geo。
    params.voxel_size = getParam<double>(pnh, "voxel_size", params.voxel_size);
    params.min_points_per_voxel =
        getParam<int>(pnh, "min_points_per_voxel", params.min_points_per_voxel);
    params.min_points_cloud =
        getParam<int>(pnh, "min_points_cloud", params.min_points_cloud);

    // 2.1 几何结构质量：q_count、q_structure、q_distribution、q_vertical。
    // 为兼容旧 launch，eta_valid_voxel 仍作为 eta_voxel_count 的别名读取。
    params.eta_voxel_count = getParam<double>(
        pnh, "eta_voxel_count",
        getParam<double>(pnh, "eta_valid_voxel", params.eta_voxel_count));
    params.alpha_geo_structure =
        getParam<double>(pnh, "alpha_geo_structure", params.alpha_geo_structure);
    params.alpha_geo_distribution =
        getParam<double>(pnh, "alpha_geo_distribution", params.alpha_geo_distribution);
    params.alpha_geo_vertical =
        getParam<double>(pnh, "alpha_geo_vertical", params.alpha_geo_vertical);
    params.eta_vertical =
        getParam<double>(pnh, "eta_vertical", params.eta_vertical);
    params.min_horizontal_norm =
        getParam<double>(pnh, "min_horizontal_norm", params.min_horizontal_norm);
    params.cloud_in_world_frame =
        getParam<bool>(pnh, "cloud_in_world_frame", params.cloud_in_world_frame);

    // 2.2 位姿约束可靠性。
    params.use_pose_reliability =
        getParam<bool>(pnh, "use_pose_reliability", params.use_pose_reliability);
    params.eta_pose = getParam<double>(pnh, "eta_pose", params.eta_pose);
    params.alpha_pose_self =
        getParam<double>(pnh, "alpha_pose_self", params.alpha_pose_self);

    const double covariance_trans_weight =
        getParam<double>(pnh, "covariance_trans_weight", 1.0);
    const double covariance_rot_weight =
        getParam<double>(pnh, "covariance_rot_weight", 0.2);
    params.covariance_weight.setZero();
    for (int i = 0; i < 3; ++i)
    {
      params.covariance_weight(i, i) = covariance_trans_weight;
      params.covariance_weight(i + 3, i + 3) = covariance_rot_weight;
    }

    // 2.3 轨迹连续性。注意：这里的 2.3 是删除空间分布后的新编号。
    params.neighbor_radius =
        getParam<int>(pnh, "neighbor_radius", params.neighbor_radius);
    params.max_keyframe_gap = getParam<double>(
        pnh, "max_keyframe_gap",
        getParam<double>(pnh, "trans_continuity_thresh", params.max_keyframe_gap));
    params.eta_turn = getParam<double>(pnh, "eta_turn", params.eta_turn);

    // 综合评分：只包含 q_geo、q_pose、q_continuity。
    params.w_geo = getParam<double>(pnh, "w_geo", params.w_geo);
    params.w_pose = getParam<double>(pnh, "w_pose", params.w_pose);
    params.w_continuity = getParam<double>(
        pnh, "w_continuity",
        getParam<double>(pnh, "w_con", params.w_continuity));

    // 阈值与 Python 的 tau_geo、tau_pose、tau_total 对齐。
    // 兼容旧参数名 min_geo_score、min_pose_score、min_total_score。
    params.tau_geo = getParam<double>(
        pnh, "tau_geo",
        getParam<double>(pnh, "min_geo_score", params.tau_geo));
    params.tau_pose = getParam<double>(
        pnh, "tau_pose",
        getParam<double>(pnh, "min_pose_score", params.tau_pose));
    params.tau_total = getParam<double>(
        pnh, "tau_total",
        getParam<double>(pnh, "min_total_score", params.tau_total));

    return params;
  }

  void stateEstimationCallback(const nav_msgs::OdometryConstPtr& msg)
  {
    if (msg->header.stamp.isZero())
    {
      ROS_WARN_THROTTLE(2.0, "loop_candidate_selector received state_estimation with zero stamp");
      return;
    }

    TimedStatePose item;
    item.stamp = msg->header.stamp;
    item.pose = msg->pose.pose;

    auto it = std::lower_bound(
        state_estimation_buffer_.begin(),
        state_estimation_buffer_.end(),
        item.stamp,
        [](const TimedStatePose& lhs, const ros::Time& rhs) {
          return lhs.stamp < rhs;
        });

    if (it != state_estimation_buffer_.end() && it->stamp == item.stamp)
    {
      *it = item;
    }
    else
    {
      state_estimation_buffer_.insert(it, item);
    }

    pruneStateEstimationBuffer(item.stamp);
  }

  void pruneStateEstimationBuffer(const ros::Time& newest_stamp)
  {
    const ros::Time min_stamp = subtractDurationClampedToZero(
        newest_stamp, state_estimation_cache_duration_);
    while (!state_estimation_buffer_.empty() &&
           state_estimation_buffer_.front().stamp < min_stamp)
    {
      state_estimation_buffer_.pop_front();
    }
  }

  bool lookupStateEstimationPose(const ros::Time& stamp,
                                 geometry_msgs::Pose& pose,
                                 double& best_time_diff) const
  {
    best_time_diff = std::numeric_limits<double>::infinity();
    if (stamp.isZero() || state_estimation_buffer_.empty())
    {
      return false;
    }

    const TimedStatePose* best = nullptr;
    auto it = std::lower_bound(
        state_estimation_buffer_.begin(),
        state_estimation_buffer_.end(),
        stamp,
        [](const TimedStatePose& lhs, const ros::Time& rhs) {
          return lhs.stamp < rhs;
        });

    auto consider = [&](std::deque<TimedStatePose>::const_iterator candidate) {
      const double dt = std::abs((candidate->stamp - stamp).toSec());
      if (dt < best_time_diff)
      {
        best_time_diff = dt;
        best = &(*candidate);
      }
    };

    if (it != state_estimation_buffer_.end())
    {
      consider(it);
    }
    if (it != state_estimation_buffer_.begin())
    {
      consider(std::prev(it));
    }

    if (best == nullptr || best_time_diff > state_estimation_max_time_diff_)
    {
      return false;
    }

    pose = best->pose;
    return true;
  }

  void submapCallback(const dislam_msgs::SubMapConstPtr& msg)
  {
    KeyframeCloudScorer::CloudT::Ptr cloud(new KeyframeCloudScorer::CloudT);
    pcl::fromROSMsg(msg->keyframePC, *cloud);
    if (!cloud || cloud->empty())
    {
      ROS_WARN_THROTTLE(2.0, "loop_candidate_selector received an empty submap cloud");
      return;
    }

    const int index = static_cast<int>(keyframes_.size());
    const int source_id =
        msg->id == 0
            ? index
            : static_cast<int>(std::min<uint32_t>(
                  msg->id, static_cast<uint32_t>(std::numeric_limits<int>::max())));
    const ros::Time submap_stamp = msg->keyframePC.header.stamp;

    geometry_msgs::PoseWithCovariance keyframe_pose = msg->pose;
    geometry_msgs::Pose state_pose;
    double state_time_diff = std::numeric_limits<double>::infinity();
    if (lookupStateEstimationPose(submap_stamp, state_pose, state_time_diff))
    {
      keyframe_pose.pose = state_pose;
    }
    else
    {
      ROS_WARN_STREAM_THROTTLE(
          2.0,
          "loop_candidate_selector cannot match state_estimation for submap stamp "
              << submap_stamp.toSec()
              << ", buffer_size=" << state_estimation_buffer_.size()
              << ", best_dt="
              << (std::isfinite(state_time_diff) ? state_time_diff : -1.0)
              << "s, max_dt=" << state_estimation_max_time_diff_ << "s");
      if (require_state_estimation_pose_)
      {
        return;
      }
    }

    KeyframeCloudScorer::KeyFrame kf;
    kf.robot_id = msg->robot_id != 0 ? msg->robot_id : default_robot_id_;
    kf.index = source_id;
    kf.pose = poseMsgToIsometry(keyframe_pose.pose);
    kf.covariance = covarianceMsgToEigen(msg->pose.covariance);
    kf.cloud = cloud;

    keyframes_.emplace_back(kf);
    dislam_msgs::SubMap submap = *msg;
    submap.pose = keyframe_pose;
    submaps_.emplace_back(submap);

    if (static_cast<int>(keyframes_.size()) < min_keyframes_before_selection_)
    {
      publishCandidateViews(std::vector<int>(), std::vector<KeyframeCloudScorer::Score>());
      return;
    }

    const auto selection_start = std::chrono::steady_clock::now();
    const std::vector<KeyframeCloudScorer::Score> scores =
        scorer_.evaluateAll(keyframes_);
    const auto evaluate_done = std::chrono::steady_clock::now();
    const std::vector<int> selected =
        scorer_.selectCandidatesByMode(
            keyframes_, scores, nms_distance_, max_candidate_num_, selection_mode_);
    const auto selection_done = std::chrono::steady_clock::now();

    logSelectionTiming(ros::Time::now(),
                       kf.robot_id,
                       kf.index,
                       static_cast<int>(keyframes_.size()),
                       static_cast<int>(selected.size()),
                       elapsedMilliseconds(selection_start, evaluate_done),
                       elapsedMilliseconds(evaluate_done, selection_done),
                       elapsedMilliseconds(selection_start, selection_done));

    publishCandidateViews(selected, scores);
    publishCandidates(selected, scores);
  }

  void initializeTimingLog()
  {
    if (!enable_timing_log_)
    {
      return;
    }

    if (timing_log_path_.empty())
    {
      ROS_WARN("loop_candidate_selector timing log is enabled but timing_log_path is empty");
      enable_timing_log_ = false;
      return;
    }

    bool write_header = true;
    {
      std::ifstream existing(timing_log_path_.c_str(),
                             std::ios::in | std::ios::binary | std::ios::ate);
      if (existing.good() && existing.tellg() > 0)
      {
        write_header = false;
      }
    }

    timing_log_stream_.open(timing_log_path_.c_str(), std::ios::out | std::ios::app);
    if (!timing_log_stream_.is_open())
    {
      ROS_ERROR_STREAM("failed to open loop_candidate_selector timing log: "
                       << timing_log_path_);
      enable_timing_log_ = false;
      return;
    }

    if (write_header)
    {
      timing_log_stream_
          << "ros_time,robot_id,submap_id,keyframe_count,selected_count,"
          << "selection_mode,evaluate_ms,nms_ms,total_ms\n";
      timing_log_stream_.flush();
    }

    ROS_INFO_STREAM("loop_candidate_selector timing log: " << timing_log_path_);
  }

  void logSelectionTiming(const ros::Time& stamp,
                          int robot_id,
                          int submap_id,
                          int keyframe_count,
                          int selected_count,
                          double evaluate_ms,
                          double nms_ms,
                          double total_ms)
  {
    if (!enable_timing_log_ || !timing_log_stream_.is_open())
    {
      return;
    }

    timing_log_stream_ << std::fixed << std::setprecision(9)
                       << stamp.toSec() << ','
                       << robot_id << ','
                       << submap_id << ','
                       << keyframe_count << ','
                       << selected_count << ','
                       << selection_mode_name_ << ','
                       << std::setprecision(3)
                       << evaluate_ms << ','
                       << nms_ms << ','
                       << total_ms << '\n';
    timing_log_stream_.flush();
  }

  dislam_msgs::LoopCandidate makeCandidateMessage(
      int idx,
      const KeyframeCloudScorer::Score& score,
      const ros::Time& stamp) const
  {
    dislam_msgs::LoopCandidate out;
    out.header.stamp = stamp;
    out.header.frame_id = frame_id_;
    out.robot_id = static_cast<uint8_t>(keyframes_[idx].robot_id);
    out.id = static_cast<uint32_t>(keyframes_[idx].index);
    out.keyframePC = submaps_[idx].keyframePC;
    out.orthoImage = submaps_[idx].orthoImage;
    out.pose = submaps_[idx].pose;
    out.describe = submaps_[idx].describe;
    out.q_geo = score.q_geo;
    out.q_pose = score.q_pose;

    // 与旧 LoopCandidate.msg 兼容：
    //   q_spa 原为空间分布评分，最新算法已删除，固定置 0。
    //   q_con 仍使用轨迹连续性 q_continuity。
    out.q_spa = 0.0;
    out.q_con = score.q_continuity;
    out.q_total = score.q_total;

    // 与旧字段兼容输出几何调试量。
    out.valid_voxel_num = static_cast<uint32_t>(score.geo_valid_voxels);
    out.valid_sector_num = static_cast<uint32_t>(score.geo_direction_voxels);
    out.height_range = score.geo_z_range;
    out.max_gap_angle = 0.0;

    // 当前消息已经通过 selection_mode_ 对应阈值与 NMS 筛选，
    // 因此发布出去的 LoopCandidate 均标记为候选。
    out.is_candidate = true;
    return out;
  }

  void publishCandidates(
      const std::vector<int>& selected,
      const std::vector<KeyframeCloudScorer::Score>& scores)
  {
    const ros::Time stamp = ros::Time::now();

    dislam_msgs::LoopCandidates array_msg;
    array_msg.header.stamp = stamp;
    array_msg.header.frame_id = frame_id_;
    array_msg.candidates.reserve(selected.size());

    for (int idx : selected)
    {
      const dislam_msgs::LoopCandidate out =
          makeCandidateMessage(idx, scores[idx], stamp);
      array_msg.candidates.emplace_back(out);

      const bool already_published =
          published_candidate_indices_.count(idx) > 0;
      if (publish_all_candidates_each_update_ || !already_published)
      {
        candidate_pub_.publish(out);
        published_candidate_indices_.insert(idx);

        ROS_INFO_STREAM("loop candidate selected robot=" << keyframes_[idx].robot_id
                        << " id=" << keyframes_[idx].index
                        << " total=" << scores[idx].q_total
                        << " geo=" << scores[idx].q_geo
                        << " pose=" << scores[idx].q_pose
                        << " cont=" << scores[idx].q_continuity
                        << " geo_count=" << scores[idx].q_geo_count
                        << " geo_struct=" << scores[idx].q_geo_structure
                        << " geo_dir=" << scores[idx].q_geo_distribution
                        << " geo_z=" << scores[idx].q_geo_vertical);
      }
    }

    candidate_array_pub_.publish(array_msg);
  }

  void publishCandidateViews(
      const std::vector<int>& selected,
      const std::vector<KeyframeCloudScorer::Score>& scores)
  {
    geometry_msgs::PoseArray pose_array;
    pose_array.header.stamp = ros::Time::now();
    pose_array.header.frame_id = frame_id_;
    pose_array.poses.reserve(selected.size());

    pcl::PointCloud<pcl::PointXYZI> score_cloud;
    score_cloud.header.frame_id = frame_id_;
    score_cloud.header.stamp = 0;
    score_cloud.height = 1;
    score_cloud.is_dense = false;
    score_cloud.points.reserve(selected.size());

    for (int idx : selected)
    {
      pose_array.poses.emplace_back(submaps_[idx].pose.pose);

      pcl::PointXYZI pt;
      const Eigen::Vector3d t = keyframes_[idx].pose.translation();
      pt.x = static_cast<float>(t.x());
      pt.y = static_cast<float>(t.y());
      pt.z = static_cast<float>(t.z());
      pt.intensity =
          scores.empty() ? 0.0f : static_cast<float>(scores[idx].q_total);
      score_cloud.points.emplace_back(pt);
    }

    score_cloud.width = static_cast<uint32_t>(score_cloud.points.size());

    sensor_msgs::PointCloud2 score_cloud_msg;
    pcl::toROSMsg(score_cloud, score_cloud_msg);
    score_cloud_msg.header.stamp = pose_array.header.stamp;
    score_cloud_msg.header.frame_id = frame_id_;

    candidate_pose_pub_.publish(pose_array);
    candidate_score_cloud_pub_.publish(score_cloud_msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber submap_sub_;
  ros::Subscriber state_estimation_sub_;
  ros::Publisher candidate_pub_;
  ros::Publisher candidate_array_pub_;
  ros::Publisher candidate_pose_pub_;
  ros::Publisher candidate_score_cloud_pub_;

  KeyframeCloudScorer scorer_;
  KeyframeCloudScorer::SelectionMode selection_mode_ =
      KeyframeCloudScorer::SelectionMode::PROPOSED;
  KeyframeCloudScorer::KeyFrameVector keyframes_;
  std::deque<TimedStatePose> state_estimation_buffer_;
  std::vector<dislam_msgs::SubMap> submaps_;
  std::set<int> published_candidate_indices_;

  std::string submap_topic_;
  std::string state_estimation_topic_;
  std::string candidate_topic_;
  std::string candidate_array_topic_;
  std::string candidate_pose_topic_;
  std::string candidate_score_cloud_topic_;
  std::string selection_mode_name_ = "proposed";
  std::string timing_log_path_;
  std::string frame_id_;
  int default_robot_id_ = 0;
  double nms_distance_ = 3.0;
  double state_estimation_cache_duration_ = 30.0;
  double state_estimation_max_time_diff_ = 0.05;
  int max_candidate_num_ = 0;
  int min_keyframes_before_selection_ = 1;
  bool require_state_estimation_pose_ = true;
  bool publish_all_candidates_each_update_ = false;
  bool enable_timing_log_ = true;
  std::ofstream timing_log_stream_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "loop_candidate_selector");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  LoopCandidateSelector selector(nh, pnh);
  ros::spin();
  return 0;
}
