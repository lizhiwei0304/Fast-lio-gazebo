#include "keyframe_cloud_scorer.h"

#include <dislam_msgs/LoopCandidate.h>
#include <dislam_msgs/LoopCandidates.h>
#include <dislam_msgs/SubMap.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Geometry>

#include <boost/array.hpp>

#include <algorithm>
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
    publish_all_candidates_each_update_ =
        getParam<bool>(pnh_, "publish_all_candidates_each_update", false);

    submap_sub_ = nh_.subscribe(submap_topic_, 20,
                                &LoopCandidateSelector::submapCallback, this);
    candidate_pub_ =
        nh_.advertise<dislam_msgs::LoopCandidate>(candidate_topic_, 20);
    candidate_array_pub_ =
        nh_.advertise<dislam_msgs::LoopCandidates>(candidate_array_topic_, 1, true);
    candidate_pose_pub_ =
        nh_.advertise<geometry_msgs::PoseArray>(candidate_pose_topic_, 1, true);
    candidate_score_cloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>(candidate_score_cloud_topic_, 1, true);

    ROS_INFO_STREAM("loop_candidate_selector subscribe: " << submap_topic_);
    ROS_INFO_STREAM("loop_candidate_selector candidate: " << candidate_topic_);
    ROS_INFO_STREAM("loop_candidate_selector candidate array: "
                    << candidate_array_topic_);
  }

private:
  static KeyframeCloudScorer::Params loadScorerParams(const ros::NodeHandle& pnh)
  {
    KeyframeCloudScorer::Params params;

    params.voxel_size = getParam<double>(pnh, "voxel_size", params.voxel_size);
    params.min_points_per_voxel =
        getParam<int>(pnh, "min_points_per_voxel", params.min_points_per_voxel);
    params.w_line = getParam<double>(pnh, "w_line", params.w_line);
    params.w_plane = getParam<double>(pnh, "w_plane", params.w_plane);
    params.w_scatter = getParam<double>(pnh, "w_scatter", params.w_scatter);
    params.eta_valid_voxel =
        getParam<double>(pnh, "eta_valid_voxel", params.eta_valid_voxel);

    params.use_pose_reliability =
        getParam<bool>(pnh, "use_pose_reliability", params.use_pose_reliability);
    params.eta_pose_self =
        getParam<double>(pnh, "eta_pose_self", params.eta_pose_self);
    params.eta_pose_neighbor =
        getParam<double>(pnh, "eta_pose_neighbor", params.eta_pose_neighbor);
    params.alpha_pose_self =
        getParam<double>(pnh, "alpha_pose_self", params.alpha_pose_self);

    const double covariance_trans_weight =
        getParam<double>(pnh, "covariance_trans_weight", 1.0);
    const double covariance_rot_weight =
        getParam<double>(pnh, "covariance_rot_weight", 1.0);
    params.covariance_weight.setZero();
    for (int i = 0; i < 3; ++i)
    {
      params.covariance_weight(i, i) = covariance_trans_weight;
      params.covariance_weight(i + 3, i + 3) = covariance_rot_weight;
    }

    params.sector_num = getParam<int>(pnh, "sector_num", params.sector_num);
    params.min_points_per_sector =
        getParam<int>(pnh, "min_points_per_sector", params.min_points_per_sector);
    params.w_azimuth_cover =
        getParam<double>(pnh, "w_azimuth_cover", params.w_azimuth_cover);
    params.w_azimuth_uniform =
        getParam<double>(pnh, "w_azimuth_uniform", params.w_azimuth_uniform);
    params.w_max_gap = getParam<double>(pnh, "w_max_gap", params.w_max_gap);
    params.w_height = getParam<double>(pnh, "w_height", params.w_height);
    params.eta_height = getParam<double>(pnh, "eta_height", params.eta_height);
    params.cloud_in_world_frame =
        getParam<bool>(pnh, "cloud_in_world_frame", params.cloud_in_world_frame);

    params.neighbor_radius =
        getParam<int>(pnh, "neighbor_radius", params.neighbor_radius);
    params.trans_continuity_thresh =
        getParam<double>(pnh, "trans_continuity_thresh",
                         params.trans_continuity_thresh);
    const double rot_continuity_thresh_deg =
        getParam<double>(pnh, "rot_continuity_thresh_deg",
                         params.rot_continuity_thresh * 180.0 /
                             3.14159265358979323846);
    params.rot_continuity_thresh =
        rot_continuity_thresh_deg * 3.14159265358979323846 / 180.0;

    params.w_geo = getParam<double>(pnh, "w_geo", params.w_geo);
    params.w_pose = getParam<double>(pnh, "w_pose", params.w_pose);
    params.w_spa = getParam<double>(pnh, "w_spa", params.w_spa);
    params.w_con = getParam<double>(pnh, "w_con", params.w_con);

    params.min_geo_score =
        getParam<double>(pnh, "min_geo_score", params.min_geo_score);
    params.min_pose_score =
        getParam<double>(pnh, "min_pose_score", params.min_pose_score);
    params.min_spa_score =
        getParam<double>(pnh, "min_spa_score", params.min_spa_score);
    params.min_con_score =
        getParam<double>(pnh, "min_con_score", params.min_con_score);
    params.min_total_score =
        getParam<double>(pnh, "min_total_score", params.min_total_score);

    return params;
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
    KeyframeCloudScorer::KeyFrame kf;
    kf.robot_id = msg->robot_id != 0 ? msg->robot_id : default_robot_id_;
    kf.index = index;
    kf.pose = poseMsgToIsometry(msg->pose.pose);
    kf.covariance = covarianceMsgToEigen(msg->pose.covariance);
    kf.cloud = cloud;

    keyframes_.emplace_back(kf);
    submaps_.emplace_back(*msg);

    if (static_cast<int>(keyframes_.size()) < min_keyframes_before_selection_)
    {
      publishCandidateViews(std::vector<int>(), std::vector<KeyframeCloudScorer::Score>());
      return;
    }

    const std::vector<KeyframeCloudScorer::Score> scores =
        scorer_.evaluateAll(keyframes_);
    const std::vector<int> selected =
        scorer_.selectCandidates(keyframes_, scores, nms_distance_, max_candidate_num_);

    for (int idx : selected)
    {
      global_candidate_indices_.insert(idx);
    }

    const std::vector<int> global_candidates(global_candidate_indices_.begin(),
                                             global_candidate_indices_.end());
    publishCandidateViews(global_candidates, scores);
    publishCandidates(global_candidates, scores);
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
    out.q_spa = score.q_spa;
    out.q_con = score.q_con;
    out.q_total = score.q_total;
    out.valid_voxel_num = score.valid_voxel_num;
    out.valid_sector_num = score.valid_sector_num;
    out.height_range = score.height_range;
    out.max_gap_angle = score.max_gap_angle;
    out.is_candidate = score.is_candidate;
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
                        << " spa=" << scores[idx].q_spa
                        << " con=" << scores[idx].q_con);
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
  ros::Publisher candidate_pub_;
  ros::Publisher candidate_array_pub_;
  ros::Publisher candidate_pose_pub_;
  ros::Publisher candidate_score_cloud_pub_;

  KeyframeCloudScorer scorer_;
  std::vector<KeyframeCloudScorer::KeyFrame> keyframes_;
  std::vector<dislam_msgs::SubMap> submaps_;
  std::set<int> global_candidate_indices_;
  std::set<int> published_candidate_indices_;

  std::string submap_topic_;
  std::string candidate_topic_;
  std::string candidate_array_topic_;
  std::string candidate_pose_topic_;
  std::string candidate_score_cloud_topic_;
  std::string frame_id_;
  int default_robot_id_ = 0;
  double nms_distance_ = 3.0;
  int max_candidate_num_ = 0;
  int min_keyframes_before_selection_ = 1;
  bool publish_all_candidates_each_update_ = false;
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
