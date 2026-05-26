#pragma once

#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

class KeyframeCloudScorer
{
public:
  using PointT = pcl::PointXYZI;
  using CloudT = pcl::PointCloud<PointT>;

  struct KeyFrame
  {
    int robot_id = 0;
    int index = 0;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Zero();
    CloudT::Ptr cloud;
  };

  struct Params
  {
    double voxel_size = 1.0;
    int min_points_per_voxel = 8;
    double w_line = 0.35;
    double w_plane = 0.45;
    double w_scatter = 0.20;
    double eta_valid_voxel = 0.01;

    bool use_pose_reliability = true;
    double eta_pose_self = 1.0;
    double eta_pose_neighbor = 1.0;
    double alpha_pose_self = 0.6;
    Eigen::Matrix<double, 6, 6> covariance_weight =
        Eigen::Matrix<double, 6, 6>::Identity();

    int sector_num = 36;
    int min_points_per_sector = 20;
    double w_azimuth_cover = 0.30;
    double w_azimuth_uniform = 0.30;
    double w_max_gap = 0.20;
    double w_height = 0.20;
    double eta_height = 1.0;
    bool cloud_in_world_frame = false;

    int neighbor_radius = 3;
    double trans_continuity_thresh = 2.0;
    double rot_continuity_thresh = 20.0 * 3.14159265358979323846 / 180.0;

    double w_geo = 0.35;
    double w_pose = 0.20;
    double w_spa = 0.30;
    double w_con = 0.15;

    double min_geo_score = 0.25;
    double min_pose_score = 0.20;
    double min_spa_score = 0.25;
    double min_con_score = 0.50;
    double min_total_score = 0.45;
  };

  struct Score
  {
    double q_geo = 0.0;
    double q_pose = 0.0;
    double q_spa = 0.0;
    double q_con = 0.0;
    double q_total = 0.0;
    int valid_voxel_num = 0;
    int valid_sector_num = 0;
    double height_range = 0.0;
    double max_gap_angle = 0.0;
    bool is_candidate = false;
  };

  KeyframeCloudScorer();
  explicit KeyframeCloudScorer(const Params& params);

  Score evaluateOne(const std::vector<KeyFrame>& keyframes, int target_idx) const
  {
    Score score;
    if (target_idx < 0 || target_idx >= static_cast<int>(keyframes.size()))
    {
      return score;
    }

    const KeyFrame& kf = keyframes[target_idx];
    if (!kf.cloud || kf.cloud->empty())
    {
      return score;
    }

    score.q_geo = computeGeometryScore(kf, score.valid_voxel_num);
    score.q_pose = computePoseReliabilityScore(keyframes, target_idx);
    score.q_spa = computeSpatialDistributionScore(
        kf, score.valid_sector_num, score.height_range, score.max_gap_angle);
    score.q_con = computeTrajectoryContinuityScore(keyframes, target_idx);

    score.q_total =
        params_.w_geo * score.q_geo +
        params_.w_pose * score.q_pose +
        params_.w_spa * score.q_spa +
        params_.w_con * score.q_con;
    score.q_total = clamp01(score.q_total);

    score.is_candidate =
        score.q_geo >= params_.min_geo_score &&
        score.q_pose >= params_.min_pose_score &&
        score.q_spa >= params_.min_spa_score &&
        score.q_con >= params_.min_con_score &&
        score.q_total >= params_.min_total_score;

    return score;
  }

  std::vector<Score> evaluateAll(const std::vector<KeyFrame>& keyframes) const
  {
    std::vector<Score> scores;
    scores.reserve(keyframes.size());
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i)
    {
      scores.emplace_back(evaluateOne(keyframes, i));
    }
    return scores;
  }

  std::vector<int> selectCandidates(const std::vector<KeyFrame>& keyframes,
                                    const std::vector<Score>& scores,
                                    double nms_distance,
                                    int max_candidate_num = 0) const
  {
    std::vector<int> indices;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i)
    {
      if (scores[i].is_candidate)
      {
        indices.emplace_back(i);
      }
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return scores[a].q_total > scores[b].q_total;
    });

    std::vector<int> selected;
    const double min_dist = std::max(0.0, nms_distance);
    for (int idx : indices)
    {
      const Eigen::Vector3d p = keyframes[idx].pose.translation();
      bool keep = true;
      for (int kept : selected)
      {
        const Eigen::Vector3d q = keyframes[kept].pose.translation();
        if ((p - q).norm() < min_dist)
        {
          keep = false;
          break;
        }
      }

      if (keep)
      {
        selected.emplace_back(idx);
        if (max_candidate_num > 0 &&
            static_cast<int>(selected.size()) >= max_candidate_num)
        {
          break;
        }
      }
    }
    return selected;
  }

private:
  struct VoxelKey
  {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey& k) const
    {
      const std::size_t h1 = std::hash<int>()(k.x);
      const std::size_t h2 = std::hash<int>()(k.y);
      const std::size_t h3 = std::hash<int>()(k.z);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  double computeGeometryScore(const KeyFrame& kf, int& valid_voxel_num) const
  {
    valid_voxel_num = 0;
    if (params_.voxel_size <= 1e-6)
    {
      return 0.0;
    }

    std::unordered_map<VoxelKey, std::vector<Eigen::Vector3d>, VoxelKeyHash> voxel_map;
    voxel_map.reserve(kf.cloud->size());

    for (const auto& pt : kf.cloud->points)
    {
      if (!pcl::isFinite(pt))
      {
        continue;
      }

      Eigen::Vector3d p(pt.x, pt.y, pt.z);
      if (params_.cloud_in_world_frame)
      {
        p = kf.pose.inverse() * p;
      }

      VoxelKey key;
      key.x = static_cast<int>(std::floor(p.x() / params_.voxel_size));
      key.y = static_cast<int>(std::floor(p.y() / params_.voxel_size));
      key.z = static_cast<int>(std::floor(p.z() / params_.voxel_size));
      voxel_map[key].emplace_back(p);
    }

    double voxel_score_sum = 0.0;
    for (const auto& item : voxel_map)
    {
      const auto& points = item.second;
      if (static_cast<int>(points.size()) < params_.min_points_per_voxel)
      {
        continue;
      }

      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const auto& p : points)
      {
        mean += p;
      }
      mean /= static_cast<double>(points.size());

      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (const auto& p : points)
      {
        const Eigen::Vector3d d = p - mean;
        cov += d * d.transpose();
      }
      cov /= static_cast<double>(points.size());

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
      if (solver.info() != Eigen::Success)
      {
        continue;
      }

      const Eigen::Vector3d eig = solver.eigenvalues();
      const double l1 = std::max(0.0, eig(2));
      const double l2 = std::max(0.0, eig(1));
      const double l3 = std::max(0.0, eig(0));
      const double denom = l1 + 1e-9;

      const double linearity = clamp01((l1 - l2) / denom);
      const double planarity = clamp01((l2 - l3) / denom);
      const double scattering = clamp01(l3 / denom);

      const double q_voxel =
          params_.w_line * linearity +
          params_.w_plane * planarity +
          params_.w_scatter * scattering;

      voxel_score_sum += clamp01(q_voxel);
      valid_voxel_num++;
    }

    if (valid_voxel_num == 0)
    {
      return 0.0;
    }

    const double mean_voxel_score =
        voxel_score_sum / static_cast<double>(valid_voxel_num);
    const double voxel_num_gain =
        1.0 - std::exp(-params_.eta_valid_voxel *
                       static_cast<double>(valid_voxel_num));

    return clamp01(voxel_num_gain * mean_voxel_score);
  }

  double computePoseReliabilityScore(const std::vector<KeyFrame>& keyframes,
                                     int target_idx) const
  {
    if (!params_.use_pose_reliability)
    {
      return 1.0;
    }

    const int n = static_cast<int>(keyframes.size());
    const KeyFrame& kf = keyframes[target_idx];

    const double u_self = poseUncertainty(kf.covariance);
    const double q_self = clamp01(std::exp(-params_.eta_pose_self * u_self));

    const int left = std::max(0, target_idx - params_.neighbor_radius);
    const int right = std::min(n - 1, target_idx + params_.neighbor_radius);

    double u_sum = 0.0;
    int count = 0;
    for (int i = left; i <= right; ++i)
    {
      u_sum += poseUncertainty(keyframes[i].covariance);
      count++;
    }

    double q_neighbor = q_self;
    if (count > 0)
    {
      const double u_mean = u_sum / static_cast<double>(count);
      q_neighbor = clamp01(std::exp(-params_.eta_pose_neighbor * u_mean));
    }

    return clamp01(params_.alpha_pose_self * q_self +
                   (1.0 - params_.alpha_pose_self) * q_neighbor);
  }

  double computeSpatialDistributionScore(const KeyFrame& kf,
                                         int& valid_sector_num,
                                         double& height_range,
                                         double& max_gap_angle) const
  {
    valid_sector_num = 0;
    height_range = 0.0;
    max_gap_angle = 0.0;

    const int B = std::max(4, params_.sector_num);
    const double two_pi = 2.0 * 3.14159265358979323846;
    std::vector<int> sector_counts(B, 0);
    std::vector<double> z_values;
    z_values.reserve(kf.cloud->size());

    for (const auto& pt : kf.cloud->points)
    {
      if (!pcl::isFinite(pt))
      {
        continue;
      }

      Eigen::Vector3d p(pt.x, pt.y, pt.z);
      if (params_.cloud_in_world_frame)
      {
        p = kf.pose.inverse() * p;
      }

      double angle = std::atan2(p.y(), p.x());
      if (angle < 0.0)
      {
        angle += two_pi;
      }

      int sector_id = static_cast<int>(std::floor(angle / two_pi * B));
      sector_id = std::max(0, std::min(B - 1, sector_id));
      sector_counts[sector_id]++;
      z_values.emplace_back(p.z());
    }

    if (z_values.empty())
    {
      return 0.0;
    }

    std::vector<int> valid_indices;
    valid_indices.reserve(B);
    int total_valid_points = 0;
    for (int i = 0; i < B; ++i)
    {
      if (sector_counts[i] >= params_.min_points_per_sector)
      {
        valid_indices.emplace_back(i);
        total_valid_points += sector_counts[i];
      }
    }

    valid_sector_num = static_cast<int>(valid_indices.size());
    const double q_cover =
        clamp01(static_cast<double>(valid_sector_num) / static_cast<double>(B));

    double q_uniform = 0.0;
    if (valid_sector_num > 1 && total_valid_points > 0)
    {
      double entropy = 0.0;
      for (int idx : valid_indices)
      {
        const double prob = static_cast<double>(sector_counts[idx]) /
                            static_cast<double>(total_valid_points);
        entropy += -prob * std::log(prob + 1e-12);
      }
      q_uniform = clamp01(entropy / std::log(static_cast<double>(valid_sector_num)));
    }

    double q_gap = 0.0;
    if (!valid_indices.empty())
    {
      std::sort(valid_indices.begin(), valid_indices.end());
      int max_empty_bins = 0;
      for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i)
      {
        const int cur = valid_indices[i];
        int next = valid_indices[(i + 1) % valid_indices.size()];
        int diff = next - cur;
        if (diff <= 0)
        {
          diff += B;
        }
        max_empty_bins = std::max(max_empty_bins, diff - 1);
      }

      max_gap_angle =
          static_cast<double>(max_empty_bins) * two_pi / static_cast<double>(B);
      q_gap = clamp01(1.0 - max_gap_angle / two_pi);
    }

    std::sort(z_values.begin(), z_values.end());
    const double z5 = percentile(z_values, 0.05);
    const double z95 = percentile(z_values, 0.95);
    height_range = std::max(0.0, z95 - z5);
    const double q_height =
        clamp01(1.0 - std::exp(-height_range / std::max(1e-6, params_.eta_height)));

    return clamp01(params_.w_azimuth_cover * q_cover +
                   params_.w_azimuth_uniform * q_uniform +
                   params_.w_max_gap * q_gap +
                   params_.w_height * q_height);
  }

  double computeTrajectoryContinuityScore(const std::vector<KeyFrame>& keyframes,
                                          int target_idx) const
  {
    const int n = static_cast<int>(keyframes.size());
    const int left = std::max(0, target_idx - params_.neighbor_radius);
    const int right = std::min(n - 1, target_idx + params_.neighbor_radius);
    if (right <= left)
    {
      return 1.0;
    }

    int valid_pair_count = 0;
    int total_pair_count = 0;
    for (int i = left; i < right; ++i)
    {
      const Eigen::Vector3d t1 = keyframes[i].pose.translation();
      const Eigen::Vector3d t2 = keyframes[i + 1].pose.translation();
      const double trans_diff = (t2 - t1).norm();

      const Eigen::Matrix3d R_rel =
          keyframes[i].pose.rotation().transpose() *
          keyframes[i + 1].pose.rotation();
      const Eigen::AngleAxisd aa(R_rel);
      const double rot_diff = std::abs(aa.angle());

      if (trans_diff <= params_.trans_continuity_thresh &&
          rot_diff <= params_.rot_continuity_thresh)
      {
        valid_pair_count++;
      }
      total_pair_count++;
    }

    if (total_pair_count == 0)
    {
      return 1.0;
    }
    return static_cast<double>(valid_pair_count) /
           static_cast<double>(total_pair_count);
  }

  double poseUncertainty(const Eigen::Matrix<double, 6, 6>& cov) const
  {
    const Eigen::Matrix<double, 6, 6> weighted = params_.covariance_weight * cov;
    return std::max(0.0, weighted.trace());
  }

  static double percentile(const std::vector<double>& sorted_values, double ratio)
  {
    if (sorted_values.empty())
    {
      return 0.0;
    }

    ratio = std::max(0.0, std::min(1.0, ratio));
    const double pos = ratio * static_cast<double>(sorted_values.size() - 1);
    int idx0 = static_cast<int>(std::floor(pos));
    int idx1 = static_cast<int>(std::ceil(pos));
    idx0 = std::max(0, std::min(idx0, static_cast<int>(sorted_values.size()) - 1));
    idx1 = std::max(0, std::min(idx1, static_cast<int>(sorted_values.size()) - 1));

    const double alpha = pos - static_cast<double>(idx0);
    return (1.0 - alpha) * sorted_values[idx0] + alpha * sorted_values[idx1];
  }

  static double clamp01(double x)
  {
    if (std::isnan(x) || std::isinf(x))
    {
      return 0.0;
    }
    if (x < 0.0)
    {
      return 0.0;
    }
    if (x > 1.0)
    {
      return 1.0;
    }
    return x;
  }

  void normalizeWeights()
  {
    normalize3(params_.w_line, params_.w_plane, params_.w_scatter);
    normalize4(params_.w_azimuth_cover,
               params_.w_azimuth_uniform,
               params_.w_max_gap,
               params_.w_height);
    normalize4(params_.w_geo, params_.w_pose, params_.w_spa, params_.w_con);
  }

  static void normalize3(double& a, double& b, double& c)
  {
    const double sum = a + b + c;
    if (sum <= 1e-9)
    {
      a = b = c = 1.0 / 3.0;
      return;
    }
    a /= sum;
    b /= sum;
    c /= sum;
  }

  static void normalize4(double& a, double& b, double& c, double& d)
  {
    const double sum = a + b + c + d;
    if (sum <= 1e-9)
    {
      a = b = c = d = 0.25;
      return;
    }
    a /= sum;
    b /= sum;
    c /= sum;
    d /= sum;
  }

private:
  Params params_;
};

inline KeyframeCloudScorer::KeyframeCloudScorer()
    : KeyframeCloudScorer(Params())
{
}

inline KeyframeCloudScorer::KeyframeCloudScorer(const Params& params)
    : params_(params)
{
  normalizeWeights();
}
