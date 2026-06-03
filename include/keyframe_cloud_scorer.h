#pragma once

#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

class KeyframeCloudScorer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using PointT = pcl::PointXYZI;
  using CloudT = pcl::PointCloud<PointT>;

  struct KeyFrame
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int robot_id = 0;
    int index = 0;

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Zero();

    CloudT::Ptr cloud;
  };

  using KeyFrameVector = std::vector<KeyFrame, Eigen::aligned_allocator<KeyFrame>>;
  using Vector3dVector =
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

  enum class SelectionMode
  {
    GEO,       // 对应 Python anchors_geo.csv: q_geo >= tau_geo
    GEO_POSE,  // 对应 Python anchors_pose.csv: q_geo_pose >= tau_pose
    PROPOSED  // 对应 Python anchors_proposed.csv: q_total >= tau_total
  };

  struct Params
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Params()
    {
      covariance_weight.setZero();
      covariance_weight.diagonal() << 1.0, 1.0, 1.0, 0.2, 0.2, 0.2;
    }

    // 与 select_loop_anchors_from_dislam_bag_modified.py 的新版默认参数保持一致。
    double voxel_size = 0.5;
    int min_points_per_voxel = 5;
    int min_points_cloud = 100;

    // 2.1 几何结构质量：有效结构数量 + 局部结构显著性 + 水平约束方向分布 + 垂直结构支撑。
    double eta_voxel_count = 150.0;
    double alpha_geo_structure = 0.30;
    double alpha_geo_distribution = 0.55;
    double alpha_geo_vertical = 0.15;
    double eta_vertical = 2.0;
    double min_horizontal_norm = 0.2;

    // false: keyframe cloud 已经在关键帧局部坐标系。
    // true : keyframe cloud 在世界坐标系，需要用 pose.inverse() 转到局部坐标系。
    bool cloud_in_world_frame = false;

    // 2.2 位姿约束可靠性。
    bool use_pose_reliability = true;
    double eta_pose = -1.0;  // <=0 时使用所有有效 raw uncertainty 的 90% 分位数。
    double alpha_pose_self = 0.5;
    Eigen::Matrix<double, 6, 6> covariance_weight;

    // 轨迹连续性。
    int neighbor_radius = 3;
    double max_keyframe_gap = 5.0;
    double eta_turn = 1.0;

    // 综合质量评分：只包含 q_geo、q_pose、q_continuity。
    double w_geo = 0.45;
    double w_pose = 0.40;
    double w_continuity = 0.15;

    // 与 Python tau_* 保持一致。
    double tau_geo = 0.65;
    double tau_pose = 0.65;
    double tau_total = 0.65;
  };

  struct Score
  {
    // 2.1 几何结构质量及其调试分量。
    double q_geo = 0.0;
    double q_geo_count = 0.0;
    double q_geo_structure = 0.0;
    double q_geo_distribution = 0.0;
    double q_geo_vertical = 0.0;

    int geo_total_voxels = 0;
    int geo_valid_voxels = 0;
    int geo_direction_voxels = 0;
    double geo_z_range = 0.0;

    // 2.2 位姿约束可靠性。
    double q_pose = 0.0;

    // 轨迹连续性。
    double q_continuity = 0.0;

    // 组合评分，对应 Python 输出字段。
    double q_geo_pose = 0.0;
    double q_total = 0.0;

    bool is_candidate = false;
  };

  KeyframeCloudScorer() = default;

  explicit KeyframeCloudScorer(const Params& params)
      : params_(params)
  {
  }

  Score evaluateOne(const KeyFrameVector& keyframes, int target_idx) const
  {
    const double eta_pose = computeEtaPose(keyframes);
    return evaluateOneWithEta(keyframes, target_idx, eta_pose);
  }

  std::vector<Score> evaluateAll(const KeyFrameVector& keyframes) const
  {
    const double eta_pose = computeEtaPose(keyframes);

    std::vector<Score> scores;
    scores.reserve(keyframes.size());
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i)
    {
      scores.emplace_back(evaluateOneWithEta(keyframes, i, eta_pose));
    }
    return scores;
  }

  // 保留原接口：默认按 Proposed 的 q_total 做阈值筛选和 NMS。
  std::vector<int> selectCandidates(const KeyFrameVector& keyframes,
                                    const std::vector<Score>& scores,
                                    double nms_distance,
                                    int max_candidate_num = 0) const
  {
    return selectCandidatesByMode(
        keyframes, scores, nms_distance, max_candidate_num, SelectionMode::PROPOSED);
  }

  // 新接口：分别对应 Python 中 anchors_geo、anchors_pose、anchors_proposed 三种输出。
  std::vector<int> selectCandidatesByMode(const KeyFrameVector& keyframes,
                                          const std::vector<Score>& scores,
                                          double nms_distance,
                                          int max_candidate_num,
                                          SelectionMode mode) const
  {
    std::vector<int> indices;
    const int candidate_count =
        std::min(static_cast<int>(keyframes.size()),
                 static_cast<int>(scores.size()));
    for (int i = 0; i < candidate_count; ++i)
    {
      if (scoreForMode(scores[i], mode) >= thresholdForMode(mode))
      {
        indices.emplace_back(i);
      }
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return scoreForMode(scores[a], mode) > scoreForMode(scores[b], mode);
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

  Score evaluateOneWithEta(const KeyFrameVector& keyframes,
                           int target_idx,
                           double eta_pose) const
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

    if (static_cast<int>(kf.cloud->size()) < params_.min_points_cloud)
    {
      return score;
    }

    score.q_geo = computeGeometryScore(kf, score);
    score.q_pose = computePoseReliabilityScore(keyframes, target_idx, eta_pose);
    score.q_continuity = computeTrajectoryContinuityScore(keyframes, target_idx);

    score.q_geo_pose = clamp01(
        weightedAverage2(score.q_geo, params_.w_geo, score.q_pose, params_.w_pose));

    const double w_sum = params_.w_geo + params_.w_pose + params_.w_continuity;
    if (w_sum > 1e-12)
    {
      score.q_total = clamp01(
          (params_.w_geo * score.q_geo +
           params_.w_pose * score.q_pose +
           params_.w_continuity * score.q_continuity) /
          w_sum);
    }
    else
    {
      score.q_total = 0.0;
    }

    // 与 Python select_anchors(method="proposed") 一致：proposed 只用 q_total 阈值。
    score.is_candidate = score.q_total >= params_.tau_total;

    return score;
  }

  double computeGeometryScore(const KeyFrame& kf, Score& score) const
  {
    score.geo_total_voxels = 0;
    score.geo_valid_voxels = 0;
    score.geo_direction_voxels = 0;
    score.geo_z_range = 0.0;
    score.q_geo_count = 0.0;
    score.q_geo_structure = 0.0;
    score.q_geo_distribution = 0.0;
    score.q_geo_vertical = 0.0;

    if (params_.voxel_size <= 1e-6)
    {
      return 0.0;
    }

    std::unordered_map<VoxelKey, Vector3dVector, VoxelKeyHash> voxel_map;
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

    score.geo_total_voxels = static_cast<int>(voxel_map.size());
    if (score.geo_total_voxels == 0)
    {
      return 0.0;
    }

    const double eps = 1e-9;
    std::vector<double> structure_scores;
    structure_scores.reserve(voxel_map.size());

    Vector3dVector valid_voxel_centers;
    valid_voxel_centers.reserve(voxel_map.size());

    Eigen::Matrix2d direction_matrix = Eigen::Matrix2d::Zero();
    int direction_voxels = 0;

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

      // Eigen 默认按升序输出特征值，对应 Python 中排序后的 l1 >= l2 >= l3。
      const Eigen::Vector3d eig = solver.eigenvalues();
      const double l1 = std::max(0.0, eig(2));
      const double l2 = std::max(0.0, eig(1));
      const double l3 = std::max(0.0, eig(0));
      if (l1 < eps)
      {
        continue;
      }

      // 局部结构显著性：s_v = (lambda1 - lambda3) / sum(lambda)。
      const double eig_sum = l1 + l2 + l3;
      const double s_v = clamp01((l1 - l3) / (eig_sum + eps));
      structure_scores.emplace_back(s_v);
      valid_voxel_centers.emplace_back(mean);

      // 水平约束方向分布：取最小特征值对应的特征向量作为局部法向参考。
      const Eigen::Vector3d normal = solver.eigenvectors().col(0);
      Eigen::Vector2d d_xy(normal.x(), normal.y());
      const double d_norm = d_xy.norm();

      // 地面法向接近竖直方向时，水平投影很小，不作为水平回环约束方向。
      if (d_norm > params_.min_horizontal_norm)
      {
        d_xy /= (d_norm + eps);
        direction_matrix += s_v * (d_xy * d_xy.transpose());
        direction_voxels++;
      }
    }

    if (structure_scores.empty())
    {
      return 0.0;
    }

    const int n_eff = static_cast<int>(structure_scores.size());
    score.geo_valid_voxels = n_eff;

    // 有效结构数量评分：1 - exp(-n_eff / eta_voxel_count)。
    score.q_geo_count = clamp01(
        1.0 - std::exp(-static_cast<double>(n_eff) /
                       std::max(params_.eta_voxel_count, eps)));

    double structure_sum = 0.0;
    for (double v : structure_scores)
    {
      structure_sum += v;
    }
    score.q_geo_structure = clamp01(structure_sum / static_cast<double>(n_eff));

    // 水平约束方向均衡性：2 * sqrt(beta1 * beta2)。
    score.geo_direction_voxels = direction_voxels;
    const double tr = direction_matrix.trace();
    if (direction_voxels >= 2 && tr > eps)
    {
      const Eigen::Matrix2d A = direction_matrix / (tr + eps);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver2(A);
      if (solver2.info() == Eigen::Success)
      {
        const Eigen::Vector2d beta = solver2.eigenvalues();
        const double beta0 = std::max(0.0, beta(0));
        const double beta1 = std::max(0.0, beta(1));
        score.q_geo_distribution =
            clamp01(2.0 * std::sqrt(std::max(0.0, beta0 * beta1)));
      }
    }

    // 垂直结构支撑：有效体素中心高度的 95% - 5% 分位差。
    if (valid_voxel_centers.size() >= 2)
    {
      std::vector<double> z_values;
      z_values.reserve(valid_voxel_centers.size());
      for (const auto& c : valid_voxel_centers)
      {
        z_values.emplace_back(c.z());
      }
      std::sort(z_values.begin(), z_values.end());
      const double z5 = percentile(z_values, 0.05);
      const double z95 = percentile(z_values, 0.95);
      score.geo_z_range = std::max(0.0, z95 - z5);
    }

    score.q_geo_vertical = clamp01(
        1.0 - std::exp(-std::max(0.0, score.geo_z_range) /
                       std::max(params_.eta_vertical, eps)));

    const double alpha_sum =
        params_.alpha_geo_structure +
        params_.alpha_geo_distribution +
        params_.alpha_geo_vertical;
    if (alpha_sum <= eps)
    {
      return 0.0;
    }

    const double q_geo =
        score.q_geo_count *
        (params_.alpha_geo_structure * score.q_geo_structure +
         params_.alpha_geo_distribution * score.q_geo_distribution +
         params_.alpha_geo_vertical * score.q_geo_vertical) /
        alpha_sum;

    return clamp01(q_geo);
  }

  double computePoseReliabilityScore(const KeyFrameVector& keyframes,
                                     int target_idx,
                                     double eta_pose) const
  {
    if (!params_.use_pose_reliability)
    {
      return 1.0;
    }

    // 与 Python 一致：如果没有有效 covariance，则所有 q_pose 设置为 1.0。
    if (eta_pose <= 0.0)
    {
      return 1.0;
    }

    const int n = static_cast<int>(keyframes.size());
    const double u_self = poseUncertainty(keyframes[target_idx].covariance);
    const double q_self = clamp01(std::exp(-u_self / eta_pose));

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
      q_neighbor = clamp01(std::exp(-u_mean / eta_pose));
    }

    return clamp01(params_.alpha_pose_self * q_self +
                   (1.0 - params_.alpha_pose_self) * q_neighbor);
  }

  double computeTrajectoryContinuityScore(const KeyFrameVector& keyframes,
                                          int target_idx) const
  {
    const int n = static_cast<int>(keyframes.size());
    const int left = std::max(0, target_idx - params_.neighbor_radius);
    const int right = std::min(n - 1, target_idx + params_.neighbor_radius);

    const int point_num = right - left + 1;
    if (point_num < 3)
    {
      return 0.5;
    }

    Vector3dVector pts;
    pts.reserve(point_num);
    for (int i = left; i <= right; ++i)
    {
      pts.emplace_back(keyframes[i].pose.translation());
    }

    Vector3dVector segs;
    std::vector<double> dist;
    segs.reserve(pts.size() - 1);
    dist.reserve(pts.size() - 1);

    for (std::size_t i = 1; i < pts.size(); ++i)
    {
      const Eigen::Vector3d s = pts[i] - pts[i - 1];
      segs.emplace_back(s);
      dist.emplace_back(s.norm());
    }

    std::vector<double> valid_dist;
    valid_dist.reserve(dist.size());
    for (double d : dist)
    {
      if (d > 1e-6)
      {
        valid_dist.emplace_back(d);
      }
    }

    if (valid_dist.size() < 2)
    {
      return 0.3;
    }

    double mean_d = 0.0;
    for (double d : valid_dist)
    {
      mean_d += d;
    }
    mean_d /= static_cast<double>(valid_dist.size());

    double var_d = 0.0;
    for (double d : valid_dist)
    {
      const double e = d - mean_d;
      var_d += e * e;
    }
    var_d /= static_cast<double>(valid_dist.size());
    const double std_d = std::sqrt(std::max(0.0, var_d));

    const double q_spacing = std::exp(-std_d / (mean_d + 1e-6));

    const double max_d = *std::max_element(valid_dist.begin(), valid_dist.end());
    double q_jump = 1.0;
    if (max_d > params_.max_keyframe_gap)
    {
      q_jump = std::exp(-(max_d - params_.max_keyframe_gap) /
                        (params_.max_keyframe_gap + 1e-6));
    }

    Vector3dVector unit;
    unit.reserve(segs.size());
    for (const auto& s : segs)
    {
      unit.emplace_back(s / (s.norm() + 1e-9));
    }

    double mean_turn = 0.0;
    int turn_count = 0;
    for (std::size_t i = 1; i < unit.size(); ++i)
    {
      double c = unit[i].dot(unit[i - 1]);
      c = std::max(-1.0, std::min(1.0, c));
      mean_turn += std::acos(c);
      turn_count++;
    }
    if (turn_count > 0)
    {
      mean_turn /= static_cast<double>(turn_count);
    }

    const double q_turn = std::exp(-mean_turn / std::max(params_.eta_turn, 1e-9));

    double valid_ratio = 1.0;
    if (params_.neighbor_radius > 0)
    {
      valid_ratio = std::min(1.0,
                             static_cast<double>(pts.size() - 1) /
                                 static_cast<double>(2 * params_.neighbor_radius));
    }

    const double q_cont = valid_ratio *
                          (0.45 * q_spacing + 0.45 * q_turn + 0.10 * q_jump);

    return clamp01(q_cont);
  }

  double poseUncertainty(const Eigen::Matrix<double, 6, 6>& cov) const
  {
    double u = 0.0;
    for (int i = 0; i < 6; ++i)
    {
      const double cov_diag = std::max(0.0, cov(i, i));
      const double weight = params_.covariance_weight(i, i);
      u += weight * cov_diag;
    }
    return std::max(0.0, u);
  }

  double computeEtaPose(const KeyFrameVector& keyframes) const
  {
    if (!params_.use_pose_reliability)
    {
      return 1.0;
    }

    if (params_.eta_pose > 0.0)
    {
      return params_.eta_pose;
    }

    std::vector<double> uncertainties;
    uncertainties.reserve(keyframes.size());
    for (const auto& kf : keyframes)
    {
      const double u = poseUncertainty(kf.covariance);
      if (u > 1e-12)
      {
        uncertainties.emplace_back(u);
      }
    }

    if (uncertainties.empty())
    {
      return -1.0;
    }

    std::sort(uncertainties.begin(), uncertainties.end());
    return std::max(1e-9, percentile(uncertainties, 0.90));
  }

  double scoreForMode(const Score& score, SelectionMode mode) const
  {
    switch (mode)
    {
      case SelectionMode::GEO:
        return score.q_geo;
      case SelectionMode::GEO_POSE:
        return score.q_geo_pose;
      case SelectionMode::PROPOSED:
      default:
        return score.q_total;
    }
  }

  double thresholdForMode(SelectionMode mode) const
  {
    switch (mode)
    {
      case SelectionMode::GEO:
        return params_.tau_geo;
      case SelectionMode::GEO_POSE:
        return params_.tau_pose;
      case SelectionMode::PROPOSED:
      default:
        return params_.tau_total;
    }
  }

  static double weightedAverage2(double a, double wa, double b, double wb)
  {
    const double sum = wa + wb;
    if (sum <= 1e-12)
    {
      return 0.0;
    }
    return (wa * a + wb * b) / sum;
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

private:
  Params params_;
};
