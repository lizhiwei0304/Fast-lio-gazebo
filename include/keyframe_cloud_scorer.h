#pragma once

// PCL 点云基础功能：
// point_tests.h 用于判断点是否为有限值，例如 pcl::isFinite(pt)
#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Eigen 用于矩阵、位姿、特征值分解等数学计算
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

/**
 * @brief 关键帧点云评分器
 *
 * 该类用于对历史关键帧点云进行质量评估，并从中筛选适合作为
 * 主动回环锚点或候选回环关键帧的节点。
 *
 * 评分主要包含四个方面：
 * 1. q_geo  ：几何结构质量评分
 * 2. q_pose ：位姿可靠性评分
 * 3. q_spa  ：空间观测分布评分
 * 4. q_con  ：轨迹连续性评分
 *
 * 最终综合评分：
 *
 * q_total = w_geo * q_geo
 *         + w_pose * q_pose
 *         + w_spa * q_spa
 *         + w_con * q_con
 *
 * 如果各项评分和总评分都超过阈值，则该关键帧被认为是候选回环锚点。
 */
class KeyframeCloudScorer
{
public:
  // 点类型：XYZ + intensity
  using PointT = pcl::PointXYZI;

  // 点云类型
  using CloudT = pcl::PointCloud<PointT>;

  /**
   * @brief 关键帧数据结构
   *
   * 每个关键帧包含：
   * - 所属机器人编号
   * - 当前关键帧索引
   * - 关键帧位姿
   * - 位姿协方差
   * - 对应局部点云
   */
  struct KeyFrame
  {
    // 该关键帧所属机器人编号
    int robot_id = 0;

    // 关键帧索引，一般对应 keyframes_ 中的下标
    int index = 0;

    // 关键帧位姿，使用 Eigen::Isometry3d 表示 SE(3) 位姿
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    // 6 维位姿协方差矩阵
    // 一般对应 [x, y, z, roll, pitch, yaw] 或李代数扰动量
    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Zero();

    // 关键帧点云
    CloudT::Ptr cloud;
  };

  /**
   * @brief 评分器参数结构体
   *
   * 这里集中存放所有评分相关参数，包括：
   * - 体素几何结构评分参数
   * - 位姿可靠性评分参数
   * - 空间分布评分参数
   * - 轨迹连续性评分参数
   * - 综合评分权重
   * - 候选关键帧筛选阈值
   */
  struct Params
  {
    /**
     * -------------------------------
     * 1. 几何结构评分相关参数
     * -------------------------------
     */

    // 体素大小，单位通常为 m
    // 点云会按照该尺寸划分为多个体素
    double voxel_size = 1.0;

    // 每个体素内至少需要多少个点才参与几何评分
    // 点数过少时，该体素的协方差和特征值不稳定
    int min_points_per_voxel = 8;

    // 线性结构权重
    // 线性结构通常对应边缘、杆状物、长条结构等
    double w_line = 0.35;

    // 平面结构权重
    // 平面结构通常对应墙面、地面、立面等
    double w_plane = 0.45;

    // 散乱结构权重
    // 散乱结构表示点云在三个方向都有分布
    double w_scatter = 0.20;

    // 有效体素数量增益系数
    // 有效体素越多，说明该关键帧点云覆盖的结构越丰富
    double eta_valid_voxel = 0.01;

    /**
     * -------------------------------
     * 2. 位姿可靠性评分相关参数
     * -------------------------------
     */

    // 是否使用位姿可靠性评分
    // 如果为 false，则 q_pose 恒为 1.0
    bool use_pose_reliability = true;

    // 当前关键帧自身协方差惩罚系数
    // 协方差越大，exp(-eta_pose_self * uncertainty) 越小
    double eta_pose_self = 1.0;

    // 邻域关键帧平均协方差惩罚系数
    double eta_pose_neighbor = 1.0;

    // 当前关键帧自身位姿可靠性所占权重
    // 1 - alpha_pose_self 是邻域平均可靠性权重
    double alpha_pose_self = 0.6;

    // 协方差加权矩阵
    // 可用于对平移或旋转方向赋予不同权重
    Eigen::Matrix<double, 6, 6> covariance_weight =
        Eigen::Matrix<double, 6, 6>::Identity();

    /**
     * -------------------------------
     * 3. 空间分布评分相关参数
     * -------------------------------
     */

    // 方位角方向划分的扇区数量
    // 例如 36 表示每 10 度一个扇区
    int sector_num = 36;

    // 每个扇区至少需要多少个点才认为该扇区有效
    int min_points_per_sector = 20;

    // 方位覆盖率评分权重
    // 有效扇区越多，说明点云水平观测覆盖越完整
    double w_azimuth_cover = 0.30;

    // 方位均匀性评分权重
    // 使用熵衡量不同扇区点数分布是否均匀
    double w_azimuth_uniform = 0.30;

    // 最大空缺角惩罚项权重
    // 空缺角越大，说明某个方向长期没有观测
    double w_max_gap = 0.20;

    // 高度变化评分权重
    // 高度变化越丰富，说明三维结构越明显
    double w_height = 0.20;

    // 高度变化增益系数
    double eta_height = 1.0;

    // 输入点云是否已经在世界坐标系下
    // false：默认点云已经在关键帧局部坐标系下
    // true ：点云在世界系下，需要通过 pose.inverse() 转回局部系
    bool cloud_in_world_frame = false;

    /**
     * -------------------------------
     * 4. 轨迹连续性评分相关参数
     * -------------------------------
     */

    // 邻域半径
    // 对 target_idx 前后 neighbor_radius 个关键帧进行邻域评价
    int neighbor_radius = 3;

    // 相邻关键帧之间的最大平移连续性阈值
    // 如果相邻关键帧平移距离超过该值，则认为轨迹存在跳变
    double trans_continuity_thresh = 2.0;

    // 相邻关键帧之间的最大旋转连续性阈值
    // 默认 20 度，转为弧度
    double rot_continuity_thresh = 20.0 * 3.14159265358979323846 / 180.0;

    /**
     * -------------------------------
     * 5. 综合评分权重
     * -------------------------------
     */

    // 几何结构评分权重
    double w_geo = 0.35;

    // 位姿可靠性评分权重
    double w_pose = 0.20;

    // 空间分布评分权重
    double w_spa = 0.30;

    // 轨迹连续性评分权重
    double w_con = 0.15;

    /**
     * -------------------------------
     * 6. 候选关键帧筛选阈值
     * -------------------------------
     */

    // 几何评分最低阈值
    double min_geo_score = 0.25;

    // 位姿评分最低阈值
    double min_pose_score = 0.20;

    // 空间分布评分最低阈值
    double min_spa_score = 0.25;

    // 轨迹连续性评分最低阈值
    double min_con_score = 0.50;

    // 综合评分最低阈值
    double min_total_score = 0.45;
  };

  /**
   * @brief 单个关键帧的评分结果
   */
  struct Score
  {
    // 几何结构质量评分
    double q_geo = 0.0;

    // 位姿可靠性评分
    double q_pose = 0.0;

    // 空间观测分布评分
    double q_spa = 0.0;

    // 轨迹连续性评分
    double q_con = 0.0;

    // 综合评分
    double q_total = 0.0;

    // 有效体素数量
    int valid_voxel_num = 0;

    // 有效方位扇区数量
    int valid_sector_num = 0;

    // 点云高度范围
    // 使用 95% 分位高度与 5% 分位高度之差，避免极端离群点影响
    double height_range = 0.0;

    // 最大方位空缺角
    // 表示水平观测中最大未覆盖角度
    double max_gap_angle = 0.0;

    // 是否满足候选关键帧筛选条件
    bool is_candidate = false;
  };

  /**
   * @brief 默认构造函数
   *
   * 使用默认参数 Params()
   */
  KeyframeCloudScorer();

  /**
   * @brief 带参数构造函数
   *
   * @param params 外部传入的评分参数
   */
  explicit KeyframeCloudScorer(const Params& params);

  /**
   * @brief 评估单个关键帧
   *
   * @param keyframes 全部历史关键帧
   * @param target_idx 待评估关键帧索引
   * @return Score 当前关键帧的评分结果
   *
   * 该函数会依次计算：
   * 1. 几何结构质量 q_geo
   * 2. 位姿可靠性 q_pose
   * 3. 空间分布质量 q_spa
   * 4. 轨迹连续性 q_con
   *
   * 然后根据权重计算综合评分 q_total。
   * 如果各项评分均超过对应阈值，则 is_candidate = true。
   */
  Score evaluateOne(const std::vector<KeyFrame>& keyframes, int target_idx) const
  {
    Score score;

    // 索引越界保护
    if (target_idx < 0 || target_idx >= static_cast<int>(keyframes.size()))
    {
      return score;
    }

    const KeyFrame& kf = keyframes[target_idx];

    // 点云为空时无法评价，直接返回默认 0 分
    if (!kf.cloud || kf.cloud->empty())
    {
      return score;
    }

    // 计算几何结构质量评分
    // 同时返回有效体素数量
    score.q_geo = computeGeometryScore(kf, score.valid_voxel_num);

    // 计算位姿可靠性评分
    // 主要依据当前关键帧及其邻域关键帧的位姿协方差
    score.q_pose = computePoseReliabilityScore(keyframes, target_idx);

    // 计算空间分布评分
    // 同时返回有效扇区数、高度范围、最大空缺角
    score.q_spa = computeSpatialDistributionScore(
        kf, score.valid_sector_num, score.height_range, score.max_gap_angle);

    // 计算轨迹连续性评分
    // 判断当前关键帧附近的轨迹是否平滑、连续
    score.q_con = computeTrajectoryContinuityScore(keyframes, target_idx);

    // 计算综合评分
    score.q_total =
        params_.w_geo * score.q_geo +
        params_.w_pose * score.q_pose +
        params_.w_spa * score.q_spa +
        params_.w_con * score.q_con;

    // 将综合评分限制在 [0, 1]
    score.q_total = clamp01(score.q_total);

    // 判断是否满足候选关键帧条件
    // 这里不仅要求总分足够高，也要求每个子指标不能太差
    score.is_candidate =
        score.q_geo >= params_.min_geo_score &&
        score.q_pose >= params_.min_pose_score &&
        score.q_spa >= params_.min_spa_score &&
        score.q_con >= params_.min_con_score &&
        score.q_total >= params_.min_total_score;

    return score;
  }

  /**
   * @brief 评估全部关键帧
   *
   * @param keyframes 全部历史关键帧
   * @return std::vector<Score> 每个关键帧对应一个评分结果
   *
   * 注意：
   * scores[i] 与 keyframes[i] 一一对应。
   */
  std::vector<Score> evaluateAll(const std::vector<KeyFrame>& keyframes) const
  {
    std::vector<Score> scores;
    scores.reserve(keyframes.size());

    // 逐个关键帧进行评分
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i)
    {
      scores.emplace_back(evaluateOne(keyframes, i));
    }

    return scores;
  }

  /**
   * @brief 从所有评分结果中选择候选关键帧
   *
   * @param keyframes 全部关键帧
   * @param scores 全部关键帧评分
   * @param nms_distance 非极大值抑制距离
   * @param max_candidate_num 最大候选数量，<= 0 表示不限制数量
   * @return std::vector<int> 被选中的关键帧索引
   *
   * 筛选流程：
   * 1. 先取出 is_candidate = true 的关键帧；
   * 2. 按照 q_total 从高到低排序；
   * 3. 进行空间非极大值抑制，避免候选点过于密集；
   * 4. 如果设置了最大数量，则达到数量后停止。
   */
  std::vector<int> selectCandidates(const std::vector<KeyFrame>& keyframes,
                                    const std::vector<Score>& scores,
                                    double nms_distance,
                                    int max_candidate_num = 0) const
  {
    std::vector<int> indices;

    // 先筛选出满足基本阈值的候选关键帧
    for (int i = 0; i < static_cast<int>(scores.size()); ++i)
    {
      if (scores[i].is_candidate)
      {
        indices.emplace_back(i);
      }
    }

    // 按照综合评分从高到低排序
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return scores[a].q_total > scores[b].q_total;
    });

    std::vector<int> selected;

    // 非极大值抑制距离，保证不为负数
    const double min_dist = std::max(0.0, nms_distance);

    // 依照评分从高到低依次尝试保留候选点
    for (int idx : indices)
    {
      const Eigen::Vector3d p = keyframes[idx].pose.translation();

      bool keep = true;

      // 与已经保留的候选点比较距离
      // 如果距离太近，则当前点被抑制
      for (int kept : selected)
      {
        const Eigen::Vector3d q = keyframes[kept].pose.translation();

        if ((p - q).norm() < min_dist)
        {
          keep = false;
          break;
        }
      }

      // 当前候选点与已有候选点距离足够远，则保留
      if (keep)
      {
        selected.emplace_back(idx);

        // 如果限制了候选数量，达到数量后提前结束
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
  /**
   * @brief 体素哈希表的 key
   *
   * 用整数三元组表示体素坐标。
   */
  struct VoxelKey
  {
    int x = 0;
    int y = 0;
    int z = 0;

    // 判断两个体素 key 是否相同
    bool operator==(const VoxelKey& other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  /**
   * @brief 体素 key 的哈希函数
   *
   * 用于 unordered_map<VoxelKey, ...>。
   */
  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey& k) const
    {
      const std::size_t h1 = std::hash<int>()(k.x);
      const std::size_t h2 = std::hash<int>()(k.y);
      const std::size_t h3 = std::hash<int>()(k.z);

      // 简单组合三个方向的哈希值
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  /**
   * @brief 计算关键帧点云的几何结构质量评分
   *
   * @param kf 待评分关键帧
   * @param valid_voxel_num 输出参数，有效体素数量
   * @return double 几何结构评分 q_geo，范围 [0, 1]
   *
   * 计算逻辑：
   * 1. 将点云划分为体素；
   * 2. 对每个有效体素计算 3D 协方差矩阵；
   * 3. 对协方差矩阵做特征值分解；
   * 4. 根据特征值计算线性度、平面度和散乱度；
   * 5. 对所有有效体素取平均；
   * 6. 结合有效体素数量增益，得到最终几何评分。
   */
  double computeGeometryScore(const KeyFrame& kf, int& valid_voxel_num) const
  {
    valid_voxel_num = 0;

    // 体素大小非法时无法计算
    if (params_.voxel_size <= 1e-6)
    {
      return 0.0;
    }

    // 体素哈希表：
    // key   ：体素坐标
    // value ：该体素中的点
    std::unordered_map<VoxelKey, std::vector<Eigen::Vector3d>, VoxelKeyHash> voxel_map;
    voxel_map.reserve(kf.cloud->size());

    // 遍历点云，将点分配到不同体素
    for (const auto& pt : kf.cloud->points)
    {
      // 跳过 NaN 或 Inf 点
      if (!pcl::isFinite(pt))
      {
        continue;
      }

      Eigen::Vector3d p(pt.x, pt.y, pt.z);

      // 如果点云在世界坐标系下，则转到当前关键帧局部坐标系
      // 这样评分更关注当前关键帧自身观测结构，而不是世界坐标位置
      if (params_.cloud_in_world_frame)
      {
        p = kf.pose.inverse() * p;
      }

      // 根据体素大小计算体素索引
      VoxelKey key;
      key.x = static_cast<int>(std::floor(p.x() / params_.voxel_size));
      key.y = static_cast<int>(std::floor(p.y() / params_.voxel_size));
      key.z = static_cast<int>(std::floor(p.z() / params_.voxel_size));

      // 将点加入对应体素
      voxel_map[key].emplace_back(p);
    }

    double voxel_score_sum = 0.0;

    // 遍历所有体素，计算每个有效体素的局部几何结构分数
    for (const auto& item : voxel_map)
    {
      const auto& points = item.second;

      // 点数太少的体素不参与评分
      if (static_cast<int>(points.size()) < params_.min_points_per_voxel)
      {
        continue;
      }

      // 计算体素内点的均值
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      for (const auto& p : points)
      {
        mean += p;
      }
      mean /= static_cast<double>(points.size());

      // 计算体素内点的 3x3 协方差矩阵
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (const auto& p : points)
      {
        const Eigen::Vector3d d = p - mean;
        cov += d * d.transpose();
      }
      cov /= static_cast<double>(points.size());

      // 对协方差矩阵做特征值分解
      // SelfAdjointEigenSolver 适用于对称矩阵
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
      if (solver.info() != Eigen::Success)
      {
        continue;
      }

      // Eigen 返回的特征值默认从小到大排列
      // l1 >= l2 >= l3
      const Eigen::Vector3d eig = solver.eigenvalues();
      const double l1 = std::max(0.0, eig(2));
      const double l2 = std::max(0.0, eig(1));
      const double l3 = std::max(0.0, eig(0));

      // 使用最大特征值归一化，防止尺度影响
      const double denom = l1 + 1e-9;

      // 线性度：
      // 当 l1 远大于 l2、l3 时，点云近似线状结构
      const double linearity = clamp01((l1 - l2) / denom);

      // 平面度：
      // 当 l1、l2 较大而 l3 很小时，点云近似平面结构
      const double planarity = clamp01((l2 - l3) / denom);

      // 散乱度：
      // 当 l3 也较大时，点云在三个方向都有明显分布
      const double scattering = clamp01(l3 / denom);

      // 当前体素几何评分
      const double q_voxel =
          params_.w_line * linearity +
          params_.w_plane * planarity +
          params_.w_scatter * scattering;

      voxel_score_sum += clamp01(q_voxel);
      valid_voxel_num++;
    }

    // 没有有效体素则几何评分为 0
    if (valid_voxel_num == 0)
    {
      return 0.0;
    }

    // 所有有效体素的平均几何结构评分
    const double mean_voxel_score =
        voxel_score_sum / static_cast<double>(valid_voxel_num);

    // 有效体素数量增益
    // 有效体素越多，voxel_num_gain 越接近 1
    const double voxel_num_gain =
        1.0 - std::exp(-params_.eta_valid_voxel *
                       static_cast<double>(valid_voxel_num));

    // 最终几何评分 = 有效体素数量增益 × 平均体素结构质量
    return clamp01(voxel_num_gain * mean_voxel_score);
  }

  /**
   * @brief 计算位姿可靠性评分
   *
   * @param keyframes 全部关键帧
   * @param target_idx 当前关键帧索引
   * @return double 位姿可靠性评分 q_pose，范围 [0, 1]
   *
   * 计算逻辑：
   * 1. 当前关键帧协方差越小，自身可靠性越高；
   * 2. 邻域关键帧平均协方差越小，局部轨迹段整体越可靠；
   * 3. 将自身可靠性和邻域可靠性加权融合。
   */
  double computePoseReliabilityScore(const std::vector<KeyFrame>& keyframes,
                                     int target_idx) const
  {
    // 不使用位姿可靠性时，直接返回满分
    if (!params_.use_pose_reliability)
    {
      return 1.0;
    }

    const int n = static_cast<int>(keyframes.size());
    const KeyFrame& kf = keyframes[target_idx];

    // 当前关键帧自身不确定性
    const double u_self = poseUncertainty(kf.covariance);

    // 当前关键帧自身可靠性
    // 不确定性越大，q_self 越小
    const double q_self = clamp01(std::exp(-params_.eta_pose_self * u_self));

    // 确定邻域范围
    const int left = std::max(0, target_idx - params_.neighbor_radius);
    const int right = std::min(n - 1, target_idx + params_.neighbor_radius);

    // 计算邻域关键帧平均不确定性
    double u_sum = 0.0;
    int count = 0;
    for (int i = left; i <= right; ++i)
    {
      u_sum += poseUncertainty(keyframes[i].covariance);
      count++;
    }

    // 邻域可靠性默认取自身可靠性
    double q_neighbor = q_self;

    if (count > 0)
    {
      const double u_mean = u_sum / static_cast<double>(count);

      // 邻域平均不确定性越大，邻域可靠性越低
      q_neighbor = clamp01(std::exp(-params_.eta_pose_neighbor * u_mean));
    }

    // 融合自身可靠性和邻域可靠性
    return clamp01(params_.alpha_pose_self * q_self +
                   (1.0 - params_.alpha_pose_self) * q_neighbor);
  }

  /**
   * @brief 计算点云空间分布评分
   *
   * @param kf 待评分关键帧
   * @param valid_sector_num 输出：有效扇区数量
   * @param height_range 输出：高度范围
   * @param max_gap_angle 输出：最大方位空缺角
   * @return double 空间分布评分 q_spa，范围 [0, 1]
   *
   * 计算逻辑：
   * 1. 将点云按照水平角度划分为多个扇区；
   * 2. 统计有效扇区数量，得到方位覆盖率；
   * 3. 用熵衡量有效扇区点数分布均匀性；
   * 4. 计算最大连续空缺扇区对应角度；
   * 5. 根据高度分位数计算高度变化范围；
   * 6. 加权得到空间分布评分。
   */
  double computeSpatialDistributionScore(const KeyFrame& kf,
                                         int& valid_sector_num,
                                         double& height_range,
                                         double& max_gap_angle) const
  {
    valid_sector_num = 0;
    height_range = 0.0;
    max_gap_angle = 0.0;

    // 至少划分为 4 个扇区，避免扇区数量过小
    const int B = std::max(4, params_.sector_num);

    const double two_pi = 2.0 * 3.14159265358979323846;

    // 每个扇区的点数统计
    std::vector<int> sector_counts(B, 0);

    // 保存所有有效点的 z 值，用于计算高度范围
    std::vector<double> z_values;
    z_values.reserve(kf.cloud->size());

    // 遍历点云，统计方位角扇区和高度值
    for (const auto& pt : kf.cloud->points)
    {
      // 跳过无效点
      if (!pcl::isFinite(pt))
      {
        continue;
      }

      Eigen::Vector3d p(pt.x, pt.y, pt.z);

      // 如果点云在世界系下，则转换到当前关键帧局部坐标系
      if (params_.cloud_in_world_frame)
      {
        p = kf.pose.inverse() * p;
      }

      // 计算水平角度 atan2(y, x)，范围为 [-pi, pi]
      double angle = std::atan2(p.y(), p.x());

      // 转换到 [0, 2pi]
      if (angle < 0.0)
      {
        angle += two_pi;
      }

      // 根据角度确定所属扇区编号
      int sector_id = static_cast<int>(std::floor(angle / two_pi * B));

      // 防止数值误差导致越界
      sector_id = std::max(0, std::min(B - 1, sector_id));

      // 当前扇区点数加一
      sector_counts[sector_id]++;

      // 记录高度值
      z_values.emplace_back(p.z());
    }

    // 没有有效点，则空间分布评分为 0
    if (z_values.empty())
    {
      return 0.0;
    }

    // 统计有效扇区
    std::vector<int> valid_indices;
    valid_indices.reserve(B);

    int total_valid_points = 0;
    for (int i = 0; i < B; ++i)
    {
      // 点数超过阈值的扇区被认为是有效扇区
      if (sector_counts[i] >= params_.min_points_per_sector)
      {
        valid_indices.emplace_back(i);
        total_valid_points += sector_counts[i];
      }
    }

    // 有效扇区数量
    valid_sector_num = static_cast<int>(valid_indices.size());

    // 方位覆盖率评分
    // 有效扇区越多，q_cover 越接近 1
    const double q_cover =
        clamp01(static_cast<double>(valid_sector_num) / static_cast<double>(B));

    // 方位均匀性评分
    // 使用归一化熵衡量有效扇区点数分布是否均匀
    double q_uniform = 0.0;
    if (valid_sector_num > 1 && total_valid_points > 0)
    {
      double entropy = 0.0;

      for (int idx : valid_indices)
      {
        const double prob = static_cast<double>(sector_counts[idx]) /
                            static_cast<double>(total_valid_points);

        // 信息熵
        entropy += -prob * std::log(prob + 1e-12);
      }

      // 用 log(valid_sector_num) 归一化到 [0, 1]
      q_uniform = clamp01(entropy / std::log(static_cast<double>(valid_sector_num)));
    }

    // 最大空缺角评分
    // 如果某些方向长时间没有点云观测，则说明空间分布不完整
    double q_gap = 0.0;
    if (!valid_indices.empty())
    {
      // 对有效扇区编号排序
      std::sort(valid_indices.begin(), valid_indices.end());

      int max_empty_bins = 0;

      // 查找相邻有效扇区之间最大的空缺扇区数量
      // 注意这里考虑了首尾环绕
      for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i)
      {
        const int cur = valid_indices[i];
        int next = valid_indices[(i + 1) % valid_indices.size()];

        int diff = next - cur;

        // 如果 diff <= 0，说明跨越了 2pi 到 0 的环绕边界
        if (diff <= 0)
        {
          diff += B;
        }

        // diff - 1 表示两个有效扇区之间的空扇区数量
        max_empty_bins = std::max(max_empty_bins, diff - 1);
      }

      // 最大空缺角
      max_gap_angle =
          static_cast<double>(max_empty_bins) * two_pi / static_cast<double>(B);

      // 空缺角越大，q_gap 越小
      q_gap = clamp01(1.0 - max_gap_angle / two_pi);
    }

    // 计算高度范围
    // 使用 5% 和 95% 分位数，降低离群点影响
    std::sort(z_values.begin(), z_values.end());
    const double z5 = percentile(z_values, 0.05);
    const double z95 = percentile(z_values, 0.95);

    height_range = std::max(0.0, z95 - z5);

    // 高度变化评分
    // 高度范围越大，q_height 越接近 1
    const double q_height =
        clamp01(1.0 - std::exp(-height_range / std::max(1e-6, params_.eta_height)));

    // 加权融合空间分布各项指标
    return clamp01(params_.w_azimuth_cover * q_cover +
                   params_.w_azimuth_uniform * q_uniform +
                   params_.w_max_gap * q_gap +
                   params_.w_height * q_height);
  }

  /**
   * @brief 计算轨迹连续性评分
   *
   * @param keyframes 全部关键帧
   * @param target_idx 当前关键帧索引
   * @return double 轨迹连续性评分 q_con，范围 [0, 1]
   *
   * 计算逻辑：
   * 在当前关键帧邻域内，检查相邻关键帧之间的平移变化和旋转变化。
   * 如果相邻帧之间的平移和旋转都小于阈值，则认为该相邻对连续。
   *
   * q_con = 连续相邻对数量 / 总相邻对数量
   */
  double computeTrajectoryContinuityScore(const std::vector<KeyFrame>& keyframes,
                                          int target_idx) const
  {
    const int n = static_cast<int>(keyframes.size());

    // 取当前关键帧附近的局部窗口
    const int left = std::max(0, target_idx - params_.neighbor_radius);
    const int right = std::min(n - 1, target_idx + params_.neighbor_radius);

    // 邻域内没有形成相邻帧对时，认为连续性为满分
    if (right <= left)
    {
      return 1.0;
    }

    int valid_pair_count = 0;
    int total_pair_count = 0;

    // 遍历邻域内相邻关键帧对
    for (int i = left; i < right; ++i)
    {
      // 当前帧和平移后一帧的位置
      const Eigen::Vector3d t1 = keyframes[i].pose.translation();
      const Eigen::Vector3d t2 = keyframes[i + 1].pose.translation();

      // 相邻关键帧之间的平移距离
      const double trans_diff = (t2 - t1).norm();

      // 相邻关键帧之间的相对旋转
      const Eigen::Matrix3d R_rel =
          keyframes[i].pose.rotation().transpose() *
          keyframes[i + 1].pose.rotation();

      // 将相对旋转矩阵转换为角轴形式，提取旋转角
      const Eigen::AngleAxisd aa(R_rel);
      const double rot_diff = std::abs(aa.angle());

      // 如果平移和旋转都在阈值范围内，则认为该相邻帧对是连续的
      if (trans_diff <= params_.trans_continuity_thresh &&
          rot_diff <= params_.rot_continuity_thresh)
      {
        valid_pair_count++;
      }

      total_pair_count++;
    }

    // 没有相邻帧对时默认满分
    if (total_pair_count == 0)
    {
      return 1.0;
    }

    // 连续帧对比例作为轨迹连续性评分
    return static_cast<double>(valid_pair_count) /
           static_cast<double>(total_pair_count);
  }

  /**
   * @brief 计算位姿不确定性标量
   *
   * @param cov 6x6 位姿协方差矩阵
   * @return double 标量不确定性
   *
   * 当前实现：
   * uncertainty = trace(covariance_weight * cov)
   *
   * trace 越大，表示整体位姿不确定性越大。
   */
  double poseUncertainty(const Eigen::Matrix<double, 6, 6>& cov) const
  {
    const Eigen::Matrix<double, 6, 6> weighted = params_.covariance_weight * cov;

    // 返回加权协方差矩阵迹，并保证非负
    return std::max(0.0, weighted.trace());
  }

  /**
   * @brief 计算已排序数组的分位数
   *
   * @param sorted_values 已经升序排序的数值数组
   * @param ratio 分位比例，范围 [0, 1]
   * @return double 对应分位数
   *
   * 例如：
   * ratio = 0.05 表示 5% 分位数；
   * ratio = 0.95 表示 95% 分位数。
   *
   * 这里使用线性插值。
   */
  static double percentile(const std::vector<double>& sorted_values, double ratio)
  {
    if (sorted_values.empty())
    {
      return 0.0;
    }

    // 将 ratio 限制在 [0, 1]
    ratio = std::max(0.0, std::min(1.0, ratio));

    // 分位数对应的连续下标位置
    const double pos = ratio * static_cast<double>(sorted_values.size() - 1);

    // 左右两个整数下标
    int idx0 = static_cast<int>(std::floor(pos));
    int idx1 = static_cast<int>(std::ceil(pos));

    // 防止越界
    idx0 = std::max(0, std::min(idx0, static_cast<int>(sorted_values.size()) - 1));
    idx1 = std::max(0, std::min(idx1, static_cast<int>(sorted_values.size()) - 1));

    // 线性插值系数
    const double alpha = pos - static_cast<double>(idx0);

    // 返回线性插值结果
    return (1.0 - alpha) * sorted_values[idx0] + alpha * sorted_values[idx1];
  }

  /**
   * @brief 将数值限制到 [0, 1]
   *
   * @param x 输入值
   * @return double 输出值
   *
   * 如果 x 是 NaN 或 Inf，则返回 0。
   */
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

  /**
   * @brief 归一化所有权重参数
   *
   * 作用：
   * 保证不同评分模块内部权重和为 1。
   *
   * 包括：
   * 1. 几何评分中的线性、平面、散乱权重；
   * 2. 空间分布评分中的覆盖率、均匀性、最大空缺角、高度权重；
   * 3. 综合评分中的几何、位姿、空间、连续性权重。
   */
  void normalizeWeights()
  {
    normalize3(params_.w_line, params_.w_plane, params_.w_scatter);

    normalize4(params_.w_azimuth_cover,
               params_.w_azimuth_uniform,
               params_.w_max_gap,
               params_.w_height);

    normalize4(params_.w_geo, params_.w_pose, params_.w_spa, params_.w_con);
  }

  /**
   * @brief 归一化三个权重
   *
   * 如果三个权重之和过小，则平均分配为 1/3。
   */
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

  /**
   * @brief 归一化四个权重
   *
   * 如果四个权重之和过小，则平均分配为 0.25。
   */
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
  // 评分器参数
  Params params_;
};

/**
 * @brief 默认构造函数实现
 *
 * 使用默认参数构造评分器。
 */
inline KeyframeCloudScorer::KeyframeCloudScorer()
    : KeyframeCloudScorer(Params())
{
}

/**
 * @brief 带参数构造函数实现
 *
 * 保存外部传入参数后，对所有权重进行归一化。
 */
inline KeyframeCloudScorer::KeyframeCloudScorer(const Params& params)
    : params_(params)
{
  normalizeWeights();
}