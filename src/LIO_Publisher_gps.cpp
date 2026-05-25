#include "ros/ros.h"
#include "ros/publisher.h"
#include "ros/subscriber.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <math.h>
#include <iostream>
#include <string>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/filters/passthrough.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include "dislam_msgs/SubMap.h"

using namespace std;
using namespace message_filters;

typedef sync_policies::ApproximateTime<nav_msgs::Odometry, nav_msgs::Odometry> MySyncPolicy;

class LIO_Pub
{
public:
  LIO_Pub(ros::NodeHandle &n);
  ~LIO_Pub();

private:
  double dis_th;
  int signal_num = 0;
  bool isFirstOdom = true;
  bool Signal = false;
  float distance = 0;
  float x0, y0, z0;
  float x1, y1, z1;
  string NameSpace;
  string RobotID;
  string SensorName;

  double origin_x_;
  double origin_y_;
  double origin_z_;
  bool use_gps_ = false;
  bool is_first_gps_ = true;

  // Declare your message_filters::Subscriber and Synchronizer
  message_filters::Subscriber<nav_msgs::Odometry> gps_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, nav_msgs::Odometry>> sync_;

  ros::Subscriber pointCloudSubscriber_;

  ros::Publisher pointCloudPublisher_;
  ros::Publisher signalPublisher_;
  ros::Publisher subMapPublisher_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr registeredCloud;

  void PCCallback(sensor_msgs::PointCloud2 msg);
  void OdomCallback(const nav_msgs::Odometry::ConstPtr &gps_msg, const nav_msgs::Odometry::ConstPtr &odom_msg);

  void pub_Signal();
  void pub_PC(sensor_msgs::PointCloud2 msg);
  void pub_TF(nav_msgs::Odometry msg);
  void reformatRobotName();
  Eigen::Isometry3d odom2isometry(const nav_msgs::Odometry odom_msg);
};

LIO_Pub::~LIO_Pub()
{
  ROS_WARN("Good Bye!!!");
}

LIO_Pub::LIO_Pub(ros::NodeHandle &n) : gps_sub_(n, "/robot_1/gnss_odom", 1), // 初始化 GPS 订阅者
                                       odom_sub_(n, "/robot_1/Odometry", 1),        // 初始化里程计订阅者
                                       sync_(MySyncPolicy(10), gps_sub_, odom_sub_) // 创建同步器
{
  // get the params
  n.getParam("dis_th", dis_th);
  n.getParam("NameSpace", NameSpace);
  n.getParam("RobotID", RobotID);
  n.getParam("SensorName", SensorName);
  n.getParam("UseGps", use_gps_);

  ROS_INFO("Get param dis_th = %lf", dis_th);

  reformatRobotName();

  registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

  // Subscriber
  pointCloudSubscriber_ = n.subscribe(NameSpace + "/cloud_registered", 1, &LIO_Pub::PCCallback, this);
  sync_.registerCallback(boost::bind(&LIO_Pub::OdomCallback, this, _1, _2));

  // Publisher
  pointCloudPublisher_ = n.advertise<sensor_msgs::PointCloud2>(NameSpace + "/merged_cloud_registered", 5);
  signalPublisher_ = n.advertise<std_msgs::Bool>(NameSpace + "/new_keyframe", 5);
  subMapPublisher_ = n.advertise<dislam_msgs::SubMap>(NameSpace + "/submap", 5);
}

// reformat robot name
void LIO_Pub::reformatRobotName()
{
  ROS_INFO("Check Format");
  // check format
  std::string slash = "/";
  if ((NameSpace.find(slash)) == string::npos && !NameSpace.empty())
    NameSpace = "/" + NameSpace;
  if ((SensorName.find(slash)) == string::npos && !SensorName.empty())
    SensorName = "/" + SensorName;

  ROS_INFO("Check Format Done");
}

void LIO_Pub::OdomCallback(const nav_msgs::Odometry::ConstPtr &gps_msg, const nav_msgs::Odometry::ConstPtr &odom_msg)
{
  // 初始坐标
  if (isFirstOdom)
  {
    origin_x_ = gps_msg->pose.pose.position.x;
    origin_y_ = gps_msg->pose.pose.position.y;
    origin_z_ = gps_msg->pose.pose.position.z;

    x0 = gps_msg->pose.pose.position.x; // 初始值
    y0 = gps_msg->pose.pose.position.y;
    z0 = gps_msg->pose.pose.position.z;

    isFirstOdom = false;
  }

  // 创建 GPS 消息的副本以进行修改
  nav_msgs::Odometry gps_msg_copy = *gps_msg;

  // 更新 GPS 位置
  gps_msg_copy.pose.pose.position.x -= origin_x_;
  gps_msg_copy.pose.pose.position.y -= origin_y_;
  gps_msg_copy.pose.pose.position.z -= origin_z_;

  // 使用副本更新 odom_msg
  nav_msgs::Odometry odom_msg_copy = *odom_msg;
  odom_msg_copy.pose.pose.position.x = gps_msg_copy.pose.pose.position.x;
  odom_msg_copy.pose.pose.position.y = gps_msg_copy.pose.pose.position.y;
  odom_msg_copy.pose.pose.position.z = gps_msg_copy.pose.pose.position.z;

  x1 = odom_msg_copy.pose.pose.position.x;
  y1 = odom_msg_copy.pose.pose.position.y;
  z1 = odom_msg_copy.pose.pose.position.z;

  // 计算距离
  distance = sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2) + pow((z1 - z0), 2));

  pub_TF(odom_msg_copy); // 使用副本传递给 pub_TF

  if (distance > dis_th)
  {
    // 新起点
    x0 = x1;
    y0 = y1;
    z0 = z1;

    signal_num += 1;
    Signal = true;
    pub_Signal();
    Signal = false;
    distance = 0;

    // 处理注册点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr registeredCloudBody(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Isometry3d transform = odom2isometry(odom_msg_copy); // 使用副本进行转换
    Eigen::Matrix4d transformMatrix = transform.inverse().matrix();
    pcl::transformPointCloud(*registeredCloud, *registeredCloudBody, transformMatrix);

    // 体素过滤
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    float leafSize = 0.2;
    voxel.setInputCloud(registeredCloudBody);
    voxel.setLeafSize(leafSize, leafSize, leafSize);
    voxel.filter(*registeredCloudBody);

    // 过滤直到点云大小 <= 15000
    while (registeredCloudBody->size() > 15000)
    {
      voxel.filter(*registeredCloudBody);
      leafSize += 0.1; // 增加体素大小
      voxel.setLeafSize(leafSize, leafSize, leafSize);
      voxel.setInputCloud(registeredCloudBody);
      voxel.filter(*registeredCloudBody);
    }

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*registeredCloudBody, output);
    std::string tmpNameSpace = !NameSpace.empty() && NameSpace[0] == '/' ? NameSpace.substr(1) : NameSpace;
    output.header.frame_id = tmpNameSpace + SensorName;

    // 发布子图
    dislam_msgs::SubMap submapMsg;
    submapMsg.keyframePC = output;
    submapMsg.pose.position = odom_msg_copy.pose.pose.position;       // 使用副本的位姿
    submapMsg.pose.orientation = odom_msg_copy.pose.pose.orientation; // 使用副本的四元数
    subMapPublisher_.publish(submapMsg);
    pointCloudPublisher_.publish(output);
    registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
  }
}

void LIO_Pub::PCCallback(sensor_msgs::PointCloud2 msg)
{
  pcl::PCLPointCloud2 registeredCloudPCL;
  pcl_conversions::toPCL(msg, registeredCloudPCL);
  pcl::PointCloud<pcl::PointXYZI>::Ptr localRegisteredCloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromPCLPointCloud2(registeredCloudPCL, *localRegisteredCloud);

  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(localRegisteredCloud);
  voxel.setLeafSize(0.2, 0.2, 0.2);
  voxel.filter(*localRegisteredCloud);
  *registeredCloud += *localRegisteredCloud;
  // pointCloud.push_back(localRegisteredCloud);
}

void LIO_Pub::pub_PC(sensor_msgs::PointCloud2 msg)
{
  pointCloudPublisher_.publish(msg);
}

void LIO_Pub::pub_Signal()
{
  // ROS_INFO("No. %d signal", signal_num);
  std_msgs::Bool signal;
  signal.data = Signal;
  signalPublisher_.publish(signal);
}

void LIO_Pub::pub_TF(nav_msgs::Odometry msg)
{
  static tf::TransformBroadcaster tf_broadcaster;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z));
  q.setValue(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);
  transform.setRotation(q);
  usleep(100);
  tf_broadcaster.sendTransform(tf::StampedTransform(transform, msg.header.stamp, NameSpace + "/odom", NameSpace + SensorName));
}

Eigen::Isometry3d LIO_Pub::odom2isometry(const nav_msgs::Odometry odom_msg)
{
  const auto &orientation = odom_msg.pose.pose.orientation;
  const auto &position = odom_msg.pose.pose.position;

  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "LIO_Publisher");
  ros::NodeHandle n("~");
  LIO_Pub LIO_Pub(n);
  ROS_INFO("Init well!");
  ros::spin();

  return 0;
}
