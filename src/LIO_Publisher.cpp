#include "ros/ros.h"
#include "ros/publisher.h"
#include "ros/subscriber.h"

#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Bool.h"

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <math.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <unistd.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/pcl_macros.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/filters/passthrough.h>

#include "dislam_msgs/SubMap.h"

using namespace std;

class LIO_Pub
{
public:
  LIO_Pub(ros::NodeHandle &n);
  ~LIO_Pub();

private:
  double dis_th = 1.5;

  int signal_num = 0;

  bool isFirstOdom = true;
  bool Signal = false;

  float distance = 0.0;
  float x0 = 0.0;
  float y0 = 0.0;
  float z0 = 0.0;
  float x1 = 0.0;
  float y1 = 0.0;
  float z1 = 0.0;

  std::string NameSpace;
  std::string SensorName;

  // 这里改成 int，不再用 string
  int RobotID = 0;

  std::string tp_sub_pcd;
  std::string tp_sub_odom;

  ros::Subscriber odometrySubscriber_;
  ros::Subscriber pointCloudSubscriber_;

  ros::Publisher pointCloudPublisher_;
  ros::Publisher signalPublisher_;
  ros::Publisher subMapPublisher_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr registeredCloud;

  void OdomCallback(nav_msgs::Odometry msg);
  void PCCallback(sensor_msgs::PointCloud2 msg);

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

LIO_Pub::LIO_Pub(ros::NodeHandle &n)
{
  // get the params
  n.param<double>("dis_th", dis_th, 1.5);
  n.param<std::string>("NameSpace", NameSpace, "vehicle0");
  n.param<int>("RobotID", RobotID, 0);
  n.param<std::string>("SensorName", SensorName, "velodyne");

  n.param<std::string>("tp_sub_pcd", tp_sub_pcd, "cloud_registered");
  n.param<std::string>("tp_sub_odom", tp_sub_odom, "Odometry");

  // 防止异常 robot id
  if (RobotID < 0)
  {
    ROS_WARN("RobotID is negative: %d, reset to 0", RobotID);
    RobotID = 0;
  }

  if (RobotID > 255)
  {
    ROS_WARN("RobotID is larger than 255: %d, reset to 255", RobotID);
    RobotID = 255;
  }

  ROS_INFO("Robot_Namespace: %s", NameSpace.c_str());
  ROS_INFO("RobotID: %d", RobotID);
  ROS_INFO("SensorName: %s", SensorName.c_str());
  ROS_INFO("Subscribe pointcloud topic: %s", tp_sub_pcd.c_str());
  ROS_INFO("Subscribe odometry topic: %s", tp_sub_odom.c_str());
  ROS_INFO("Get param dis_th = %lf", dis_th);

  reformatRobotName();

  registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

  // Subscriber
  odometrySubscriber_ =
      n.subscribe(NameSpace + "/" + tp_sub_odom, 1, &LIO_Pub::OdomCallback, this);

  pointCloudSubscriber_ =
      n.subscribe(NameSpace + "/" + tp_sub_pcd, 1, &LIO_Pub::PCCallback, this);

  // Publisher
  pointCloudPublisher_ =
      n.advertise<sensor_msgs::PointCloud2>(NameSpace + "/merged_cloud_registered", 5);

  signalPublisher_ =
      n.advertise<std_msgs::Bool>(NameSpace + "/new_keyframe", 5);

  subMapPublisher_ =
      n.advertise<dislam_msgs::SubMap>(NameSpace + "/submap", 5);
}

// reformat robot name
void LIO_Pub::reformatRobotName()
{
  ROS_INFO("Check Format");

  std::string slash = "/";

  if ((NameSpace.find(slash)) == string::npos && !NameSpace.empty())
  {
    NameSpace = "/" + NameSpace;
  }

  if ((SensorName.find(slash)) == string::npos && !SensorName.empty())
  {
    SensorName = "/" + SensorName;
  }

  ROS_INFO("Formatted NameSpace: %s", NameSpace.c_str());
  ROS_INFO("Formatted SensorName: %s", SensorName.c_str());
  ROS_INFO("Check Format Done");
}

void LIO_Pub::OdomCallback(nav_msgs::Odometry msg)
{
  if (isFirstOdom)
  {
    x0 = msg.pose.pose.position.x;
    y0 = msg.pose.pose.position.y;
    z0 = msg.pose.pose.position.z;
    isFirstOdom = false;
  }

  x1 = msg.pose.pose.position.x;
  y1 = msg.pose.pose.position.y;
  z1 = msg.pose.pose.position.z;

  // Compute distance
  distance = sqrt(pow((x1 - x0), 2) +
                  pow((y1 - y0), 2) +
                  pow((z1 - z0), 2));

  pub_TF(msg);

  if (distance > dis_th)
  {
    // New start point
    x0 = x1;
    y0 = y1;
    z0 = z1;

    signal_num += 1;

    Signal = true;
    pub_Signal();
    Signal = false;

    distance = 0.0;

    pcl::PointCloud<pcl::PointXYZI>::Ptr registeredCloudBody(
        new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Isometry3d transform = odom2isometry(msg);
    Eigen::Matrix4d transformMatrix = transform.inverse().matrix();

    pcl::transformPointCloud(*registeredCloud, *registeredCloudBody, transformMatrix);

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(registeredCloudBody);
    voxel.setLeafSize(0.2, 0.2, 0.2);
    voxel.filter(*registeredCloudBody);

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*registeredCloudBody, output);

    std::string tmpNameSpace = NameSpace;
    if (!tmpNameSpace.empty() && tmpNameSpace[0] == '/')
    {
      tmpNameSpace = tmpNameSpace.substr(1);
    }

    output.header.stamp = msg.header.stamp;
    output.header.frame_id = tmpNameSpace + SensorName;

    // Publish submap
    dislam_msgs::SubMap submapMsg;
    submapMsg.keyframePC = output;
    submapMsg.pose = msg.pose;

    // 关键帧编号
    submapMsg.id = static_cast<uint32_t>(signal_num);

    // 机器人编号
    // 这里已经不用 std::stoi 了，避免 stoi 崩溃
    submapMsg.robot_id = static_cast<uint8_t>(RobotID);

    subMapPublisher_.publish(submapMsg);
    pointCloudPublisher_.publish(output);

    ROS_INFO("Publish submap: robot_id = %d, submap_id = %d, points = %lu",
             RobotID,
             signal_num,
             registeredCloudBody->size());

    registeredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
  }
}

void LIO_Pub::PCCallback(sensor_msgs::PointCloud2 msg)
{
  pcl::PCLPointCloud2 registeredCloudPCL;
  pcl_conversions::toPCL(msg, registeredCloudPCL);

  pcl::PointCloud<pcl::PointXYZI>::Ptr localRegisteredCloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  pcl::fromPCLPointCloud2(registeredCloudPCL, *localRegisteredCloud);

  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(localRegisteredCloud);
  voxel.setLeafSize(0.2, 0.2, 0.2);
  voxel.filter(*localRegisteredCloud);

  *registeredCloud += *localRegisteredCloud;
}

void LIO_Pub::pub_PC(sensor_msgs::PointCloud2 msg)
{
  pointCloudPublisher_.publish(msg);
}

void LIO_Pub::pub_Signal()
{
  std_msgs::Bool signal;
  signal.data = Signal;
  signalPublisher_.publish(signal);
}

void LIO_Pub::pub_TF(nav_msgs::Odometry msg)
{
  static tf::TransformBroadcaster tf_broadcaster;

  tf::Transform transform;
  tf::Quaternion q;

  transform.setOrigin(tf::Vector3(msg.pose.pose.position.x,
                                  msg.pose.pose.position.y,
                                  msg.pose.pose.position.z));

  q.setValue(msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w);

  transform.setRotation(q);

  usleep(100);

  std::string tmpNameSpace = NameSpace;
  if (!tmpNameSpace.empty() && tmpNameSpace[0] == '/')
  {
    tmpNameSpace = tmpNameSpace.substr(1);
  }

  tf_broadcaster.sendTransform(
      tf::StampedTransform(transform,
                           msg.header.stamp,
                           tmpNameSpace + "/odom",
                           tmpNameSpace + SensorName));
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