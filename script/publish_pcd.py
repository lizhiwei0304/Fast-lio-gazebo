#!/usr/bin/env python3
import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

class PCDPublisher:
    def __init__(self):
        rospy.init_node("pcd_repeater", anonymous=True)

        self.pcd_path = rospy.get_param("~pcd_path", "your_map.pcd")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.rate_hz = rospy.get_param("~rate", 1.0)  # 发布频率，单位 Hz

        self.pub = rospy.Publisher("/pcd_cloud", PointCloud2, queue_size=1)

        rospy.loginfo("Loading PCD: %s", self.pcd_path)
        self.cloud_o3d = o3d.io.read_point_cloud(self.pcd_path)
        self.cloud_msg = self.convert_to_ros(self.cloud_o3d)

    def convert_to_ros(self, cloud, frame_id="map"):
        points = np.asarray(cloud.points)

        if cloud.has_colors():
            colors = np.asarray(cloud.colors)
            data = []
            for p, c in zip(points, colors):
                r = int(c[0] * 255)
                g = int(c[1] * 255)
                b = int(c[2] * 255)
                rgb = (r << 16) | (g << 8) | b
                data.append([p[0], p[1], p[2], rgb])
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)
            ]
        else:
            data = points
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return pc2.create_cloud(header, fields, data)

    def start(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.cloud_msg.header.stamp = rospy.Time.now()  # 更新时间戳
            self.pub.publish(self.cloud_msg)
            rate.sleep()

if __name__ == "__main__":
    node = PCDPublisher()
    node.start()

