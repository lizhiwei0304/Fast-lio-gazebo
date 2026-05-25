#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from dislam_msgs.msg import SubMap, NewSubMap  # 确保你已编译出 NewSubMap.msg


class SubMapConverter:
    def __init__(self):
        self.robot_id = rospy.get_param("~robot_id", 1)
        self.in_topic = rospy.get_param("~in_topic", "/robot_1/submap")
        self.out_topic = rospy.get_param("~out_topic", "/robot_1/new_submap")
        self.start_id = rospy.get_param("~start_id", 0)

        self._counter = int(self.start_id)

        self.pub = rospy.Publisher(self.out_topic, NewSubMap, queue_size=10)
        self.sub = rospy.Subscriber(self.in_topic, SubMap, self.cb, queue_size=50)

        rospy.loginfo("[submap_converter] robot_id=%d in=%s out=%s start_id=%d",
                      self.robot_id, self.in_topic, self.out_topic, self.start_id)

    def cb(self, msg: SubMap):
        out = NewSubMap()
        out.robot_id = int(self.robot_id)
        out.id = int(self._counter)

        # 原样复制
        out.keyframePC = msg.keyframePC
        out.orthoImage = msg.orthoImage
        out.pose = msg.pose

        # describe 不管：保持默认值即可（不赋也行）
        # out.describe = out.describe

        self.pub.publish(out)
        self._counter += 1


def main():
    rospy.init_node("submap_to_newsubmap")
    SubMapConverter()
    rospy.spin()


if __name__ == "__main__":
    main()

