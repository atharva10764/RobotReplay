#!/usr/bin/env python3

import copy
import math
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String


class LocalizationJumpInjector(Node):

    def __init__(self):
        super().__init__("robotreplay_localization_jump_injector")

        self.pose = None

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.pose_callback,
            10,
        )

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            10,
        )

        self.event_pub = self.create_publisher(
            String,
            "/robotreplay/fault_event",
            10,
        )

        self.timer = self.create_timer(
            0.1,
            self.inject_when_ready,
        )

        self.injected = False

    def pose_callback(self, msg):
        self.pose = msg

    def inject_when_ready(self):

        if self.injected or self.pose is None:
            return

        original = self.pose
        injected = copy.deepcopy(original)

        # Controlled localization fault:
        # shift estimated robot position by +2 m in map X.
        injected.header.stamp = self.get_clock().now().to_msg()
        injected.header.frame_id = "map"

        injected.pose.pose.position.x += 2.0

        self.pose_pub.publish(injected)

        event = String()

        now = self.get_clock().now()
        stamp = now.nanoseconds / 1e9

        event.data = (
            f"LOCALIZATION_JUMP;"
            f"offset_x=2.0;"
            f"ros_time={stamp:.6f}"
        )

        self.event_pub.publish(event)

        self.get_logger().warn(
            "Injected localization jump: +2.0 m in map X"
        )

        self.injected = True

        # Allow DDS time to deliver the event before shutdown.
        self.create_timer(1.0, self.shutdown_once)

    def shutdown_once(self):
        rclpy.shutdown()


def main():

    rclpy.init()

    node = LocalizationJumpInjector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
