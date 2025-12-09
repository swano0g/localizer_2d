#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math
import time

def quat_to_yaw(qx, qy, qz, qw):
    """Quaternion -> yaw (rad)"""
    # ZYX 순
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class CollisionReact(Node):

    def __init__(self):
        super().__init__("collision_react")
        self.get_logger().info("Collision react initialized")

        self.prev_time = time.time()

        # ----- Parameters -----
        self.step_z = 0.3
        self.step_xy = 0.5

        # ----- Internal State -----
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        # ----- QoS -----
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        qos_ctrl = QoSProfile(depth=10)
        qos_ctrl.reliability = ReliabilityPolicy.RELIABLE
        qos_ctrl.history = HistoryPolicy.KEEP_LAST

        # ----- Subscribers -----
        self.sub_collision = self.create_subscription(
            Bool, "/collision/detected", self.collision_callback, qos
        )

        self.sub_odom = self.create_subscription(
            Odometry, "/cf/odom", self.odom_callback, qos
        )

        # ----- Publisher -----
        self.pub_goto = self.create_publisher(
            PoseStamped, "/cf/hl/goto", qos_ctrl
        )

        self.get_logger().info("Collision React Node Started")

    # -----------------------------------
    # Odometry Callback
    # -----------------------------------
    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        # orientation → yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.yaw = quat_to_yaw(qx, qy, qz, qw)

    # -----------------------------------
    # Collision Callback
    # -----------------------------------
    def collision_callback(self, msg: Bool):
        # self.get_logger().info(f"Collision callback called: Detected result {msg.data}")
        if not msg.data or time.time() - self.prev_time < 5:
            return  # 충돌 False → 무시

        self.prev_time = time.time()
            
        if not self.have_odom:
            self.get_logger().warn("No odom yet, cannot react to collision")
            return

        # 이동할 z 계산
        new_z = self.z + (self.step_z)
        new_z = max(0.0, min(2.0, new_z))  # 안전한 범위 제한

        # 이동할 x계산
        new_x = self.x - (self.step_xy)
        new_x = max(0.0, min(2.0, new_x))

        self.get_logger().warn(
            f"[COLLISION] Detected! Moving DOWN by {self.step_z * 2.0:.2f} m {self.step_xy * 2.0:.2f}"
        )

        # Publish goto command
        cmd = PoseStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "map"
        cmd.pose.position.x = float(new_x)
        cmd.pose.position.y = float(self.y)
        cmd.pose.position.z = float(new_z)

        # yaw 유지
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cmd.pose.orientation.z = sy
        cmd.pose.orientation.w = cy
        cmd.pose.orientation.x = 0.0
        cmd.pose.orientation.y = 0.0

        self.pub_goto.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CollisionReact()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
