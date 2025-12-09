import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Pose2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class CollisionDetect(Node):
    def __init__(self):
        super().__init__('collision_detector')

        # hyperparameters
        self.distance_threshold = 640  # 픽셀 단위 거리 임계값
        
        self.get_logger().info("Collision Detect is Running")

        self.drone_pos = None  # (x, y)
        self.object_pos = None  # (x, y)
        self.drone_hw = None 
        self.object_hw = None

        qos_profile = qos_profile_sensor_data

        # Parameter for detection
        self.drone_id = "drone"
        self.object_id = "ball"

        # Subscribers
        self.create_subscription(Detection2DArray, '/detections_bb', self.boundingbox_callback, qos_profile)

        # Publishers
        self.pub_collision = self.create_publisher(Bool, '/collision/detected', qos_profile)
  
    def boundingbox_callback(self, msg):
        self.drone_pos = None
        self.object_pos = None

        for det in msg.detections:
            cx = det.bbox.center.position.x 
            cy = det.bbox.center.position.y 
            w = det.bbox.size_x
            h = det.bbox.size_y
            
            object_class = det.results[0].hypothesis.class_id 
            
            if object_class == self.drone_id:
                self.drone_pos = (cx, cy)
                self.drone_hw = (h, w)
                
            elif object_class == self.object_id:
                self.object_pos = (cx, cy)
                self.object_hw = (h, w)
        
        will_collide = self.will_collide()
        
        self.publish_collisions(will_collide)
        
    def publish_collisions(self, collision):
        msg = Bool()
        msg.data = collision
        self.pub_collision.publish(msg)
  
    def will_collide(self):
        # drone과 object 위치가 모두 감지되었는지 확인
        if self.drone_pos is None or self.object_pos is None:
            return False
        
        # 유클리드 거리 계산
        distance = np.sqrt(
            (self.drone_pos[0] - self.object_pos[0]) ** 2 + 
            (self.drone_pos[1] - self.object_pos[1]) ** 2
        )
        
        # 거리가 임계값보다 작으면 충돌 위험
        if distance < self.distance_threshold:
            self.get_logger().warn(f"Collision risk! Distance: {distance:.2f}")
            return True
        
        return False


def main(args=None):
    rclpy.init(args=args)

    node = CollisionDetect()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




# import rclpy
# import numpy as np
# from rclpy.node import Node
# from rclpy.qos import qos_profile_sensor_data
# from std_msgs.msg import Bool
# from vision_msgs.msg import Detection2DArray
# from geometry_msgs.msg import Pose2D
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


# class CollisionDetect(Node):
#     def __init__(self):
#         # hyperparameters
#         super().__init__('collision_detector')

#         self.distance_threshold = 10
#         self.speed_threshold = 1
        

#         self.get_logger().info("Collision Detect is Running")

#         self.drone_queue = [] # List[(x, y)]
#         self.object_queue = [] # List [(x, y)]

#         # Configure QoS profile for publishing and subscribing
#         # qos_profile = QoSProfile(  # https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Quality-of-Service-Settings.html#qos-policies
#         #     reliability=ReliabilityPolicy.RELIABLE,
#         #     durability=DurabilityPolicy.VOLATILE,
#         #     history=HistoryPolicy.KEEP_LAST,
#         #     depth=1
#         # )

#         qos_profile = qos_profile_sensor_data

#         # Parameter for detection
#         self.drone_id = "drone"
#         self.object_id = "ball"

#         # Subscribers
#         self.create_subscription(Detection2DArray, '/detections_bb', self.boundingbox_callback, qos_profile) # when bounding box information got

#         # Publishers
#         self.pub_collision = self.create_publisher(Bool, '/collision/detected', qos_profile)
  
#     def boundingbox_callback(self, msg):
#         for det in msg.detections:
#             cx = det.bbox.center.position.x 
#             cy = det.bbox.center.position.y 
            
#             object_class = det.results[0].hypothesis.class_id 
            
#             if object_class == self.drone_id:
#                 self.drone_queue.append((cx, cy))
                
#             elif object_class == self.object_id:
#                 self.object_queue.append((cx, cy))
        
#         will_collide = self.will_collide()
        
#         self.publish_collisions(will_collide)
        
#     def publish_collisions(self, collision):
#         msg = Bool()
#         msg.data = collision

#         self.pub_collision.publish(msg)
  
#     def will_collide(self):
#         # only calculate if there are enough number of datas in the queue
#         if len(self.object_queue) <= 1 or len(self.drone_queue) <= 1:
#             return False 
        
#         ball_pos_1 = self.object_queue[-1]
#         ball_pos_2 = self.object_queue[-2]
        
#         speed = np.sqrt((ball_pos_1[1] - ball_pos_2[1]) ** 2 + (ball_pos_1[0] - ball_pos_2[0]) ** 2)

#         drone_pos = self.drone_queue[-1]

#         m = (ball_pos_1[1] - ball_pos_2[1]) / (ball_pos_1[0] - ball_pos_2[0])
#         c = ball_pos_1[1] - m * ball_pos_1[0]

#         d = abs(drone_pos[1] - m * drone_pos[0] + c) / np.sqrt(1 + m ** 2)

#         d_ball_pos_1_drone = np.sqrt((ball_pos_1[1] - drone_pos[1]) ** 2 + (ball_pos_1[0] - drone_pos[0]) ** 2)
#         d_ball_pos_2_drone = np.sqrt((ball_pos_2[1] - drone_pos[1]) ** 2 + (ball_pos_2[0] - drone_pos[0]) ** 2)

#         # is_getting_close = (0 < d_ball_pos_2_drone - d_ball_pos_1_drone)

        

#         if d < self.distance_threshold and speed > self.speed_threshold and is_getting_close:
#             self.get_logger().info("Collision Detected")
#             return True 
#         return False


# def main(args=None):
#     rclpy.init(args=args)

#     node = CollisionDetect()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
