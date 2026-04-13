import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class LidarAvoidance(Node):
    def __init__(self):
        super().__init__("lidar_avoidance_node")
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.publisher = self.create_publisher(Twist, "/model/full_robot/cmd_vel", 10)
        self.get_logger().info("Kontrol Düğümü Başlatıldı. Veri bekleniyor...")

    def lidar_callback(self, msg):
        # Ön bölgeyi (0-180 derece arası) tara
        front_ranges = msg.ranges[0:90]
        self.get_logger().info(f"Lidar verisi alındı! İlk mesafe: {msg.ranges[0]}")
        valid_ranges = [r for r in front_ranges if r > msg.range_min and r < msg.range_max]
        min_dist = min(valid_ranges) if valid_ranges else 10.0
        
        cmd = Twist()
        if min_dist < 0.8:
            self.get_logger().warn(f"ENGEL! Mesafe: {min_dist:.2f}m. Manevra yapılıyor.")
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
        else:
            self.get_logger().info(f"Yol temiz. Mesafe: {min_dist:.2f}m.")
            cmd.linear.x = 0.0
            cmd.angular.z = 3.0
        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LidarAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
