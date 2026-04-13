import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class MoveAndStop(Node):
    def __init__(self):
        super().__init__("move_and_stop")
        # Abone (Subscriber): Konum bilgisini dinler
        self.subscription = self.create_subscription(Odometry, "/model/full_robot/odometry", self.odom_callback, 10)
        # Yayıncı (Publisher): Hareket komutu gönderir
        self.publisher = self.create_publisher(Twist, "/model/full_robot/cmd_vel", 10)
        
        self.stop_point = 10.0  # Durma sınırı
        self.is_stopped = False # Robot durdu mu kontrolü
        
    def odom_callback(self, msg):
        # Robotun mevcut X konumunu al
        current_x = msg.pose.pose.position.x
        
        if not self.is_stopped:
            if current_x < self.stop_point:
                # Hedefe daha var, gitmeye devam et
                move_msg = Twist()
                move_msg.linear.x = 0.5
                self.publisher.publish(move_msg) # BURASI DÜZELTİLDİ: publsiher -> publisher
                self.get_logger().info(f"Gidiyor... Mevcut X : {current_x:.2f}")
            else:
                # Hedefe varıldı, hızı sıfırla
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                self.publisher.publish(stop_msg)
                self.is_stopped = True
                self.get_logger().info("HEDEFE VARILDI! Robot durduruldu.")
                
def main(args=None):
    rclpy.init(args=args)
    node = MoveAndStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
