import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry 

class PositionSubscriber(Node):
	def __init__(self):
		super().__init__("PositionSubscriber")
		
		self.subscription = self.create_subscription(
		Odometry,
		"/model/full_robot/odometry",
		self.odom_callback,
		10)
		
		
		
	def odom_callback(self,msg):
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y
		self.get_logger().info(f'Robot Konumu -> X: {x:.2f}, Y: {y:.2f}')
		
def main(args=None):
	rclpy.init(args=args)
	node = PositionSubscriber()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
