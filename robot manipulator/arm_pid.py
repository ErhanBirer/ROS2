#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class ArmTorqueController(Node):
    def __init__(self):
        super().__init__('arm_pid_controller')
        
        # PID Parametreleri [Kp, Ki, Kd]
        self.pid_map = {
            'joint1': [100.0, 1.0, 15.0],
            'joint2': [1500.0, 10.0, 60.0],
            'joint3': [600.0, 5.0, 20.0],
            'joint4': [50.0, 0.5, 5.0]
        }

        # Hedef açılar (Radyan)
        self.targets = [0.8, -0.4, 0.6, 0.0] 
        self.prev_errors = [0.0, 0.0, 0.0, 0.0]
        self.integrals = [0.0, 0.0, 0.0, 0.0]
        
        self.p1 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint1/cmd_force', 10)
        self.p2 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint2/cmd_force', 10)
        self.p3 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint3/cmd_force', 10)
        self.p4 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint4/cmd_force', 10)

        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        self.last_time = self.get_clock().now()

    def cb(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        
        if dt < 0.001: return

        try:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in joint_names]
            curr = [msg.position[i] for i in idx]
        except (ValueError, IndexError):
            return

        pubs = [self.p1, self.p2, self.p3, self.p4]
        
        # Terminale başlık yazdır (Okunabilirlik için)
        self.get_logger().info("-" * 50)
        
        for i, name in enumerate(joint_names):
            kp, ki, kd = self.pid_map[name]
            error = self.targets[i] - curr[i]
            
            self.integrals[i] += error * dt
            deriv = (error - self.prev_errors[i]) / dt
            
            torque = (kp * error) + (ki * self.integrals[i]) + (kd * deriv)
            
            # Torku yayınla
            msg_out = Float64()
            msg_out.data = float(torque)
            pubs[i].publish(msg_out)
            
            # Terminale Açı ve Hata bilgilerini yazdır
            self.get_logger().info(
                f"[{name}] Hedef: {self.targets[i]:.2f} | Mevcut: {curr[i]:.2f} | Hata: {error:.4f}"
            )
            
            self.prev_errors[i] = error

        self.last_time = now

def main(args=None):
    rclpy.init(args=args)
    node = ArmTorqueController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
