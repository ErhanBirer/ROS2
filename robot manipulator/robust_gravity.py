#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv
import os

class RobustArmController(Node):
    def __init__(self):
        super().__init__("robust_arm_controller")
        
        # 1. Fiziksel Parametreler
        self.m = [0.5, 0.5, 0.5, 0.5]
        self.L = [0.1, 0.4, 0.35, 0.1]
        self.R = [0.08, 0.05, 0.04, 0.03]
        self.g = 9.81
        
        # 2. Robust Kontrol Kazançları
        self.lam = np.diag([5.0, 5.0, 5.0, 5.0])    
        self.Kv = np.diag([25.0, 25.0, 25.0, 25.0]) 
        self.epsilon = 0.2                         
        
        # --- YÖRÜNGE PARAMETRELERİ (A * sin(2*pi*f*t) + b) ---
        # Buradaki değerleri her eklem için ayrı ayrı güncelleyebilirsin
        self.traj_params = {
            # Joint: [Genlik(A), Frekans(f), Ofset(b)]
            0: [0.5, 0.1, 0.0],  # Joint 1 (Yaw)
            1: [0.3, 0.1, 0.5],  # Joint 2 (Pitch 1)
            2: [0.4, 0.1, -0.5], # Joint 3 (Pitch 2)
            3: [0.8, 0.2, 0.0]   # Joint 4 (Roll)
        }
        
        # 3. Parametrik Üst Sınırlar
        self.mu2, self.vB = self.calculate_parametric_bounds()
        
        # 4. Kontrol Değişkenleri
        self.prev_e = np.zeros(4)
        
        # 5. CSV Kayıt Hazırlığı
        self.csv_filename = 'ROBUST_JOINT_SINE_TRACKING.csv'
        self.file_path = os.path.join(os.getcwd(), self.csv_filename)
        self.csv_file = open(self.file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time', 'x_ref', 'x_act', 'y_ref', 'y_act', 'z_ref', 'z_act',
            'q1_ref', 'q1_act', 'q2_ref', 'q2_act', 'q3_ref', 'q3_act', 'tau1', 'tau2'
        ])
        
        # 6. ROS Yayıncı ve Aboneleri
        self.pubs = [
            self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10)
            for i in range(4)
        ]
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.start_time = self.get_clock().now()
        self.last_time = self.get_clock().now()
        self.get_logger().info(f"Eklem Bazlı Sinüs Takibi Başladı. mu2={self.mu2:.2f}, vB={self.vB:.2f}")

    def calculate_parametric_bounds(self):
        m, L = self.m, self.L
        mu2 = abs(0.083*L[0]**2*m[0] + 0.333*L[1]**2*m[1] + L[1]**2*m[2] + 0.333*L[2]**2*m[2])
        vB = 3.0 * abs(L[3]*m[3]*(L[1] + L[2]))
        return mu2, vB
        
    def calculate_gravity(self, q):
        q2, q3 = q[1], q[2]
        m, L, g = self.m, self.L, self.g
        G = np.zeros(4)
        G[1] = (0.5*m[1]*g + m[2]*g*L[1] + m[3]*g*L[1])*np.cos(q2) + \
               (0.5*m[2]*g*L[2] + m[3]*g*L[2])*np.cos(q2+q3)
        G[2] = (0.5*m[2]*g*L[2] + m[3]*g*L[2])*np.cos(q2+q3)
        return G

    def forward_kinematics(self, q):
        q1, q2, q3, _ = q
        L1, L2, L3, _ = self.L
        r_planar = L2 * np.cos(q2) + L3 * np.cos(q2 + q3)
        x = r_planar * np.cos(q1)
        y = r_planar * np.sin(q1)
        z = L1 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        return [x, y, z]

    def cb(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001: return    
        
        try:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in joint_names]
            q_act = np.array([msg.position[i] for i in idx])
            dq_act = np.array([msg.velocity[i] for i in idx])
        except: return
        
        # --- Eklem Uzayı Yörünge Hesaplama ---
        qd = np.zeros(4)
        dqd = np.zeros(4)
        ddqd = np.zeros(4)

        for i in range(4):
            A, f, b = self.traj_params[i]
            # q = A*sin(2*pi*f*t) + b
            qd[i] = A * np.sin(2 * np.pi * f * t) + b
            # dq = A * 2*pi*f * cos(2*pi*f*t)
            dqd[i] = A * (2 * np.pi * f) * np.cos(2 * np.pi * f * t)
            # ddq = -A * (2*pi*f)^2 * sin(2*pi*f*t)
            ddqd[i] = -A * (2 * np.pi * f)**2 * np.sin(2 * np.pi * f * t)
        
        # --- Robust Kontrol Motoru ---
        e = qd - q_act
        edot = dqd - dq_act
        r = edot + self.lam @ e
        f_sig_p = ddqd + self.lam @ edot 
        
        norm_r = np.linalg.norm(r)
        F_val = self.mu2 * np.linalg.norm(f_sig_p) + \
                self.vB * np.linalg.norm(dq_act) * norm_r
                
        div = max(norm_r, self.epsilon)
        v = -(r / div) * F_val
        G = self.calculate_gravity(q_act)
        
        # --- Final Tork: Kv*r + G - v ---
        tau = (self.Kv @ r) + G - v

        # --- Yayınlama ve Kayıt ---
        c_act = self.forward_kinematics(q_act)
        c_ref = self.forward_kinematics(qd) # Referans kartezyen konumu
        for i in range(4):
            val = Float64()
            val.data = float(np.clip(tau[i], -60.0, 60.0))
            self.pubs[i].publish(val)

        # CSV Kaydı
        self.csv_writer.writerow([
            t, c_ref[0], c_act[0], c_ref[1], c_act[1], c_ref[2], c_act[2],
            qd[0], q_act[0], qd[1], q_act[1], qd[2], q_act[2], tau[0], tau[1]
        ])
        if int(t*10) % 50 == 0: self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            self.get_logger().info('CSV dosyası kapatıldı.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobustArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Durduruluyor...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
