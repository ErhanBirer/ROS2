#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv
import os

class ArmDynamicController(Node):
    def __init__(self):
        super().__init__('arm_dynamic_controller')
        
        # 1. Fiziksel Parametreler (MATLAB Modelinizle Uygun)
        self.m = [0.5, 0.5, 0.5, 0.5]      # m1, m2, m3, m4
        self.L = [0.1, 0.4, 0.35, 0.1]     # L1(base), L2, L3, L4(gripper)
        self.R = [0.08, 0.05, 0.04, 0.03]
        self.g = 9.81

        # 2. Kontrol Kazançları (PID)
        # Not: Gravity compensation olduğu için Kp ve Kd değerlerini MATLAB'daki gibi yüksek tutabilirsiniz.
        self.Kp = np.array([50.0, 60.0, 60.0, 40.0])

        self.Kd = np.array([20.0, 25.0, 25.0, 15.0]) 

        self.Ki = np.array([5.0, 8.0, 8.0, 5.0])
        
        # 3. Kontrol Değişkenleri
        self.eint = np.array([0.0, 0.0, 0.0, 0.0])
        self.prev_e = np.array([0.0, 0.0, 0.0, 0.0])
        
        # 4. CSV Hazırlığı (Hızlı Yazma Modu)
        self.csv_filename = 'PID_GRAVITY.CSV'
        self.file_path = os.path.join(os.getcwd(), self.csv_filename)
        
        # Dosyayı aç ve başlıkları yaz
        self.csv_file = open(self.file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time', 
            'x_ref', 'x_act', 'y_ref', 'y_act', 'z_ref', 'z_act', 'grip_ref', 'grip_act',
            'q1_ref', 'q1_act', 'q2_ref', 'q2_act', 'q3_ref', 'q3_act', 'q4_ref', 'q4_act'
        ])

        # 5. ROS Yayıncı ve Aboneleri
        self.pubs = [
            self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10)
            for i in range(4)
        ]
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.start_time = self.get_clock().now()
        self.last_time = self.get_clock().now()
        self.get_logger().info(f"PID + Gravity Kontrolcü Başlatıldı. Kayıt: {self.csv_filename}")

    def forward_kinematics(self, q):
        """Açılardan o anki (x, y, z) konumunu hesaplar"""
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L
        
        # Omuz ve dirsek bükülmesinin yatay iz düşümü
        r_planar = L2 * np.cos(q2) + L3 * np.cos(q2 + q3)
        
        x_act = r_planar * np.cos(q1)
        y_act = r_planar * np.sin(q1)
        z_act = L1 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        grip_act = q4 # Gripper roll değeri
        
        return [x_act, y_act, z_act, grip_act]

    def inverse_kinematics(self, tx, ty, tz, tr):
        """Hedef (x, y, z) -> Hedef (q1, q2, q3, q4)"""
        L1, L2, L3, L4 = self.L
        
        q1 = np.arctan2(ty, tx)
        r = np.sqrt(tx**2 + ty**2)
        z_rel = tz - L1
        
        D = np.sqrt(r**2 + z_rel**2)
        D = np.clip(D, abs(L2 - L3), L2 + L3 - 0.001) # Erişim sınırı
        
        cos_q3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
        q3 = -np.arccos(np.clip(cos_q3, -1.0, 1.0)) # Dirsek aşağı
        
        q2 = np.arctan2(z_rel, r) + np.arctan2(L3 * np.sin(abs(q3)), L2 + L3 * np.cos(q3))
        
        return np.array([q1, q2, q3, tr])

    def calculate_gravity(self, q):
        """Yerçekimi Vektörü (G) - Robotun Pitch eklemlerine binen yük"""
        q1, q2, q3, q4 = q
        m1, m2, m3, m4 = self.m
        L1, L2, L3, L4 = self.L
        g = self.g

        # G1 (Yaw) ve G4 (Roll) yerçekimine dik olduğu için 0'dır.
        G1 = 0.0
        # G2: Omuz eklemine binen toplam moment
        G2 = (0.5*m2*g*L2 + m3*g*L2 + m4*g*L2)*np.cos(q2) + (0.5*m3*g*L3 + m4*g*L3)*np.cos(q2+q3)
        # G3: Dirsek eklemine binen toplam moment
        G3 = (0.5*m3*g*L3 + m4*g*L3)*np.cos(q2+q3)
        G4 = 0.0
        
        return np.array([G1, G2, G3, G4])

    def cb(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.0001: return

        # 1. Eklem Verilerini Oku
        try:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in joint_names]
            q_act = np.array([msg.position[i] for i in idx])
            dq_act = np.array([msg.velocity[i] for i in idx])
        except: return

        # 2. Referans Yörünge (Daire Çizimi)
        r_dist, z_height = 0.35, 0.3
        amp = 0
        target_x = r_dist + np.cos(0.4 * t)*amp
        target_y = r_dist + np.sin(0.4 * t)*amp
        target_z = z_height
        target_roll = 1.57 # Gripper sabit 90 derece
        
        qd = self.inverse_kinematics(target_x, target_y, target_z, target_roll)
        c_act = self.forward_kinematics(q_act)

        # 3. PID Hata Hesabı
        e = qd - q_act
        self.eint += e * dt
        edot = (e - self.prev_e) / dt
        self.prev_e = e.copy()

        # 4. Yerçekimi Telafisi
        G = self.calculate_gravity(q_act)

        # 5. Kontrol Yasası: tau = (Kp*e + Kd*edot + Ki*eint) + G
        tau_pid = (self.Kp * e) + (self.Kd * edot) + (self.Ki * self.eint)
        tau = tau_pid + G

        # 6. CSV'ye Yaz (Sıralama: time, cartesian_refs, cartesian_acts, joint_refs, joint_acts)
        self.csv_writer.writerow([
            t, 
            target_x, c_act[0], target_y, c_act[1], target_z, c_act[2], target_roll, c_act[3],
            qd[0], q_act[0], qd[1], q_act[1], qd[2], q_act[2], qd[3], q_act[3]
        ])
        
        if int(t*10) % 50 == 0: self.csv_file.flush() # Her 5 saniyede bir dosyaya işle

        # 7. Torkları Yayınla
        for i in range(4):
            val = Float64()
            val.data = float(tau[i])
            self.pubs[i].publish(val)

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = ArmDynamicController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
