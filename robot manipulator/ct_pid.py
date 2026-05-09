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
        
        # 1. Fiziksel Parametreler (MATLAB ile aynı olmalı)
        self.m = [0.5, 0.5, 0.5, 0.5]
        self.L = [0.1, 0.4, 0.35, 0.1]
        self.R = [0.08, 0.05, 0.04, 0.03]
        self.g = 9.81

        # 2. Kontrol Kazançları
        self.Kp = np.array([50.0, 60.0, 60.0, 40.0])
        self.Kd = np.array([20.0, 25.0, 25.0, 15.0]) 
        self.Ki = np.array([5.0, 8.0, 8.0, 5.0])
        
        self.eint = np.array([0.0, 0.0, 0.0, 0.0])
        self.prev_e = np.array([0.0, 0.0, 0.0, 0.0])
        
        # 3. CSV Ayarları
        self.csv_file = open('robot_comparison_data.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time','x_ref','x_act','y_ref','y_act','z_ref','z_act','grip_ref','grip_act',
                                  'q1_ref','q1_act','q2_ref','q2_act','q3_ref','q3_act','q4_ref','q4_act'])

        # 4. ROS
        self.p1 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint1/cmd_force', 10)
        self.p2 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint2/cmd_force', 10)
        self.p3 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint3/cmd_force', 10)
        self.p4 = self.create_publisher(Float64, '/model/four_dof_arm/joint/joint4/cmd_force', 10)
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.start_time = self.get_clock().now()
        self.last_time = self.get_clock().now()

    def forward_kinematics(self, q):
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L
        r_planar = L2 * np.cos(q2) + L3 * np.cos(q2 + q3)
        x = r_planar * np.cos(q1)
        y = r_planar * np.sin(q1)
        z = L1 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3)
        return [x, y, z, q4]

    def inverse_kinematics(self, tx, ty, tz, tr):
        L1, L2, L3, L4 = self.L
        q1 = np.arctan2(ty, tx)
        r = np.sqrt(tx**2 + ty**2)
        z_rel = tz - L1
        D = np.sqrt(r**2 + z_rel**2)
        D = min(D, L2 + L3 - 0.001)
        cos_q3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
        q3 = -np.arccos(np.clip(cos_q3, -1.0, 1.0))
        q2 = np.arctan2(z_rel, r) + np.arctan2(L3 * np.sin(abs(q3)), L2 + L3 * np.cos(q3))
        return np.array([q1, q2, q3, tr])

    def calculate_dynamics(self, q, dq):
        """MATLAB'dan gelen sembolik denklemlerin Python karşılığı"""
        q1, q2, q3, q4 = q
        dq1, dq2, dq3, dq4 = dq
        m1, m2, m3, m4 = self.m
        L1, L2, L3, L4 = self.L
        R1, R2, R3, R4 = self.R
        g = self.g

        # --- M (Atalet) Matrisi Elemanları ---
        # MATLAB çıktındaki M11, M12 vb. formülleri buraya kopyalıyoruz
        # Örnek Yaw-Pitch-Pitch yapısı için M matrisi:
        M11 = 0.33*m1*R1**2 + m2*(L2*np.cos(q2))**2 + m3*(L2*np.cos(q2)+L3*np.cos(q2+q3))**2
        M22 = 0.33*m2*L2**2 + m3*(L2**2 + L3**2 + 2*L2*L3*np.cos(q3))
        M33 = 0.33*m3*L3**2 + m4*L4**2
        M44 = 0.5*m4*R4**2 # Gripper roll ataleti
        M = np.diag([M11, M22, M33, M44])

        # --- G (Yerçekimi) Vektörü ---
        G1 = 0.0
        G2 = (0.5*m2*g*L2 + m3*g*L2)*np.cos(q2) + 0.5*m3*g*L3*np.cos(q2+q3)
        G3 = 0.5*m3*g*L3*np.cos(q2+q3)
        G4 = 0.0
        G = np.array([G1, G2, G3, G4])

        # --- C (Coriolis & Centrifugal) Vektörü ---
        # Basitleştirilmiş temsil (Tam formül MATLAB çıktından alınmalı)
        C2 = -m3*L2*L3*np.sin(q3)*dq3**2 - 2*m3*L2*L3*np.sin(q3)*dq2*dq3
        C3 = m3*L2*L3*np.sin(q3)*dq2**2
        C = np.array([0.0, C2, C3, 0.0])

        return M, C, G

    def cb(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.0001: return

        try:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in joint_names]
            q_act = np.array([msg.position[i] for i in idx])
            dq_act = np.array([msg.velocity[i] for i in idx])
        except: return

        # 1. Referans ve FK
        target_x, target_y, target_z, target_roll = 0.3 + 0.05*np.cos(0.5*t), 0.3 + 0.05*np.sin(0.5*t), 0.3, 1.57
        qd = self.inverse_kinematics(target_x, target_y, target_z, target_roll)
        c_act = self.forward_kinematics(q_act)
        
        # 2. Hata ve PID
        e = qd - q_act
        self.eint += e * dt
        edot = (e - self.prev_e) / dt
        self.prev_e = e.copy()

        # 3. Dinamikleri Hesapla
        M, C, G = self.calculate_dynamics(q_act, dq_act)
        
        # 4. Computed Torque: tau = M * (Kp*e + Kd*edot + Ki*eint) + C + G
        # Not: qdpp (ivme referansı) 0 kabul edildi
        v = (self.Kp * e) + (self.Kd * edot) + (self.Ki * self.eint)
        tau = np.dot(M, v) + C + G

        # 5. CSV Kayıt
        self.csv_writer.writerow([t, target_x, c_act[0], target_y, c_act[1], target_z, c_act[2], target_roll, c_act[3],
                                  qd[0], q_act[0], qd[1], q_act[1], qd[2], q_act[2], qd[3], q_act[3]])
        if int(t*10) % 20 == 0: self.csv_file.flush()

        # 6. Yayınla
        pubs = [self.p1, self.p2, self.p3, self.p4]
        for i in range(4):
            f = Float64()
            f.data = float(tau[i])
            pubs[i].publish(f)

def main():
    rclpy.init()
    node = ArmDynamicController()
    try: rclpy.spin(node)
    except: pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
