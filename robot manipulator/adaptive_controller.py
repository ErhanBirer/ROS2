#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv
import matplotlib.pyplot as plt

class AdaptiveArmController(Node):
    def __init__(self):
        super().__init__("adaptive_arm_controller")
        
        # 1. Fiziksel Sabitler (URDF ile uyumlu)
        self.L = [0.1, 0.4, 0.35, 0.1]
        self.g = 9.81
        
        # 2. Kontrol ve Adaptasyon Kazançları (Boşluklar düzeltildi)
        # lam (Lambda): Hata takip hassasiyeti
        self.lam = np.diag([15.1, 15.1, 15.1, 15.1]) /200.0

        # Kv (Sönümleme): Robotun titremesini durdurmak için sönümleme terimi
        self.Kv = np.diag([15.0, 15.0, 15.0, 12.0]) / 200.0 

        # Gamma (Adaptasyon): Kütle tahmini hızı
        self.Gamma = np.diag([0.001, 0.001, 0.001, 0.001]) 

        # Kütleleri doğrudan 0.45 yap ve kımıldatmasın (Başlangıç tahmini)
        self.psi_hat = np.array([0.45, 0.45, 0.45, 0.45])
        
        # 3. Referans Yörünge Parametreleri [Genlik, Frekans, Ofset]
        self.traj_params = {
            0: [0.6, 0.1, 0.0],  1: [0.4, 0.15, 0.2], 
            2: [0.3, 0.2, -0.4], 3: [0.5, 0.25, 0.1]
        }
        
        # 4. Veri Depolama
        self.history = {'t': [], 'e': [], 'm_hat': [], 'tau': []}
        
        # 5. ROS İletişimi
        self.pubs = [self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10) for i in range(4)]
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        # 6. CSV Hazırlığı
        self.csv_filename = 'ROBOT_ADAPTIVE_RESULTS.csv'
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'e1', 'e2', 'e3', 'e4', 'm1_hat', 'm2_hat', 'm3_hat', 'm4_hat', 'tau1', 'tau2'])
        
        self.start_time = self.get_clock().now()
        self.last_time = self.get_clock().now()
        print("[BİLGİ] İndentasyon hataları düzeltildi. Kontrolcü aktif.")

    def calculate_W(self, q, dq, dqr, ddqr):
        W = np.zeros((4, 4))
        L1, L2, L3, L4 = self.L
        g = self.g

        phi1_s = q[1]
        phi2_s = q[2]
        phi1_dot = dq[1]
        phi2_dot = dq[2]
        theta1_dot = dq[0]
        
        theta1_ddot = ddqr[0]
        phi1_ddot = ddqr[1]
        phi2_ddot = ddqr[2]
        theta4_ddot = ddqr[3]

        W[0, 0] = (L1**2 * theta1_ddot) / 12
        W[0, 1] = (L2**2 * (theta1_ddot - theta1_ddot * np.cos(2 * phi1_s) + 2 * phi1_dot * theta1_dot * np.sin(2 * phi1_s))) / 8
        W[0, 2] = (L2**2 * theta1_ddot) / 2 + (L3**2 * theta1_ddot) / 8 - (L2**2 * theta1_ddot * np.cos(2 * phi1_s)) / 2 - \
                  (L3**2 * theta1_ddot * np.cos(2 * phi1_s + 2 * phi2_s)) / 8 + L2**2 * phi1_dot * theta1_dot * np.sin(2 * phi1_s) + \
                  (L2 * L3 * theta1_ddot * np.cos(phi2_s)) / 2 + (L3**2 * phi1_dot * theta1_dot * np.sin(2 * phi1_s + 2 * phi2_s)) / 4 + \
                  (L3**2 * phi2_dot * theta1_dot * np.sin(2 * phi1_s + 2 * phi2_s)) / 4 - (L2 * L3 * theta1_ddot * np.cos(2 * phi1_s + phi2_s)) / 2 - \
                  (L2 * L3 * phi2_dot * theta1_dot * np.sin(phi2_s)) / 2 + L2 * L3 * phi1_dot * theta1_dot * np.sin(2 * phi1_s + phi2_s) + \
                  (L2 * L3 * phi2_dot * theta1_dot * np.sin(2 * phi1_s + phi2_s)) / 2
        W[0, 3] = (L2**2 * theta1_ddot) / 2 + (L3**2 * theta1_ddot) / 2 - (L2**2 * theta1_ddot * np.cos(2 * phi1_s)) / 2 - \
                  (L3**2 * theta1_ddot * np.cos(2 * phi1_s + 2 * phi2_s)) / 2 + L2**2 * phi1_dot * theta1_dot * np.sin(2 * phi1_s) + \
                  L2 * L3 * theta1_ddot * np.cos(phi2_s) + L3**2 * phi1_dot * theta1_dot * np.sin(2 * phi1_s + 2 * phi2_s) + \
                  L3**2 * phi2_dot * theta1_dot * np.sin(2 * phi1_s + 2 * phi2_s) - L2 * L3 * theta1_ddot * np.cos(2 * phi1_s + phi2_s) - \
                  L2 * L3 * phi2_dot * theta1_dot * np.sin(phi2_s) + 2 * L2 * L3 * phi1_dot * theta1_dot * np.sin(2 * phi1_s + phi2_s) + \
                  L2 * L3 * phi2_dot * theta1_dot * np.sin(2 * phi1_s + phi2_s)

        W[1, 1] = (L2**2 * phi1_ddot) / 3 - (L2**2 * theta1_dot**2 * np.sin(2 * phi1_s)) / 8 - (L2 * g * np.sin(phi1_s)) / 2
        W[1, 2] = L2**2 * phi1_ddot + (L3**2 * phi1_ddot) / 4 + (L3**2 * phi2_ddot) / 4 - (L2**2 * theta1_dot**2 * np.sin(2 * phi1_s)) / 2 - \
                  (L3**2 * theta1_dot**2 * np.sin(2 * phi1_s + 2 * phi2_s)) / 8 - (L3 * g * np.sin(phi1_s + phi2_s)) / 2 - L2 * g * np.sin(phi1_s) + \
                  L2 * L3 * phi1_ddot * np.cos(phi2_s) + (L2 * L3 * phi2_ddot * np.cos(phi2_s)) / 2 - (L2 * L3 * phi2_dot**2 * np.sin(phi2_s)) / 2 - \
                  (L2 * L3 * theta1_dot**2 * np.sin(2 * phi1_s + phi2_s)) / 2 - L2 * L3 * phi1_dot * phi2_dot * np.sin(phi2_s)
        W[1, 3] = L2**2 * phi1_ddot + L3**2 * phi1_ddot + L3**2 * phi2_ddot - (L2**2 * theta1_dot**2 * np.sin(2 * phi1_s)) / 2 - \
                  (L3**2 * theta1_dot**2 * np.sin(2 * phi1_s + 2 * phi2_s)) / 2 - L3 * g * np.sin(phi1_s + phi2_s) - L2 * g * np.sin(phi1_s) + \
                  2 * L2 * L3 * phi1_ddot * np.cos(phi2_s) + L2 * L3 * phi2_ddot * np.cos(phi2_s) - L2 * L3 * phi2_dot**2 * np.sin(phi2_s) - \
                  L2 * L3 * theta1_dot**2 * np.sin(2 * phi1_s + phi2_s) - 2 * L2 * L3 * phi1_dot * phi2_dot * np.sin(phi2_s)

        W[2, 2] = (L3 * (6 * L3 * phi1_ddot + 8 * L3 * phi2_ddot - 12 * g * np.sin(phi1_s + phi2_s) + 12 * L2 * phi1_dot**2 * np.sin(phi2_s) + \
                  6 * L2 * theta1_dot**2 * np.sin(phi2_s) - 6 * L2 * theta1_dot**2 * np.sin(2 * phi1_s + phi2_s) + 12 * L2 * phi1_ddot * np.cos(phi2_s) - \
                  3 * L3 * theta1_dot**2 * np.sin(2 * phi1_s + 2 * phi2_s))) / 24
        W[2, 3] = (L3 * (2 * L3 * phi1_ddot + 2 * L3 * phi2_ddot - 2 * g * np.sin(phi1_s + phi2_s) + 2 * L2 * phi1_dot**2 * np.sin(phi2_s) + \
                  L2 * theta1_dot**2 * np.sin(phi2_s) - L2 * theta1_dot**2 * np.sin(2 * phi1_s + phi2_s) + 2 * L2 * phi1_ddot * np.cos(phi2_s) - \
                  L3 * theta1_dot**2 * np.sin(2 * phi1_s + 2 * phi2_s))) / 2

        W[3, 3] = (L4**2 * theta4_ddot) / 12
        return W

    def cb(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        if dt <= 0.0005: return
        if dt > 0.05: dt = 0.01 

        try:
            names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in names]
            q_act = np.array([msg.position[i] for i in idx])
            dq_act = np.array([msg.velocity[i] for i in idx])
        except (ValueError, IndexError): 
            return

        qd, dqd, ddqd = np.zeros(4), np.zeros(4), np.zeros(4)
        for i in range(4):
            A, f, b = self.traj_params[i]
            qd[i] = A * np.sin(2 * np.pi * f * t) + b
            dqd[i] = A * (2 * np.pi * f) * np.cos(2 * np.pi * f * t)
            ddqd[i] = -A * (2 * np.pi * f)**2 * np.sin(2 * np.pi * f * t)

        e = qd - q_act
        de = dqd - dq_act
        s = de + self.lam @ e  
        
        dqr = dqd + self.lam @ e
        ddqr = ddqd + self.lam @ de

        W = self.calculate_W(q_act, dq_act, dqr, ddqr)
        
        # Parametre Adaptasyon Kanunu
        self.psi_hat += (self.Gamma @ (W.T @ s)) * dt
        self.psi_hat = np.clip(self.psi_hat, 0.1, 2.0) 
        
        tau = (W @ self.psi_hat) + (self.Kv @ s)

        self.history['t'].append(t)
        self.history['e'].append(e.copy())
        self.history['m_hat'].append(self.psi_hat.copy())
        self.history['tau'].append(tau.copy())

        for i in range(4):
            val = Float64()
            val.data = float(np.clip(tau[i], -150.0, 150.0))
            self.pubs[i].publish(val)
        
        self.csv_writer.writerow([t, *e, *self.psi_hat, tau[0], tau[1]])

    def plot_results(self):
        if not self.history['t']: return
        t = np.array(self.history['t'])
        e = np.array(self.history['e'])
        m = np.array(self.history['m_hat'])
        tau = np.array(self.history['tau'])
        colors = ['r', 'g', 'b', 'm']

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)
        for i in range(4): plt.plot(t, e[:, i], label=f'Hata q{i+1}', color=colors[i])
        plt.title("Eklem Takip Hataları"); plt.grid(True); plt.legend()

        plt.subplot(3, 1, 2)
        for i in range(4): plt.plot(t, m[:, i], label=f'Tahmini m{i+1}', color=colors[i], lw=2)
        plt.axhline(y=0.45, color='black', linestyle='--', label='Gerçek Kütle (0.45kg)')
        plt.ylabel("Kütle (kg)"); plt.title("Adaptif Parametre Tahminleri"); plt.grid(True); plt.legend()

        plt.subplot(3, 1, 3)
        for i in range(4): plt.plot(t, tau[:, i], label=f'Tork {i+1}', color=colors[i])
        plt.title("Uygulanan Kontrol Torkları"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.show()

def main():
    rclpy.init()
    node = AdaptiveArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[BİLGİ] Simulasyon durduruldu. Grafikler çiziliyor...")
    finally:
        node.csv_file.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        node.plot_results()

if __name__ == '__main__':
    main()
