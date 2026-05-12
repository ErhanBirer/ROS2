#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv
import os

class MLP_IdealController(Node):
    def __init__(self):
        super().__init__("mlp_ideal_controller")
        
        # --- 1. KONTROL KAZANÇLARI ---
        self.lam = np.diag([10.5, 47.0, 33.0, 10.4])
        self.Kv = np.diag([3.0, 10.5, 9.5, 1.0]) / 10.0
        
        # --- 2. MLP PARAMETRELERİ (MATLAB İdeal Case Uyumu) ---
        self.H = 10   # Gizli katman nöron sayısı
        self.K = 4    # Çıkış sayısı (4 Eklem)
        self.I = 20   # Giriş sayısı [q, qd, dqd, e, ep]
        
        self.F = 0.01    # W öğrenme hızı
        self.G = 0.01    # V öğrenme hızı
        self.kappa = 0.01  # e-modification katsayısı
        self.Zb =0.01    # Robustness sınırı
        self.Kz = np.eye(self.K) * 0.0001
        
        # --- 3. AĞIRLIK MATRİSLERİ ---
        # W: (H+1 x K) -> 11x4 matris
        self.W = np.zeros((self.H + 1, self.K)) 
        # V: (I x H) -> 20x10 matris
        self.V = np.random.randn(self.I, self.H) * 0.01 
        
        # --- 4. ZAMANLAMA VE KAYIT ---
        self.last_time = None
        self.start_time = self.get_clock().now()
        # CSV dosyasının tam yolunu buraya da yazabilirsin
        self.csv_filename = 'robot_mlp_results_full.csv'
        self.init_csv()
        
        # --- 5. ROS 2 İLETİŞİM ---
        # Gazebo/Robot eklem tork yayıncıları
        self.pubs = [self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10) for i in range(4)]
        # Eklem durum aboneliği
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.get_logger().info("--- MLP Kontrolcü Başlatıldı: MATLAB Analiz Modu Aktif ---")

    def init_csv(self):
        """MATLAB'daki readtable ve VariableNamingRule için uygun başlıkları oluşturur."""
        header = ['time']
        # Takip verileri
        for i in range(1, 5):
            header.extend([f'q{i}_ref', f'q{i}_act', f'tau{i}'])
        
        # W Ağırlıkları (W_0_0, W_0_1...)
        for h in range(self.H + 1):
            for k in range(self.K):
                header.append(f'W_{h}_{k}')
        
        # V Ağırlıkları (V_0_0, V_0_1...)
        for i in range(self.I):
            for h in range(self.H):
                header.append(f'V_{i}_{h}')
        
        with open(self.csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

    def cb(self, msg):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0: return
        self.last_time = now

        # --- ADIM 1: Durum Okuma ---
        try:
            names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in names]
            q = np.array([msg.position[i] for i in idx])
            dq = np.array([msg.velocity[i] for i in idx])
        except (ValueError, IndexError):
            return

        # --- ADIM 2: Referans Yörünge ---
        qd = 0.5 * np.sin(2 * np.pi * 0.1 * t) * np.ones(4)
        dqd = 0.5 * (2 * np.pi * 0.1) * np.cos(2 * np.pi * 0.1 * t) * np.ones(4)

        # --- ADIM 3: Kayma Yüzeyi (r) ---
        e = qd - q
        ep = dqd - dq
        r = ep + self.lam @ e

        # --- ADIM 4: MLP Forward Pass ---
        x_in = np.concatenate([q, qd, dqd, e, ep]).reshape(1, -1) 
        net_in = x_in @ self.V 
        sigma_x = self.sigmoid(net_in).reshape(-1, 1) 
        sigma_x_new = np.vstack([sigma_x, [1.0]]) # Bias terimi (11x1)

        # NN Çıkışı
        tau_nn = (self.W.T @ sigma_x_new).flatten()

        # --- ADIM 5: Robustness Terimi (v_t) ---
        # Matmul hatası giderildi: Skaler çarpanlar için '*' kullanıldı
        Z_f_norm = np.sqrt(np.sum(self.W**2) + np.sum(self.V**2))
        v_t = -(Z_f_norm + self.Zb) * (self.Kz @ r)

        # --- ADIM 6: Adaptif Güncelleme (Lyapunov Based) ---
        norm_r = np.linalg.norm(r)
        
        # W_dot güncelleme
        w_dot = self.F * (sigma_x_new @ r.reshape(1, -1)) - (self.kappa * self.F * norm_r * self.W)
        
        # V_dot güncelleme
        v_dot = self.G * (x_in.T @ sigma_x.T) * norm_r - (self.kappa * self.G * norm_r * self.V)

        # Euler Entegrasyonu
        self.W += w_dot * dt
        self.V += v_dot * dt

        # --- ADIM 7: Kontrol Yasası ---
        u = tau_nn + (self.Kv @ r) - v_t

        # --- ADIM 8: Yayınla ve Kaydet ---
        for i in range(4):
            val = Float64()
            # Sistemin patlamaması için makul bir tork sınırı
            val.data = float(np.clip(u[i], -200000000000.0, 20000000000.0))
            self.pubs[i].publish(val)
        
        self.log_to_csv(t, qd, q, u)

    def log_to_csv(self, t, qd, q_act, u):
        """Verileri CSV'ye ekler."""
        row = [t]
        for i in range(4):
            row.extend([qd[i], q_act[i], u[i]])
        
        row.extend(self.W.flatten().tolist())
        row.extend(self.V.flatten().tolist())
        
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except:
            pass

def main():
    rclpy.init()
    node = MLP_IdealController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Kontrolcü kapatılıyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
