#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv

class FLNN_DataLoggerController(Node):
    def __init__(self):
        super().__init__("flnn_data_logger_controller")
        
        # 1. Kontrol Kazançları
        # Joint 4 için Kv'yi 0.1'den biraz yukarı (örneğin 1.0-2.0) çekmen gerekebilir 
        # ama önce filtreli sonucu görelim.
        self.lam = np.diag([5.0, 30.0, 20.0, 5.0])  
        self.Kv = np.diag([1.0, 1.5, 0.5, 0.1])
        
        # 2. FLNN Parametreleri
        self.H = 12       
        self.F = 0.01   # Öğrenme hızını gürültü azalana kadar düşük tutuyoruz
        self.W_weights = np.zeros((self.H, 4))
        
        # 3. Filtreleme Parametreleri (Low Pass Filter)
        # alpha: 0.0 ile 1.0 arası. 0.1 çok yumuşak/yavaş, 0.9 çok sert/gürültülü.
        self.alpha_q = 0.95  # Pozisyon filtresi
        self.alpha_dq = 0.95 # Hız filtresi (Gürültünün ana kaynağı)
        
        self.q_filt = np.zeros(4)
        self.dq_filt = np.zeros(4)
        self.first_update = True

        # 3. Yörünge Parametreleri
        self.traj_params = {
            0: [1.57/5, 0.25, 1.57],  
            1: [1.57/5, 0.25,1.57], 
            2: [1.57/5, 0.25, 1.57], 
            3: [1.57/5, 0.25, 1.57]  
        }
        
        # 4. CSV ve ROS Hazırlığı
        self.csv_filename = 'robot_data_dump.csv'
        self.init_csv()
        self.pubs = [self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10) for i in range(4)]
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.start_time = self.get_clock().now()
        self.last_time = None 
        self.get_logger().info("--- Filtreli FLNN Kontrolcü Başlatıldı ---")

    def init_csv(self):
        header = ['time']
        for i in range(1, 5):
            header.extend([f'q{i}_ref', f'q{i}_act', f'dq{i}_ref', f'dq{i}_act', f'tau{i}'])
        for h in range(1, self.H + 1):
            for k in range(1, 5):
                header.append(f'W_{h}_{k}')
        with open(self.csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def get_phi(self, q, dq, dqr, ddqr, t):
        phi = np.zeros(self.H)
        # Girişlerde ham veri yerine filtreli veriler kullanılıyor
        phi[0] = np.sum(np.sin(q))
        phi[1] = np.sum(np.cos(q))
        phi[2] = np.tanh(np.sum(dq))
        phi[3] = np.tanh(np.sum(ddqr))
        phi[4] = np.tanh(np.sum(dq))
        phi[5] = np.sin(q[1] + q[2])
        phi[6] = np.cos(q[1] + q[2])
        phi[7] = np.sum(q)
        phi[8] = np.tanh(np.sum(dqr))
        phi[9] = np.tanh(np.sum(np.square(q)))
        phi[10] = 1.0  
        phi[11] = np.tanh(t % 10.0) 
        return phi.reshape(-1, 1)

    def cb(self, msg):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        
        # Güvenlik Kontrolü: dt limitini 0.05 (20Hz) yaptık ki kontrolcü donmasın
        if dt <= 0 or dt > 0.01:
            self.last_time = now
            return

        self.last_time = now

        try:
            names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in names]
            q_raw = np.array([msg.position[i] for i in idx])
            dq_raw = np.array([msg.velocity[i] for i in idx])
        except (ValueError, IndexError):
            return

        # --- ALÇAK GEÇİREN FİLTRE UYGULAMASI ---
        if self.first_update:
            self.q_filt = q_raw
            self.dq_filt = dq_raw
            self.first_update = False
        else:
            self.q_filt = self.alpha_q * self.q_filt + (1 - self.alpha_q) * q_raw
            self.dq_filt = self.alpha_dq * self.dq_filt + (1 - self.alpha_dq) * dq_raw

        # Referans Üretici
        qd, dqd, ddqd = np.zeros(4), np.zeros(4), np.zeros(4)
        for i in range(4):
            A, f, b = self.traj_params[i]
            qd[i] = A * np.sin(2 * np.pi * f * t) + b
            dqd[i] = A * (2 * np.pi * f) * np.cos(2 * np.pi * f * t)
            ddqd[i] = -A * (2 * np.pi * f)**2 * np.sin(2 * np.pi * f * t)

        # Hata hesaplamada filtreli verileri kullanıyoruz
        e = qd - self.q_filt
        de = dqd - self.dq_filt
        
        s = de + self.lam @ e  
        dqr = dqd + self.lam @ e
        ddqr = ddqd + self.lam @ de

        # FLNN Hesaplamaları
        phi = self.get_phi(self.q_filt, self.dq_filt, dqr, ddqr, t)
        tau_nn = (self.W_weights.T @ phi).flatten()
        
        # Adaptasyon (Ağırlık güncelleme)
        w_dot = self.F * (phi @ s.reshape(1, -1))
        # "Leaky Integrator" ekleyerek ağırlıkların şişmesini engelliyoruz
        self.W_weights += (w_dot - self.W_weights) * dt
        self.W_weights = np.clip(self.W_weights, -1500000.0, 1500000.0) # Sınırı daralttık
        
        # Toplam Kontrol Torku
        tau = tau_nn + (self.Kv @ s)

        # Veri Kaydı ve Yayınlama
        self.log_to_csv(t, qd, self.q_filt, dqd, self.dq_filt, tau)

        for i in range(4):
            val = Float64()
            # Joint 4 için tork limitini daha sıkı tutabilirsin (ör: -2, 2)
            limit = 1000000000.0 if i < 3 else 3000000000.0
            val.data = float(np.clip(tau[i], -limit, limit)) 
            self.pubs[i].publish(val)

    def log_to_csv(self, t, qd, q_act, dqd, dq_act, tau):
        try:
            row = [t]
            for i in range(4):
                row.extend([qd[i], q_act[i], dqd[i], dq_act[i], tau[i]])
            row.extend(self.W_weights.flatten().tolist())
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except:
            pass

def main():
    rclpy.init()
    node = FLNN_DataLoggerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKontrolcü durduruldu.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
