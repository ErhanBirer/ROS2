#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import csv

class MLP_DataLoggerController(Node):
    def __init__(self):
        super().__init__("mlp_data_logger_controller")
        
        # 1. Kontrol Kazançları
        self.lam = np.diag([5.0, 5.0, 5.0, 5.0])  
        self.Kv = np.diag([0.2, 0.3, 1.1, 0.2])
        
        # 2. MLP Parametreleri
        self.H = 10  
        self.K = 4   
        self.I = 20  
        self.F = 0.1 
        self.G = 0.1 
        
        # Ağırlıklar
        self.W_weights = np.random.randn(self.H + 1, self.K) * 0.1
        self.V_weights = np.random.randn(self.I, self.H) * 0.1 
        
        # 3. Filtreleme
        self.alpha_q = 0.95
        self.alpha_dq = 0.95
        self.q_filt = np.zeros(4)
        self.dq_filt = np.zeros(4)
        self.first_update = True

        self.traj_params = {i: [1.57/5, 0.25, 1.57] for i in range(4)}
        self.csv_filename = 'robot_mlp_weights_data.csv'
        self.init_csv()
        
        self.pubs = [self.create_publisher(Float64, f'/model/four_dof_arm/joint/joint{i+1}/cmd_force', 10) for i in range(4)]
        self.create_subscription(JointState, '/model/four_dof_arm/joint_state', self.cb, 10)
        
        self.start_time = self.get_clock().now()
        self.last_time = None 
        self.get_logger().info("--- MLP Kontrolcü (Ağırlık Kaydı Dahil) Başlatıldı ---")

    def init_csv(self):
        header = ['time']
        # Eklemler için başlıklar
        for i in range(1, 5):
            header.extend([f'q{i}_ref', f'q{i}_act', f'tau{i}'])
        
        # W ağırlıkları için başlıklar (W_0_0, W_0_1...)
        for h in range(self.H + 1):
            for k in range(self.K):
                header.append(f'W_{h}_{k}')
        
        # V ağırlıkları için başlıklar (V_0_0, V_0_1...)
        for i in range(self.I):
            for h in range(self.H):
                header.append(f'V_{i}_{h}')
        
        with open(self.csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1.0 / (1.0 + np.exp(-x))

    def cb(self, msg):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        t = (now - self.start_time).nanoseconds / 1e9
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0 or dt > 0.05: 
            self.last_time = now
            return
        self.last_time = now

        try:
            names = ['joint1', 'joint2', 'joint3', 'joint4']
            idx = [msg.name.index(n) for n in names]
            q_raw = np.array([msg.position[i] for i in idx])
            dq_raw = np.array([msg.velocity[i] for i in idx])
        except: return

        if self.first_update:
            self.q_filt, self.dq_filt = q_raw, dq_raw
            self.first_update = False
        else:
            self.q_filt = self.alpha_q * self.q_filt + (1 - self.alpha_q) * q_raw
            self.dq_filt = self.alpha_dq * self.dq_filt + (1 - self.alpha_dq) * dq_raw

        qd, dqd = np.zeros(4), np.zeros(4)
        for i in range(4):
            A, f, b = self.traj_params[i]
            qd[i] = A * np.sin(2 * np.pi * f * t) + b
            dqd[i] = A * (2 * np.pi * f) * np.cos(2 * np.pi * f * t)

        e = qd - self.q_filt
        de = dqd - self.dq_filt
        r = de + self.lam @ e 

        # MLP Forward
        x_new = np.concatenate([self.q_filt, qd, dqd, e, de]).reshape(1, -1) 
        in_out = x_new @ self.V_weights 
        sigma_x = self.sigmoid(in_out)  
        sigma_x_new = np.insert(sigma_x, 0, 1.0).reshape(-1, 1) 
        tau_nn = (self.W_weights.T @ sigma_x_new).flatten()

        # MLP Update
        w_dot = self.F * (sigma_x_new @ r.reshape(1, -1))
        sigma_x_dot = sigma_x * (1.0 - sigma_x) 
        inner_grad = (sigma_x_dot * (r @ self.W_weights[1:, :].T)) 
        v_dot = self.G * (x_new.T @ inner_grad) 

        self.W_weights += w_dot * dt
        self.V_weights += v_dot * dt

        # Control Law
        u = tau_nn + (self.Kv @ r)

        # Log and Publish
        self.log_to_csv(t, qd, self.q_filt, u)

        for i in range(4):
            val = Float64()
            val.data = float(np.clip(u[i], -200.0, 200.0)) 
            self.pubs[i].publish(val)

    def log_to_csv(self, t, qd, q_act, tau):
        row = [t]
        # Eklem verileri
        for i in range(4):
            row.extend([qd[i], q_act[i], tau[i]])
        
        # Ağırlık verileri (Flattened)
        row.extend(self.W_weights.flatten().tolist())
        row.extend(self.V_weights.flatten().tolist())
        
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except:
            pass

def main():
    rclpy.init()
    node = MLP_DataLoggerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
