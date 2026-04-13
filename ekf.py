import numpy as np
import pandas as pd

class EKF4State:
    """4-State Extended Kalman Filter: [x, y, v, yaw]"""
    IX, IY, IV, IYAW = 0, 1, 2, 3
    STATE_DIM = 4

    def __init__(self):
        self.x = np.zeros((self.STATE_DIM, 1))
        self.P = np.eye(self.STATE_DIM) * 1.0 
        # Q: Process Noise - Smooth out the X/Y jumps
        self.Q = np.diag([0.5, 0.5, 0.1, 0.01]) 

        # R: Measurement Noise
        # 1. Back off the GNSS trust slightly to stop the teleporting
        self.R_gnss = np.diag([1.0, 1.0]) 
        
        # 2. Trust the Odometry heavily so the car drives smoothly between GPS blips
        self.R_odom = np.array([[0.05]])   
        
        # 3. Keep compass stable
        self.R_compass = np.array([[2.0]])
        
        self._initialized = False

    def initialize_state(self, x, y, v, yaw):
        self.x = np.array([[x], [y], [v], [yaw]])
        self._initialized = True

    def predict(self, dt: float, accel: float, yaw_rate: float):
        if not self._initialized or dt <= 0.0: return
        x, y, v, yaw = self.x.flatten()

        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        v_new = v + accel * dt
        yaw_new = yaw + yaw_rate * dt

        self.x = np.array([[x_new], [y_new], [v_new], [yaw_new]])

        F = np.eye(self.STATE_DIM)
        F[self.IX, self.IV] = np.cos(yaw) * dt
        F[self.IX, self.IYAW] = -v * np.sin(yaw) * dt
        F[self.IY, self.IV] = np.sin(yaw) * dt
        F[self.IY, self.IYAW] = v * np.cos(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, is_angle: bool = False):
        y = z - H @ self.x  
        
        # ANGLE WRAPPING: If this is a compass update, clamp error to [-pi, pi]
        if is_angle:
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

        S = H @ self.P @ H.T + R  
        K = self.P @ H.T @ np.linalg.inv(S)  
        self.x = self.x + K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gnss(self, meas_x: float, meas_y: float):
        if not self._initialized: return
        z = np.array([[meas_x], [meas_y]])
        H = np.zeros((2, self.STATE_DIM))
        H[0, self.IX] = 1.0
        H[1, self.IY] = 1.0
        self._update(z, H, self.R_gnss)

    def update_odom(self, meas_v: float):
        if not self._initialized: return
        z = np.array([[meas_v]])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.IV] = 1.0
        self._update(z, H, self.R_odom)

    def update_compass(self, meas_yaw: float):
        if not self._initialized: return
        z = np.array([[meas_yaw]])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.IYAW] = 1.0
        # Pass is_angle=True to trigger the wrapping math
        self._update(z, H, self.R_compass, is_angle=True)

def run_offline_ekf(run_dir):
    print(f"\n--- Starting Offline EKF Fusion in {run_dir} ---")
    imu_df = pd.read_csv(f'{run_dir}/imu_data.csv')
    gnss_df = pd.read_csv(f'{run_dir}/gnss_data.csv')
    odom_df = pd.read_csv(f'{run_dir}/odom_data.csv')
    
    timeline = pd.concat([
        imu_df.assign(type='imu'),
        gnss_df.assign(type='gnss'),
        odom_df.assign(type='odom')
    ]).sort_values(by='Timestamp').reset_index(drop=True)

    modes = {
        'mode1_imu_only': {'use_gnss': False, 'use_odom': False, 'use_compass': True},
        'mode2_imu_gnss': {'use_gnss': True, 'use_odom': False, 'use_compass': True},
        'mode3_full': {'use_gnss': True, 'use_odom': True, 'use_compass': True}
    }

    for mode_name, config in modes.items():
        ekf = EKF4State()
        results = []
        last_time = timeline.iloc[0]['Timestamp']
        compass_offset = None

        for _, row in timeline.iterrows():
            current_time = row['Timestamp']
            dt = current_time - last_time
            last_time = current_time
            sensor = row['type']

            # v4.1+ FIX: Initialize from GNSS (not GT odom) with compass calibration
            if not ekf._initialized:
                if sensor == 'gnss':
                    # Initialize position from first available GNSS
                    # Estimate initial yaw from first compass reading
                    first_compass = imu_df.iloc[0]['Compass'] if len(imu_df) > 0 else 0.0
                    first_odom_vel = odom_df.iloc[0]['Odom_Velocity'] if len(odom_df) > 0 else 0.0
                    ekf.initialize_state(row['Loc_X'], row['Loc_Y'], first_odom_vel, first_compass)
                    # Calculate compass offset once at init (transform already applied at collection)
                    compass_offset = first_compass - first_compass  # offset is pre-aligned
                    print(f"  [{mode_name}] Initialized from GNSS at t={current_time:.3f}: pos=({row['Loc_X']:.2f}, {row['Loc_Y']:.2f}), yaw={np.degrees(first_compass):.1f}°")
                continue

            if sensor == 'odom':
                if config['use_odom']:
                    # Use only velocity (not position) to avoid GT data leak
                    ekf.update_odom(row['Odom_Velocity'])

            elif sensor == 'imu' and ekf._initialized:
                ekf.predict(dt, accel=row['Accel_X'], yaw_rate=row['Gyro_Z'])
                # Apply compass update immediately after IMU prediction
                if config['use_compass']:
                    # Data already transformed at collection; apply pre-computed offset
                    if compass_offset is not None:
                        aligned_compass = row['Compass'] - compass_offset
                    else:
                        aligned_compass = row['Compass']
                    ekf.update_compass(aligned_compass)

            elif sensor == 'gnss' and ekf._initialized and config['use_gnss']:
                ekf.update_gnss(row['Loc_X'], row['Loc_Y'])

            if ekf._initialized:
                st = ekf.x.flatten()
                results.append([current_time, st[0], st[1], st[2], np.degrees(st[3]), ekf.P[0,0], ekf.P[1,1], ekf.P[0,1]])

        df = pd.DataFrame(results, columns=['Timestamp', 'Est_X', 'Est_Y', 'Est_Velocity', 'Est_Yaw', 'P_xx', 'P_yy', 'P_xy'])
        df.to_csv(f'{run_dir}/ekf_results_{mode_name}.csv', index=False)
        print(f"  [{mode_name}] Saved {len(df)} samples to ekf_results_{mode_name}.csv")