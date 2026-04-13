import carla
import csv
import math
import os
import random
from Initial import setup_environment
from Controller import VehicleController
from ekf import EKF4State, run_offline_ekf  
from evaluation import run_evaluation 

# --- Global Sensor Buffer ---
# Added 'compass' key to store the heading
sensor_data = {'accel_x': 0.0, 'gyro_z': 0.0, 'compass': 0.0, 'gnss_x': None, 'gnss_y': None}

def setup_run_directory():
    os.makedirs('results', exist_ok=True)
    existing_runs = [d for d in os.listdir('results') if d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    next_run = max(run_nums) + 1 if run_nums else 1
    run_dir = f"results/run_{next_run}"
    os.makedirs(run_dir)
    return run_dir

def init_csv(filename, header):
    with open(filename, mode='w', newline='') as file:
        csv.writer(file).writerow(header)

def save_imu(data, run_dir):
    # 1. Coordinate Transforms (Left to Right-Handed)
    # Step 1: Flip for LH→RH
    rh_compass_raw = -data.compass
    # Step 2: Add NED→ENU correction (π/2 rotation)
    rh_compass = rh_compass_raw + (math.pi / 2.0)
    # Step 3: Wrap to [-π, π] to avoid sending out-of-range values to EKF
    rh_compass = math.atan2(math.sin(rh_compass), math.cos(rh_compass))
    
    rh_gyro_z = -data.gyroscope.z 

    # 2. PHYSICAL CLAMPING (Kill the Unreal Engine Noise)
    # A normal car max braking is ~ -10 m/s^2 (1 G). Max accel is ~ 8 m/s^2.
    clamped_accel_x = max(-10.0, min(8.0, data.accelerometer.x))
    
    # A normal car rarely spins faster than 1.0 rad/s (~57 deg/s) unless drifting.
    clamped_gyro_z = max(-1.0, min(1.0, rh_gyro_z))

    # Save clean, realistic data to the global buffer
    sensor_data['accel_x'] = clamped_accel_x
    sensor_data['gyro_z'] = clamped_gyro_z
    sensor_data['compass'] = rh_compass
    
    with open(f'{run_dir}/imu_data.csv', mode='a', newline='') as file:
        # Write the clamped data to the CSV so the offline tuner gets clean data too
        csv.writer(file).writerow([
            data.timestamp, 
            clamped_accel_x, 
            data.accelerometer.y, 
            data.accelerometer.z, 
            clamped_gyro_z, 
            rh_compass
        ])
def save_gnss(data, run_dir):
    # v4.1+ FIX: Transform at collection time (LH->RH: invert Y)
    transform = data.transform
    sensor_data['gnss_x'] = transform.location.x
    sensor_data['gnss_y'] = -transform.location.y  # Transform: invert Y for RH frame
    with open(f'{run_dir}/gnss_data.csv', mode='a', newline='') as file:
        csv.writer(file).writerow([data.timestamp, transform.location.x, sensor_data['gnss_y'], data.altitude])

def main():
    run_dir = setup_run_directory()
    print(f"Setting up environment... Data will be saved to {run_dir}")
    
    actors = setup_environment()
    world = actors['world']
    ego_vehicle = actors['ego_vehicle']
    spectator = actors['spectator']

    # Added 'Compass' to the IMU CSV Header
    init_csv(f'{run_dir}/imu_data.csv', ['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_Z', 'Compass'])
    init_csv(f'{run_dir}/gnss_data.csv', ['Timestamp', 'Loc_X', 'Loc_Y', 'Altitude'])
    init_csv(f'{run_dir}/odom_data.csv', ['Timestamp', 'Loc_X', 'Loc_Y', 'Yaw_Degrees', 'GT_Velocity', 'Odom_Velocity'])

    actors['imu_sensor'].listen(lambda data: save_imu(data, run_dir))
    actors['gnss_sensor'].listen(lambda data: save_gnss(data, run_dir))

    controller = VehicleController(ego_vehicle, world, mode='auto')
    ekf_online = EKF4State()

    print("Simulation running. Loop Closed (Magnetometer Active!). Press CTRL+C to stop.")
    
    compass_offset = None
    try:
        last_timestamp = 0.0
        
        while True:
            snapshot = world.wait_for_tick()
            timestamp = snapshot.timestamp.elapsed_seconds 
            dt = timestamp - last_timestamp if last_timestamp > 0.0 else 0.05
            last_timestamp = timestamp
            
            # Ground Truth & Noisy Odometry
            gt_transform = ego_vehicle.get_transform()
            velocity_vec = ego_vehicle.get_velocity()
            gt_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            odom_speed = gt_speed * 1.02 + random.gauss(0.0, 0.1)
            
            # --- 1. CONVERT GT TO RIGHT-HANDED FOR EKF MATH ---
            rh_gt_y = -gt_transform.location.y
            rh_gt_yaw = -math.radians(gt_transform.rotation.yaw)

            # --- ONLINE EKF ---
            # Wait 1.5s for CARLA physics to settle before capturing compass offset
            # (spawn drop causes -6M G acceleration spike that poisons calibration)
            if not ekf_online._initialized and sensor_data['gnss_x'] is not None and timestamp > 5:
                # Physics has settled. Capture clean, accurate offset.
                compass_offset = sensor_data['compass'] - rh_gt_yaw
                ekf_online.initialize_state(sensor_data['gnss_x'], sensor_data['gnss_y'], odom_speed, rh_gt_yaw)
                print(f"[INFO] EKF Initialized cleanly at t={timestamp:.1f}s with offset: {compass_offset:.4f}")
            
            elif ekf_online._initialized:
                ekf_online.predict(dt, sensor_data['accel_x'], sensor_data['gyro_z'])
                
                if compass_offset is not None:
                    aligned_compass = sensor_data['compass'] - compass_offset
                    ekf_online.update_compass(aligned_compass)

                if sensor_data['gnss_x'] is not None:
                    ekf_online.update_gnss(sensor_data['gnss_x'], sensor_data['gnss_y'])
                    sensor_data['gnss_x'] = None 
                
                ekf_online.update_odom(odom_speed)
            
            # --- 2. CONTROL PIPELINE: TRANSLATE EKF BACK TO LEFT-HANDED FOR CARLA CONTROLLER ---
            if ekf_online._initialized:
                st = ekf_online.x.flatten()
                # Invert RH coordinates back to LH: est_y = -st[1], est_yaw = -st[3]
                est_x, est_y, est_v, est_yaw = st[0], -st[1], st[2], -st[3]
            else:
                # If EKF isn't ready, pass raw CARLA left-handed GT directly
                est_x, est_y, est_v, est_yaw = gt_transform.location.x, gt_transform.location.y, gt_speed, math.radians(gt_transform.rotation.yaw)

            keep_running = controller.process_control(dt, est_x, est_y, est_v, est_yaw)
            if not keep_running: break
            
            forward = gt_transform.get_forward_vector()
            spectator.set_transform(carla.Transform(
                gt_transform.location - carla.Location(x=forward.x * 6, y=forward.y * 6, z=0) + carla.Location(z=3),
                carla.Rotation(pitch=-15, yaw=gt_transform.rotation.yaw, roll=0)
            ))
            
            # v4.1+ FIX: Transform GT at save time (LH->RH: invert Y and yaw) for consistency
            gt_y_transformed = -gt_transform.location.y
            gt_yaw_transformed = -gt_transform.rotation.yaw
            with open(f'{run_dir}/odom_data.csv', mode='a', newline='') as file:
                csv.writer(file).writerow([timestamp, gt_transform.location.x, gt_y_transformed, gt_yaw_transformed, gt_speed, odom_speed])

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        print("Cleaning up CARLA actors...")
        actors['imu_sensor'].stop()
        actors['gnss_sensor'].stop()
        actors['imu_sensor'].destroy()
        actors['gnss_sensor'].destroy()
        ego_vehicle.destroy()
        
        run_offline_ekf(run_dir)
        run_evaluation(run_dir)

if __name__ == '__main__':
    main()