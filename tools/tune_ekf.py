"""
EKF Hyperparameter Tuning via Grid Search (Sequential - Optimized)

This script performs a grid search over Q and R matrix values to find optimal
hyperparameters that minimize Position RMSE. Uses pre-built timeline and numpy
arrays for speed.

Usage:
  python tune_ekf.py --run_dir results/run_2 --is_legacy
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import product
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ekf import EKF4State


def load_and_transform_data(run_dir, is_legacy_data=False):
    """
    Load CSV data from run_dir. If is_legacy_data=True, apply LH->RH transforms.
    """
    imu_df = pd.read_csv(os.path.join(run_dir, 'imu_data.csv'))
    gnss_df = pd.read_csv(os.path.join(run_dir, 'gnss_data.csv'))
    odom_df = pd.read_csv(os.path.join(run_dir, 'odom_data.csv'))
    
    if is_legacy_data:
        print(f"[INFO] Applying legacy LH->RH transforms to data from {run_dir}")
        # imu_df['Gyro_Z'] *= -1  # REMOVED: Gyro_Z should NOT be inverted
        imu_df['Compass'] *= -1
        gnss_df['Loc_Y'] *= -1
        odom_df['Loc_Y'] *= -1
        odom_df['Yaw_Degrees'] *= -1
    
    return imu_df, gnss_df, odom_df


def run_ekf_with_params(timeline, imu_df, gnss_df, odom_df, q_diag, r_gnss, r_odom, r_compass):
    """
    Run EKF with specified Q and R parameters using pre-built timeline.
    Optimized with numpy arrays instead of dataframe iteration.
    """
    ekf = EKF4State()
    ekf.Q = np.diag(q_diag)
    ekf.R_gnss = np.diag([r_gnss, r_gnss])
    ekf.R_odom = np.array([[r_odom]])
    ekf.R_compass = np.array([[r_compass]])
    
    results = []
    last_time = timeline.iloc[0]['Timestamp']
    compass_offset = None
    
    # Convert to numpy arrays for speed
    timestamps = timeline['Timestamp'].values
    sensor_types = timeline['type'].values
    
    accel_x_vals = np.full(len(timeline), np.nan)
    gyro_z_vals = np.full(len(timeline), np.nan)
    compass_vals = np.full(len(timeline), np.nan)
    gnss_x_vals = np.full(len(timeline), np.nan)
    gnss_y_vals = np.full(len(timeline), np.nan)
    odom_vel_vals = np.full(len(timeline), np.nan)
    
    imu_mask = sensor_types == 'imu'
    gnss_mask = sensor_types == 'gnss'
    odom_mask = sensor_types == 'odom'
    
    accel_x_vals[imu_mask] = timeline.loc[imu_mask, 'Accel_X'].values
    gyro_z_vals[imu_mask] = timeline.loc[imu_mask, 'Gyro_Z'].values
    compass_vals[imu_mask] = timeline.loc[imu_mask, 'Compass'].values
    
    gnss_x_vals[gnss_mask] = timeline.loc[gnss_mask, 'Loc_X'].values
    gnss_y_vals[gnss_mask] = timeline.loc[gnss_mask, 'Loc_Y'].values
    
    odom_vel_vals[odom_mask] = timeline.loc[odom_mask, 'Odom_Velocity'].values
    
    for idx in range(len(timeline)):
        current_time = timestamps[idx]
        dt = current_time - last_time
        last_time = current_time
        sensor = sensor_types[idx]
        
        if not ekf._initialized:
            if sensor == 'gnss':
                first_compass = imu_df.iloc[0]['Compass'] if len(imu_df) > 0 else 0.0
                first_odom_vel = odom_df.iloc[0]['Odom_Velocity'] if len(odom_df) > 0 else 0.0
                ekf.initialize_state(gnss_x_vals[idx], gnss_y_vals[idx], first_odom_vel, first_compass)
                compass_offset = first_compass - first_compass
            continue
        
        if sensor == 'odom':
            ekf.update_odom(odom_vel_vals[idx])
        elif sensor == 'imu' and ekf._initialized:
            ekf.predict(dt, accel=accel_x_vals[idx], yaw_rate=gyro_z_vals[idx])
            aligned_compass = compass_vals[idx] - compass_offset if compass_offset is not None else compass_vals[idx]
            ekf.update_compass(aligned_compass)
        elif sensor == 'gnss' and ekf._initialized:
            ekf.update_gnss(gnss_x_vals[idx], gnss_y_vals[idx])
        
        if ekf._initialized:
            st = ekf.x.flatten()
            results.append([
                current_time, st[0], st[1], st[2], np.degrees(st[3]),
                ekf.P[0,0], ekf.P[1,1], ekf.P[0,1]
            ])
    
    return results


def compute_rmse(est_results, odom_df):
    """Compute Position RMSE by merging EKF output with ground truth."""
    if not est_results:
        return float('inf'), None
    
    est_df = pd.DataFrame(est_results, columns=[
        'Timestamp', 'Est_X', 'Est_Y', 'Est_Velocity', 'Est_Yaw',
        'P_xx', 'P_yy', 'P_xy'
    ])
    
    merged = pd.merge_asof(
        est_df.sort_values('Timestamp'),
        odom_df.sort_values('Timestamp'),
        on='Timestamp', direction='nearest'
    ).dropna()
    
    if len(merged) == 0:
        return float('inf'), None
    
    merged['error_x'] = merged['Est_X'] - merged['Loc_X']
    merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
    pos_error = np.sqrt(merged['error_x']**2 + merged['error_y']**2)
    rmse = float(np.sqrt(np.mean(pos_error**2)))
    
    return rmse, merged


def plot_tuning_results(est_results, odom_df, output_dir, trial_name="tuning"):
    """Generate Summary Plots (3 plots)."""
    os.makedirs(output_dir, exist_ok=True)
    
    est_df = pd.DataFrame(est_results, columns=[
        'Timestamp', 'Est_X', 'Est_Y', 'Est_Velocity', 'Est_Yaw',
        'P_xx', 'P_yy', 'P_xy'
    ])
    
    merged = pd.merge_asof(
        est_df.sort_values('Timestamp'),
        odom_df.sort_values('Timestamp'),
        on='Timestamp', direction='nearest'
    ).dropna()
    
    # PLOT 1: Trajectory Comparison
    margin = 5.0
    plt.figure(figsize=(10, 8))
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(est_df['Est_X'], est_df['Est_Y'], 'b-', label='EKF Estimate (Full Fusion)', linewidth=2)
    plt.xlim(min(odom_df['Loc_X']) - margin, max(odom_df['Loc_X']) + margin)
    plt.ylim(min(odom_df['Loc_Y']) - margin, max(odom_df['Loc_Y']) + margin)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Trajectory Comparison [{trial_name}]')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, 'trajectory_sensor_fusion_comparison.png'), dpi=100)
    plt.close()
    
    # PLOT 2: EKF Error Map
    merged['error_x'] = merged['Est_X'] - merged['Loc_X']
    merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
    merged['pos_error'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged['Est_X'], merged['Est_Y'], c=merged['pos_error'],
                         cmap='inferno', s=20)
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', alpha=0.3, label='Ground Truth')
    plt.colorbar(scatter, label='Position Error (m)')
    plt.title(f'EKF Error Map [{trial_name}]')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ekf_error_map.png'), dpi=100)
    plt.close()
    
    # PLOT 3: Covariance Ellipses
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(est_df['Est_X'], est_df['Est_Y'], 'b-', label='Estimated Path', linewidth=1.5)
    
    for idx, row in merged.iloc[::100].iterrows():
        if pd.isna(row['P_xx']) or pd.isna(row['P_yy']):
            continue
        cov = np.array([[row['P_xx'], row['P_xy']], [row['P_xy'], row['P_yy']]])
        try:
            vals, vecs = np.linalg.eigh(cov)
            if np.all(vals > 0):
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * 2 * np.sqrt(vals)
                ellip = Ellipse(xy=(row['Est_X'], row['Est_Y']),
                              width=width, height=height, angle=theta,
                              color='green', alpha=0.3)
                ax.add_artist(ellip)
        except:
            pass
    
    ax.set_title(f'EKF Covariance Ellipses [2σ] [{trial_name}]')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'covariance_ellipses.png'), dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EKF Hyperparameter Grid Search')
    parser.add_argument('--run_dir', type=str, default='run_2',
                       help='Directory containing run data')
    parser.add_argument('--is_legacy', action='store_true',
                       help='Apply legacy LH->RH coordinate transforms')
    args = parser.parse_args()
    
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(os.getcwd(), args.run_dir)
    
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"EKF HYPERPARAMETER TUNING via Grid Search".center(70))
    print(f"{'='*70}")
    print(f"[INFO] Run Directory: {run_dir}")
    print(f"[INFO] Legacy Data Mode: {args.is_legacy}\n")
    
    # Load data
    imu_df, gnss_df, odom_df = load_and_transform_data(run_dir, is_legacy_data=args.is_legacy)
    print(f"[INFO] Loaded: {len(imu_df)} IMU, {len(gnss_df)} GNSS, {len(odom_df)} Odom rows\n")
    
    # Build timeline once
    print("[INFO] Pre-building timeline (one-time operation)...")
    timeline = pd.concat([
        imu_df.assign(type='imu'),
        gnss_df.assign(type='gnss'),
        odom_df.assign(type='odom')
    ]).sort_values(by='Timestamp').reset_index(drop=True)
    print(f"[INFO] Timeline has {len(timeline)} total events\n")
    
    # Define hyperparameter grid
    param_space = {
        'R_gnss': [0.5, 1.0, 2.0],
        'R_odom': [0.1, 0.5, 1.0],
        'R_compass': [0.5, 1.0, 2.0],
        'Q_pos': [0.1, 0.5],
        'Q_vel': [0.05, 0.1],
        'Q_yaw': [0.01, 0.05]
    }
    
    param_combinations = list(product(
        param_space['R_gnss'], param_space['R_odom'], param_space['R_compass'],
        param_space['Q_pos'], param_space['Q_vel'], param_space['Q_yaw']
    ))
    
    total_trials = len(param_combinations)
    print(f"[INFO] Grid Space Size: {total_trials} combinations\n")
    print(f"{'Trial':<8} {'RMSE (m)':<12} {'R_gnss':<8} {'R_odom':<8} {'R_compass':<10} {'Q_pos':<8} {'Q_vel':<8} {'Q_yaw':<8}")
    print(f"{'-'*80}")
    
    best_rmse = float('inf')
    best_params = None
    
    # Sequential grid search (optimized inner loop with numpy)
    for trial_idx, (r_gnss, r_odom, r_compass, q_pos, q_vel, q_yaw) in enumerate(
        tqdm(param_combinations, total=total_trials, desc="Grid Search", unit="trial")
    ):
        q_diag = [q_pos, q_pos, q_vel, q_yaw]
        
        try:
            results = run_ekf_with_params(
                timeline, imu_df, gnss_df, odom_df,
                q_diag, r_gnss, r_odom, r_compass
            )
            rmse, _ = compute_rmse(results, odom_df)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    'r_gnss': r_gnss, 'r_odom': r_odom, 'r_compass': r_compass,
                    'q_pos': q_pos, 'q_vel': q_vel, 'q_yaw': q_yaw,
                    'rmse': rmse
                }
            
            # Print EVERY trial for visibility
            print(f"{trial_idx+1:<8} {rmse:<12.4f} {r_gnss:<8.2f} {r_odom:<8.2f} {r_compass:<10.2f} {q_pos:<8.2f} {q_vel:<8.2f} {q_yaw:<8.3f}")
        
        except Exception as e:
            print(f"{trial_idx+1:<8} {'FAILED':<12} {r_gnss:<8.2f} {r_odom:<8.2f} {r_compass:<10.2f} {q_pos:<8.2f} {q_vel:<8.2f} {q_yaw:<8.3f} | Error: {str(e)[:50]}")
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETED".center(80))
    print(f"{'='*80}\n")
    
    if best_params is None:
        print("[ERROR] No valid parameters found.")
        return
    
    # Re-run best parameters to get results for plotting
    print(f"[INFO] Re-running best parameters to generate plots...")
    q_diag_best = [best_params['q_pos'], best_params['q_pos'], 
                   best_params['q_vel'], best_params['q_yaw']]
    best_results = run_ekf_with_params(
        timeline, imu_df, gnss_df, odom_df, q_diag_best,
        best_params['r_gnss'], best_params['r_odom'], best_params['r_compass']
    )
    
    # Print best parameters
    print(f"\nBEST HYPERPARAMETERS FOUND:")
    print(f"{'='*80}")
    print(f"  Position RMSE: {best_params['rmse']:.4f} m")
    print(f"  R_gnss:        {best_params['r_gnss']:.2f}")
    print(f"  R_odom:        {best_params['r_odom']:.2f}")
    print(f"  R_compass:     {best_params['r_compass']:.2f}")
    print(f"  Q_pos (X/Y):   {best_params['q_pos']:.2f}")
    print(f"  Q_vel:         {best_params['q_vel']:.2f}")
    print(f"  Q_yaw:         {best_params['q_yaw']:.3f}")
    print(f"{'='*80}\n")
    
    # Generate plots
    tuning_plots_dir = os.path.join(run_dir, 'tuning_plots')
    print(f"[INFO] Generating tuning plots in {tuning_plots_dir}...")
    plot_tuning_results(best_results, odom_df, tuning_plots_dir)
    print(f"[INFO] Saved 3 plots:")
    print(f"  - trajectory_sensor_fusion_comparison.png")
    print(f"  - ekf_error_map.png")
    print(f"  - covariance_ellipses.png\n")
    
    print(f"[DONE] Tuning complete! Best RMSE: {best_params['rmse']:.4f} m")


if __name__ == '__main__':
    main()
