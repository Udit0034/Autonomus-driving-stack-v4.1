import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import signal
import json
import os

def run_evaluation(run_dir):
    print(f"\n--- Running Full Evaluation & Plotting Suite in {run_dir} ---")
    os.makedirs(f'{run_dir}/plots', exist_ok=True)

    # 1. Load Ground Truth and EKF Modes from the specific run directory
    odom_df = pd.read_csv(f'{run_dir}/odom_data.csv').sort_values('Timestamp')
    m1_df = pd.read_csv(f'{run_dir}/ekf_results_mode1_imu_only.csv').sort_values('Timestamp')
    m2_df = pd.read_csv(f'{run_dir}/ekf_results_mode2_imu_gnss.csv').sort_values('Timestamp')
    m3_df = pd.read_csv(f'{run_dir}/ekf_results_mode3_full.csv').sort_values('Timestamp')

    # Merge GT with Mode 3 (Full Fusion) for metrics
    merged = pd.merge_asof(m3_df, odom_df, on='Timestamp', direction='nearest').dropna()

    # Calculate Position Error
    merged['error_x'] = merged['Est_X'] - merged['Loc_X']
    merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
    merged['pos_error'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2)

    # --- Calculate Jerk: Handling Spawn Drops and Discrete Math ---
    dt = merged['Timestamp'].diff()
    dt = dt.where(dt > 0.01, 0.01) # Force tiny dt values to 0.01
    
    # v4.1 FIX: Enhanced velocity smoothing to eliminate jerk spikes
    # Apply both rolling average and Savitzky-Golay filter for buttery smooth curves
    velocity_series = merged['GT_Velocity'].values
    if len(velocity_series) > 21:  # savgol requires window < len(data)
        # First pass: rolling average with centered window
        smoothed_vel = merged['GT_Velocity'].rolling(window=11, center=True, min_periods=1).mean()
        # Second pass: Savitzky-Golay filter for smooth derivatives
        try:
            smoothed_vel = pd.Series(signal.savgol_filter(smoothed_vel, window_length=21, polyorder=3), index=merged.index)
        except:
            # Fallback if savgol fails (data too short)
            smoothed_vel = merged['GT_Velocity'].rolling(window=20, center=True, min_periods=1).mean()
    else:
        # For short runs, use larger rolling window
        smoothed_vel = merged['GT_Velocity'].rolling(window=20, center=True, min_periods=1).mean()
    
    # Forward fill any remaining NaNs from windowing
    smoothed_vel = smoothed_vel.fillna(method='bfill').fillna(method='ffill')
    
    # 2. Calculate Accel and Jerk from the smoothed curve
    merged['Long_Accel'] = smoothed_vel.diff() / dt
    merged['Long_Jerk'] = merged['Long_Accel'].diff() / dt
    
    # Lateral Accel = Velocity * Yaw_Rate (using diff of Yaw)
    yaw_rate_rad = np.radians(merged['Yaw_Degrees'].diff() / dt)
    merged['Lat_Accel'] = smoothed_vel * yaw_rate_rad
    merged['Lat_Jerk'] = merged['Lat_Accel'].diff() / dt

    # 3. The "Settling Time" Fix: Ignore the first 2 seconds of the simulation
    start_time = merged['Timestamp'].min()
    valid_data = merged[merged['Timestamp'] > (start_time + 2.0)]
    
    # Drop NaNs for metric calculations using the clean, trimmed data
    abs_jerk = valid_data['Long_Jerk'].abs().dropna()

    # --- PLOT 1: Trajectory Comparison ---
    margin = 5.0  
    plt.figure(figsize=(10, 8))
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(m1_df['Est_X'], m1_df['Est_Y'], 'r-', label='IMU Only (Drift)', alpha=0.5)
    plt.plot(m2_df['Est_X'], m2_df['Est_Y'], 'g-', label='IMU + GNSS', alpha=0.7)
    plt.plot(m3_df['Est_X'], m3_df['Est_Y'], 'b-', label='Full Fusion', linewidth=2)
    plt.xlim(min(odom_df['Loc_X']) - margin, max(odom_df['Loc_X']) + margin) # clamping xlims
    plt.ylim(min(odom_df['Loc_Y']) - margin, max(odom_df['Loc_Y']) + margin)  # clamping ylims
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Trajectory Sensor Fusion Comparison')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(f'{run_dir}/plots/trajectory_sensor_fusion_comparison.png')
    plt.close()

    # --- PLOT 2: Error Comparison (Mode 2 vs 3) ---
    m2_merged = pd.merge_asof(m2_df, odom_df, on='Timestamp', direction='nearest').dropna()
    err_m2 = np.sqrt((m2_merged['Est_X']-m2_merged['Loc_X'])**2 + (m2_merged['Est_Y']-m2_merged['Loc_Y'])**2)
    
    plt.figure(figsize=(10, 4))
    plt.plot(m2_merged['Timestamp'], err_m2, 'g', alpha=0.5, label='Mode 2 (IMU+GNSS)')
    plt.plot(merged['Timestamp'], merged['pos_error'], 'b', label='Mode 3 (Full Fusion)')
    plt.title('Localization Error Over Time')
    plt.legend()
    plt.savefig(f'{run_dir}/plots/error_comparison.png')
    plt.close()

    # --- PLOT 3: Longitudinal Jerk ---
    plt.figure(figsize=(10, 4))
    plt.plot(merged['Timestamp'], merged['Long_Jerk'], color='orange', alpha=0.8)
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Comfort Limit (±3)')
    plt.axhline(y=-3, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=10, color='purple', linestyle=':', label='Safety Limit (±10)')
    plt.axhline(y=-10, color='purple', linestyle=':')
    plt.title('Longitudinal Jerk Over Time')
    plt.ylim(-15, 15)
    plt.legend()
    plt.savefig(f'{run_dir}/plots/jerk_plot.png')
    plt.close()

    # --- PLOT 4: Jerk Heatmap (Lateral vs Longitudinal) ---
    plt.figure(figsize=(8, 6))
    plt.hist2d(merged['Lat_Jerk'].fillna(0), merged['Long_Jerk'].fillna(0), bins=50, cmap='inferno')
    plt.colorbar(label='Frequency')
    plt.title('2D Jerk Heatmap')
    plt.xlabel('Lateral Jerk (m/s³)')
    plt.ylabel('Longitudinal Jerk (m/s³)')
    plt.savefig(f'{run_dir}/plots/jerk_heatmap.png')
    plt.close()

    # --- PLOT 5: EKF Error Map ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged['Est_X'], merged['Est_Y'], c=merged['pos_error'], cmap='inferno', s=20)
    plt.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', alpha=0.3, label='Ground Truth')
    plt.colorbar(scatter, label='Position Error (m)')
    plt.title('EKF Error Map (Trajectory colored by Error)')
    plt.legend()
    plt.savefig(f'{run_dir}/plots/ekf_error_map.png')
    plt.close()

    # --- PLOT 6: Covariance Ellipses ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(merged['Est_X'], merged['Est_Y'], 'b-', label='Estimated Path')
    
    # Subsample to avoid freezing (plot 1 ellipse every 100 frames)
    for idx, row in merged.iloc[::100].iterrows():
        cov = np.array([[row['P_xx'], row['P_xy']], [row['P_xy'], row['P_yy']]])
        vals, vecs = np.linalg.eigh(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(vals) # 2 Sigma
        ellip = Ellipse(xy=(row['Est_X'], row['Est_Y']), width=width, height=height, angle=theta, color='green', alpha=0.3)
        ax.add_artist(ellip)
        
    plt.title('EKF Covariance Ellipses (2σ)')
    plt.axis('equal')
    plt.savefig(f'{run_dir}/plots/covariance_ellipses.png')
    plt.close()

    # --- PLOT 7: Target Speed vs Ego Vehicle Speed ---
    plt.figure(figsize=(12, 5))
    plt.plot(merged['Timestamp'], merged['Est_Velocity'], 'b-', linewidth=2, label='EKF Estimated Velocity', alpha=0.8)
    plt.plot(merged['Timestamp'], merged['GT_Velocity'], 'k--', linewidth=1.5, label='Ground Truth Velocity', alpha=0.7)
    plt.fill_between(merged['Timestamp'], merged['Est_Velocity'], merged['GT_Velocity'], alpha=0.2, color='gray', label='Velocity Error')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('EKF Speed Tracking vs Ground Truth')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{run_dir}/plots/speed_tracking.png')
    plt.close()

    # --- Expanded Metrics Export ---
    duration = merged['Timestamp'].max() - merged['Timestamp'].min()
    
    metrics = {
        "duration_s": float(duration),
        "samples": len(merged),
        "rmse_pos": float(np.sqrt(np.mean(merged['pos_error']**2))),
        "max_error": float(merged['pos_error'].max()),
        "mean_error": float(merged['pos_error'].mean()),
        "avg_speed": float(merged['GT_Velocity'].mean()),
        "max_speed": float(merged['GT_Velocity'].max()),
        "avg_jerk": float(abs_jerk.mean()) if not abs_jerk.empty else 0.0,
        "max_jerk": float(abs_jerk.max()) if not abs_jerk.empty else 0.0,
        "rms_jerk": float(np.sqrt(np.mean(abs_jerk**2))) if not abs_jerk.empty else 0.0,
        "p95_jerk": float(np.percentile(abs_jerk, 95)) if not abs_jerk.empty else 0.0
    }
    
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Generated 7 plots in {run_dir}/plots/")
    print(f"✅ Evaluated {int(metrics['samples'])} samples over {metrics['duration_s']:.1f}s")
    print(f"✅ Final RMSE: {metrics['rmse_pos']:.3f}m | Max Jerk: {metrics['max_jerk']:.3f} m/s³")

if __name__ == '__main__':
    # For testing standalone, default to run_1 if no argument passed
    run_evaluation("results/run_1")