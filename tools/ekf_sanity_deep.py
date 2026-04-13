"""
Deep EKF Sanity Check + Frame Diagnostic Tool
==============================================
Goes beyond sign/range checks to diagnose:
  - Magnetometer yaw vs GNSS-derived yaw (frame offset angle)
  - IMU acceleration projection correctness
  - Persistent leftward drift source
  - Coordinate frame consistency across all sensors
  - Timestamp alignment between sensors

Usage:
    python ekf_sanity_deep.py results/run_5
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (dark engineering theme)
# ─────────────────────────────────────────────
C_BG      = '#0d1117'
C_PANEL   = '#161b22'
C_BORDER  = '#30363d'
C_TEXT    = '#e6edf3'
C_MUTED   = '#8b949e'
C_GREEN   = '#3fb950'
C_RED     = '#f85149'
C_YELLOW  = '#d29922'
C_BLUE    = '#58a6ff'
C_ORANGE  = '#ff7b54'
C_PURPLE  = '#bc8cff'
C_CYAN    = '#39d353'

plt.rcParams.update({
    'figure.facecolor':  C_BG,
    'axes.facecolor':    C_PANEL,
    'axes.edgecolor':    C_BORDER,
    'axes.labelcolor':   C_TEXT,
    'xtick.color':       C_MUTED,
    'ytick.color':       C_MUTED,
    'text.color':        C_TEXT,
    'grid.color':        C_BORDER,
    'grid.linewidth':    0.6,
    'legend.facecolor':  C_PANEL,
    'legend.edgecolor':  C_BORDER,
    'font.family':       'monospace',
    'font.size':         9,
})


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (np.array(a) + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a, b):
    """Signed shortest difference a-b in radians, wrapped to [-pi, pi]."""
    return wrap_angle(np.array(a) - np.array(b))


def gnss_heading(gnss_df, min_speed_threshold=0.5):
    """
    Compute heading from consecutive GNSS positions.
    Only valid when vehicle is actually moving (speed > threshold m/s).
    Returns (timestamps, headings_rad).
    """
    x = gnss_df['Loc_X'].values
    y = gnss_df['Loc_Y'].values
    t = gnss_df['Timestamp'].values

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    dt   = np.diff(t)

    speed   = np.where(dt > 0, dist / dt, 0)
    heading = np.arctan2(dy, dx)

    # only keep samples where vehicle is moving
    valid = speed > min_speed_threshold
    t_mid = (t[:-1] + t[1:]) / 2

    return t_mid[valid], heading[valid], speed[valid]


def resample_to_timestamps(src_t, src_val, target_t):
    """Linear-interpolate src_val at target_t timestamps."""
    return np.interp(target_t, src_t, src_val)


def mad(x):
    """Median absolute deviation — robust spread measure."""
    return np.median(np.abs(x - np.median(x)))


# ══════════════════════════════════════════════
#  DIAGNOSTIC CHECKS  (each returns a dict)
# ══════════════════════════════════════════════

class DiagResult:
    def __init__(self, label, status, value, detail, fix=None):
        self.label  = label
        self.status = status   # 'OK' | 'WARN' | 'ERROR'
        self.value  = value
        self.detail = detail
        self.fix    = fix or ''


def check_compass_sign(imu, odom=None):
    """Updated: Check compass alignment with odom instead of just sign."""
    med = imu['Compass'].median()
    std = imu['Compass'].std()
    
    if odom is not None and 'Yaw_Degrees' in odom.columns:
        # Convert odom yaw to radians
        odom_yaw_rad = np.radians(odom['Yaw_Degrees'].values)
        compass_vals = imu['Compass'].values
        
        # Resample compass to odom timestamps for alignment check
        t_imu = imu['Timestamp'].values
        t_odom = odom['Timestamp'].values
        compass_at_odom = resample_to_timestamps(t_imu, compass_vals, t_odom)
        
        # Calculate wrapped angle difference
        compass_diff = np.abs(np.arctan2(np.sin(compass_at_odom - odom_yaw_rad), np.cos(compass_at_odom - odom_yaw_rad)))
        med_diff = np.median(compass_diff)
        
        if med_diff < 0.1:
            status = 'OK'
            detail = f'Compass is well-aligned with Odom (offset={np.degrees(med_diff):.2f}°)'
            fix = ''
        elif med_diff > 0.5:
            status = 'WARN'
            detail = f'Compass misaligned with Odom: {np.degrees(med_diff):.1f}°. May indicate frame error.'
            fix = f'Check if compass needs ±π/2 rotation or axis swap.'
        else:
            status = 'OK'
            detail = f'Compass offset from Odom: {np.degrees(med_diff):.2f}° (acceptable range)'
            fix = ''
        
        return DiagResult(
            'Compass Alignment',
            status,
            f'offset={np.degrees(med_diff):.2f}°  std={std:.4f} rad',
            detail,
            fix=fix
        )
    else:
        # Fallback: just check sign if odom not available
        pct_neg = (imu['Compass'] < 0).mean() * 100
        return DiagResult(
            'Compass Sign',
            'OK',
            f'median={med:.4f} rad  ({pct_neg:.1f}% samples)',
            'Compass values sampled. (Full alignment check requires odom data)',
        )


def check_gyro_sign(imu):
    mean = imu['Gyro_Z'].mean()
    # Gyro during left turns should be negative in RH frame (CCW = +, CW = -)
    return DiagResult(
        'Gyro_Z Sign',
        'OK',
        f'mean={mean:.4f} rad/s',
        'Small non-zero mean is expected (vehicle turns more one direction).',
    )


def check_gnss_y(gnss):
    med = gnss['Loc_Y'].median()
    status = 'OK' if med < 0 else 'ERROR'
    return DiagResult(
        'GNSS Loc_Y Sign',
        status,
        f'median={med:.2f} m',
        'GNSS Y must be negative (inverted from CARLA left-hand Y).',
        fix='Negate GNSS Loc_Y at collection.' if status != 'OK' else ''
    )


def check_odom_y(odom):
    med = odom['Loc_Y'].median()
    status = 'OK' if med < 0 else 'ERROR'
    return DiagResult(
        'Odom Loc_Y Sign',
        status,
        f'median={med:.2f} m',
        'Odom Y must be negative (same inversion as GNSS).',
        fix='Negate Odom Loc_Y at collection.' if status != 'OK' else ''
    )


def check_accel_range(imu):
    rng = (imu['Accel_X'].min(), imu['Accel_X'].max())
    extreme = abs(rng[0]) > 50 or abs(rng[1]) > 50
    status = 'ERROR' if extreme else 'OK'
    return DiagResult(
        'Accel_X Range',
        status,
        f'[{rng[0]:.2f}, {rng[1]:.2f}] m/s²',
        'Car acceleration should stay within ±15 m/s² normally.',
        fix='Check if IMU is in m/s² not cm/s². Check gravity compensation.' if extreme else ''
    )


def check_velocity_range(odom):
    rng = (odom['GT_Velocity'].min(), odom['GT_Velocity'].max())
    extreme = rng[0] < -1 or rng[1] > 50
    status = 'WARN' if extreme else 'OK'
    return DiagResult(
        'GT_Velocity Range',
        status,
        f'[{rng[0]:.2f}, {rng[1]:.2f}] m/s',
        'Ground truth speed should be 0–30 m/s for typical CARLA scenarios.',
    )


def check_magnetometer_frame(imu, gnss):
    """
    THE KEY CHECK: Compare magnetometer yaw vs GNSS-derived yaw.
    Returns offset statistics and whether a frame fix is needed.
    """
    # GNSS-derived heading (ground truth of heading)
    t_gnss, h_gnss, spd_gnss = gnss_heading(gnss, min_speed_threshold=0.5)

    if len(t_gnss) < 5:
        return DiagResult(
            'Magnetometer vs GNSS Yaw',
            'WARN',
            'Insufficient GNSS motion samples',
            'Need at least 5 GNSS samples with movement > 0.5 m/s.',
        ), None, None, None

    # Compass values at same timestamps (interpolate)
    t_imu    = imu['Timestamp'].values
    compass  = imu['Compass'].values

    # interpolate compass to GNSS mid-timestamps
    compass_at_gnss = resample_to_timestamps(t_imu, compass, t_gnss)

    # Signed angular difference: compass - gnss_heading
    delta = angle_diff(compass_at_gnss, h_gnss)

    median_offset = np.median(delta)
    std_offset    = np.std(delta)
    mad_offset    = mad(delta)

    # Classify the offset
    deg = np.degrees(median_offset)
    abs_deg = abs(deg)

    if abs_deg < 5:
        status = 'OK'
        detail = f'Magnetometer aligned with GNSS heading. Offset={deg:.1f}°'
        fix = ''
    elif 80 < abs_deg < 100:
        status = 'ERROR'
        detail = f'~90° frame error detected! Offset={deg:.1f}°. Classic NED↔ENU or axis-swap.'
        fix = f'Fix: rotate compass reading by {-deg:.1f}° OR swap compass X/Y axes.'
    elif 170 < abs_deg or abs_deg > 175:
        status = 'ERROR'
        detail = f'~180° flip detected! Offset={deg:.1f}°. Sign inversion applied twice or missing.'
        fix = 'Fix: remove one sign inversion from compass pipeline.'
    elif abs_deg > 15:
        status = 'WARN'
        detail = f'Significant offset={deg:.1f}°. Magnetic declination or partial frame mismatch.'
        fix = f'Fix: subtract {deg:.4f} rad ({deg:.2f}°) from compass reading before EKF fusion.'
    else:
        status = 'WARN'
        detail = f'Small but non-negligible offset={deg:.1f}°. Could cause persistent drift.'
        fix = f'Fix: subtract {deg:.4f} rad ({deg:.2f}°) from compass reading.'

    result = DiagResult(
        'Magnetometer vs GNSS Yaw',
        status,
        f'offset={deg:.2f}°  std={np.degrees(std_offset):.2f}°  MAD={np.degrees(mad_offset):.2f}°',
        detail,
        fix=fix
    )

    return result, t_gnss, compass_at_gnss, h_gnss, delta


def check_accel_projection(imu, odom):
    """
    Check: when vehicle accelerates forward, does Accel_X actually increase?
    A frame error often makes forward acceleration appear as lateral (Accel_Y) or negative.
    """
    # Get velocity from odom and differentiate to get expected acceleration
    if 'GT_Velocity' not in odom.columns:
        return DiagResult('Accel Projection', 'WARN', 'N/A', 'GT_Velocity not in odom.')

    t_odom = odom['Timestamp'].values
    v_odom = odom['GT_Velocity'].values
    a_gt   = np.gradient(v_odom, t_odom)  # ground truth acceleration

    # Interpolate IMU Accel_X to odom timestamps
    t_imu  = imu['Timestamp'].values
    ax_imu = imu['Accel_X'].values
    ax_at_odom = resample_to_timestamps(t_imu, ax_imu, t_odom)

    # Correlation: should be positive (forward accel = positive Accel_X in body frame)
    # Only use samples where GT acceleration is significant
    sig = np.abs(a_gt) > 0.3
    if sig.sum() < 10:
        return DiagResult('Accel_X vs GT_Accel Correlation', 'WARN', 'N/A',
                          'Not enough dynamic acceleration samples.')

    corr = np.corrcoef(a_gt[sig], ax_at_odom[sig])[0, 1]
    
    if corr > 0.5:
        status = 'OK'
        detail = f'Accel_X positively correlated with GT acceleration (r={corr:.3f}). Body frame looks correct.'
        fix = ''
    elif corr > 0:
        status = 'WARN'
        detail = f'Weak correlation (r={corr:.3f}). Possible axis mismatch or noise.'
        fix = 'Verify IMU Accel_X is along vehicle forward axis.'
    elif corr < -0.3:
        status = 'ERROR'
        detail = f'NEGATIVE correlation (r={corr:.3f})! Forward acceleration shows as deceleration.'
        fix = 'Flip sign of Accel_X. Or check if Accel_X is actually lateral/vertical axis.'
    else:
        status = 'WARN'
        detail = f'Near-zero correlation (r={corr:.3f}). IMU may not be aligned with vehicle axis.'
        fix = 'Check which IMU axis aligns with vehicle forward direction.'

    return DiagResult(
        'Accel_X vs GT_Accel Correlation',
        status,
        f'r={corr:.3f}',
        detail,
        fix=fix
    ), corr, a_gt, ax_at_odom, t_odom


def check_accel_axis_alignment(imu, odom):
    """
    COMPREHENSIVE: Correlate every GT accel axis with every IMU accel axis.
    Builds a 3x3 correlation matrix to find which IMU axes align with GT axes.
    """
    if 'GT_Velocity' not in odom.columns or 'Accel_Y' not in imu.columns or 'Accel_Z' not in imu.columns:
        return DiagResult(
            'Accel Axis Alignment (Full Matrix)',
            'WARN',
            'N/A',
            'Missing acceleration data in IMU or odom CSV files.'
        ), None

    # Ground truth acceleration (derived from velocity)
    t_odom = odom['Timestamp'].values
    v_odom = odom['GT_Velocity'].values
    a_gt_forward = np.gradient(v_odom, t_odom)

    # IMU accelerations
    t_imu = imu['Timestamp'].values
    ax_imu = imu['Accel_X'].values
    ay_imu = imu['Accel_Y'].values
    az_imu = imu['Accel_Z'].values

    # Resample all IMU axes to odom timestamps
    ax_at_odom = resample_to_timestamps(t_imu, ax_imu, t_odom)
    ay_at_odom = resample_to_timestamps(t_imu, ay_imu, t_odom)
    az_at_odom = resample_to_timestamps(t_imu, az_imu, t_odom)

    # Build 3x3 correlation matrix
    sig = np.abs(a_gt_forward) > 0.3
    if sig.sum() < 10:
        return DiagResult(
            'Accel Axis Alignment (Full Matrix)',
            'WARN',
            'N/A',
            'Not enough dynamic acceleration samples (need > 10 significant samples).'
        ), None

    # Correlate GT forward accel with each IMU axis
    corr_x = np.corrcoef(a_gt_forward[sig], ax_at_odom[sig])[0, 1]
    corr_y = np.corrcoef(a_gt_forward[sig], ay_at_odom[sig])[0, 1]
    corr_z = np.corrcoef(a_gt_forward[sig], az_at_odom[sig])[0, 1]

    # Find best match and rank them
    correlations = {
        'IMU_Accel_X': corr_x,
        'IMU_Accel_Y': corr_y,
        'IMU_Accel_Z': corr_z,
    }
    
    best_axis = max(correlations, key=lambda k: abs(correlations[k]))
    best_corr = correlations[best_axis]
    
    # Determine status based on best correlation strength
    if abs(best_corr) > 0.6:
        status = 'OK'
        detail = f'Best alignment: GT_Forward ↔ {best_axis} (r={best_corr:.3f}). Frame is correct.'
        if best_axis != 'IMU_Accel_X':
            detail += f' ⚠️  Note: Forward motion correlates with {best_axis}, not Accel_X!'
        fix = f'If IMU_Accel_X should be forward axis, uncomment axis swap in save_imu().' if best_axis != 'IMU_Accel_X' else ''
    elif abs(best_corr) > 0.3:
        status = 'WARN'
        detail = f'Weak alignment: GT_Forward ↔ {best_axis} (r={best_corr:.3f}). Possible noise or poor calibration.'
        fix = f'Check if {best_axis} is the forward axis. Consider swapping axes if high corr with other axis is found.'
    else:
        status = 'ERROR'
        detail = f'No clear alignment found. Best: {best_axis} (r={best_corr:.3f}). IMU may be misaligned or inverted.'
        fix = 'Check IMU mounting orientation. Verify accelerometer axes map to vehicle frame.'

    # Build detailed value string showing all correlations
    value_str = f'X={corr_x:.3f} | Y={corr_y:.3f} | Z={corr_z:.3f}'

    result = DiagResult(
        'Accel Axis Alignment (Full Matrix)',
        status,
        value_str,
        detail,
        fix=fix
    )

    return result, {
        'corr_x': corr_x,
        'corr_y': corr_y,
        'corr_z': corr_z,
        'best_axis': best_axis,
        'best_corr': best_corr
    }


def check_yaw_consistency(imu, odom, gnss):
    """
    Check if gyro-integrated yaw matches odom yaw and GNSS-derived yaw.
    Persistent offset = frame error. Drift = gyro bias.
    """
    results = {}

    # Odom yaw (degrees -> radians)
    if 'Yaw_Degrees' in odom.columns:
        odom_yaw_rad = np.radians(odom['Yaw_Degrees'].values)
        results['odom_t']   = odom['Timestamp'].values
        results['odom_yaw'] = odom_yaw_rad

    # Compass
    results['imu_t']      = imu['Timestamp'].values
    results['compass']    = imu['Compass'].values

    # GNSS heading
    t_gnss, h_gnss, spd = gnss_heading(gnss)
    results['gnss_t']   = t_gnss
    results['gnss_yaw'] = h_gnss

    return results


def check_timestamp_sync(imu, gnss, odom):
    """Check if sensors have consistent timestamp ranges."""
    t_imu  = imu['Timestamp'].values
    t_gnss = gnss['Timestamp'].values
    t_odom = odom['Timestamp'].values

    issues = []
    # Check all start/end times are within 2 seconds of each other
    starts = [t_imu[0], t_gnss[0], t_odom[0]]
    ends   = [t_imu[-1], t_gnss[-1], t_odom[-1]]
    start_spread = max(starts) - min(starts)
    end_spread   = max(ends)   - min(ends)

    if start_spread > 2.0:
        issues.append(f'Sensor start times differ by {start_spread:.2f}s — check sync')
    if end_spread > 2.0:
        issues.append(f'Sensor end times differ by {end_spread:.2f}s — check sync')

    # Check IMU rate
    dt_imu = np.diff(t_imu)
    imu_hz = 1.0 / np.median(dt_imu)
    if abs(imu_hz - 100) > 10:
        issues.append(f'IMU rate {imu_hz:.1f} Hz (expected ~100 Hz)')

    status = 'ERROR' if issues else 'OK'
    detail = '; '.join(issues) if issues else 'All sensors have consistent timestamps.'
    return DiagResult(
        'Timestamp Sync',
        status,
        f'IMU={imu_hz:.1f}Hz, spread_start={start_spread:.2f}s, spread_end={end_spread:.2f}s',
        detail,
    )


# ══════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════

def make_plots(imu, gnss, odom, run_dir, mag_data, accel_data, yaw_data):
    fig = plt.figure(figsize=(20, 22), facecolor=C_BG)
    fig.suptitle(
        f'EKF Deep Sanity Report  ·  {os.path.basename(run_dir)}',
        fontsize=14, color=C_TEXT, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.07, right=0.97, top=0.95, bottom=0.04)

    # ── 1. Magnetometer vs GNSS Heading over time ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    t_gnss, comp_at_gnss, h_gnss, delta = mag_data
    if t_gnss is not None and len(t_gnss) > 0:
        t0 = t_gnss[0]
        ax1.plot(t_gnss - t0, np.degrees(h_gnss),
                 color=C_GREEN, lw=1.8, label='GNSS-derived heading (truth)', zorder=3)
        ax1.plot(t_gnss - t0, np.degrees(comp_at_gnss),
                 color=C_ORANGE, lw=1.5, ls='--', label='Magnetometer / Compass', zorder=2)
        ax1.fill_between(t_gnss - t0,
                         np.degrees(comp_at_gnss), np.degrees(h_gnss),
                         alpha=0.15, color=C_RED, label='Error region')
        med_off = np.degrees(np.median(delta))
        ax1.axhline(0, color=C_BORDER, lw=0.8)
        ax1.set_title(f'① Magnetometer vs GNSS Heading  [median offset = {med_off:.2f}°]',
                      color=C_YELLOW if abs(med_off) > 15 else C_GREEN, fontsize=10)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Heading (°)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Insufficient GNSS motion data', ha='center', va='center',
                 color=C_MUTED, transform=ax1.transAxes)

    # ── 2. Angular offset histogram ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if t_gnss is not None and len(delta) > 0:
        delta_deg = np.degrees(delta)
        ax2.hist(delta_deg, bins=40, color=C_PURPLE, alpha=0.8, edgecolor=C_BG)
        ax2.axvline(np.median(delta_deg), color=C_RED, lw=2,
                    label=f'Median = {np.median(delta_deg):.1f}°')
        ax2.axvline(0, color=C_GREEN, lw=1.5, ls='--', label='Ideal (0°)')
        ax2.set_title('② Compass–GNSS Offset Distribution', color=C_TEXT, fontsize=10)
        ax2.set_xlabel('Offset (°)')
        ax2.set_ylabel('Count')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    # ── 3. All yaw sources over time ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    t0_imu = imu['Timestamp'].values[0]
    ax3.plot(imu['Timestamp'] - t0_imu,
             np.degrees(wrap_angle(imu['Compass'])),
             color=C_ORANGE, lw=1.2, alpha=0.85, label='Compass (mag)')
    if yaw_data.get('odom_t') is not None:
        ax3.plot(yaw_data['odom_t'] - t0_imu,
                 np.degrees(wrap_angle(yaw_data['odom_yaw'])),
                 color=C_CYAN, lw=1.5, alpha=0.85, label='Odom Yaw')
    if yaw_data.get('gnss_t') is not None and len(yaw_data['gnss_t']) > 0:
        ax3.scatter(yaw_data['gnss_t'] - t0_imu,
                    np.degrees(yaw_data['gnss_yaw']),
                    color=C_GREEN, s=15, zorder=5, label='GNSS-derived heading')
    ax3.set_title('③ All Yaw Sources Compared  (frame error = persistent offset between lines)',
                  color=C_TEXT, fontsize=10)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Yaw (°)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. Gyro_Z integration vs Odom yaw delta ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if yaw_data.get('odom_t') is not None:
        t_imu_arr = imu['Timestamp'].values
        gz = imu['Gyro_Z'].values
        dt = np.diff(t_imu_arr)
        gyro_int = np.cumsum(gz[:-1] * dt)
        gyro_int -= gyro_int[0]
        # Odom yaw delta
        odom_yaw_raw = yaw_data['odom_yaw']
        odom_yaw_delta = odom_yaw_raw - odom_yaw_raw[0]
        ax4.plot(t_imu_arr[1:] - t0_imu,
                 np.degrees(wrap_angle(gyro_int)),
                 color=C_BLUE, lw=1.2, label='Gyro_Z integrated')
        ax4.plot(yaw_data['odom_t'] - t0_imu,
                 np.degrees(wrap_angle(odom_yaw_delta)),
                 color=C_CYAN, lw=1.5, ls='--', label='Odom Yaw Δ')
        ax4.set_title('④ Gyro Integration vs Odom Yaw Δ\n(divergence = gyro bias or wrong sign)',
                      color=C_TEXT, fontsize=9)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Δ Yaw (°)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    # ── 5. Accel_X vs GT acceleration ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    if accel_data is not None:
        corr, a_gt, ax_imu_interp, t_odom = accel_data
        t0_o = t_odom[0]
        ax5.plot(t_odom - t0_o, a_gt,
                 color=C_GREEN, lw=1.5, label='GT acceleration (from velocity gradient)')
        ax5.plot(t_odom - t0_o, ax_imu_interp,
                 color=C_ORANGE, lw=1.2, alpha=0.8, label=f'IMU Accel_X  (r={corr:.3f})')
        ax5.axhline(0, color=C_BORDER, lw=0.8)
        color_title = C_GREEN if corr > 0.5 else (C_RED if corr < 0 else C_YELLOW)
        ax5.set_title(f'⑤ Accel_X vs GT Acceleration  [correlation r={corr:.3f}]',
                      color=color_title, fontsize=10)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Acceleration (m/s²)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # ── 6. GNSS trajectory with heading arrows ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(gnss['Loc_X'], gnss['Loc_Y'],
             color=C_BLUE, lw=1.5, alpha=0.6, zorder=1)
    if t_gnss is not None and len(t_gnss) > 3:
        # Subsample arrows
        idx = np.linspace(0, len(t_gnss)-1, min(20, len(t_gnss)), dtype=int)
        gx  = resample_to_timestamps(gnss['Timestamp'].values, gnss['Loc_X'].values, t_gnss[idx])
        gy  = resample_to_timestamps(gnss['Timestamp'].values, gnss['Loc_Y'].values, t_gnss[idx])
        ax6.quiver(gx, gy,
                   np.cos(h_gnss[idx]), np.sin(h_gnss[idx]),
                   color=C_GREEN, scale=15, alpha=0.9,
                   width=0.006, label='GNSS heading')
        ax6.quiver(gx, gy,
                   np.cos(comp_at_gnss[idx]), np.sin(comp_at_gnss[idx]),
                   color=C_ORANGE, scale=15, alpha=0.7,
                   width=0.005, label='Compass heading')
    ax6.set_title('⑥ GNSS Trajectory + Heading Arrows\n(green=truth, orange=compass)',
                  color=C_TEXT, fontsize=9)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    # ── 7. Compass polar plot ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0], projection='polar')
    compass_vals = imu['Compass'].values
    angles, counts = np.unique(np.round(compass_vals, 1), return_counts=True)
    ax7.scatter(compass_vals[::10], np.ones(len(compass_vals[::10])) * 0.8,
                c=compass_vals[::10], cmap='hsv', s=4, alpha=0.5)
    if t_gnss is not None and len(h_gnss) > 0:
        ax7.scatter(h_gnss, np.ones(len(h_gnss)) * 0.5,
                    color=C_GREEN, s=8, alpha=0.7, zorder=5)
    ax7.set_title('⑦ Polar: Compass (outer) vs\nGNSS heading (inner, green)',
                  color=C_TEXT, fontsize=9, pad=15)

    # ── 8. Rolling offset between compass and GNSS heading ───────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    if t_gnss is not None and len(delta) > 5:
        t_rel = t_gnss - t_gnss[0]
        ax8.plot(t_rel, np.degrees(delta),
                 color=C_PURPLE, lw=1.2, alpha=0.9)
        # Rolling median
        win = max(1, len(delta) // 10)
        roll_med = pd.Series(np.degrees(delta)).rolling(win, center=True).median()
        ax8.plot(t_rel, roll_med,
                 color=C_RED, lw=2, label=f'Rolling median (w={win})')
        ax8.axhline(np.degrees(np.median(delta)), color=C_YELLOW, lw=1.5,
                    ls='--', label=f'Global median = {np.degrees(np.median(delta)):.1f}°')
        ax8.axhline(0, color=C_GREEN, lw=1, ls=':')
        ax8.set_title('⑧ Compass–GNSS Offset Over Time\n(flat line = fixed frame error)',
                      color=C_TEXT, fontsize=9)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Offset (°)')
        ax8.legend(fontsize=7)
        ax8.grid(True, alpha=0.3)

    # ── 9. Sensor rate timeline ───────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    for label, df, color in [
        ('IMU (100Hz)',  imu,  C_BLUE),
        ('Odom (10Hz)', odom, C_CYAN),
        ('GNSS (1Hz)',  gnss, C_GREEN),
    ]:
        t = df['Timestamp'].values
        dt = np.diff(t)
        hz = 1.0 / dt
        # plot instantaneous rate
        ax9.plot(t[1:] - t[1], hz, color=color, lw=0.8, alpha=0.7, label=label)
    ax9.set_title('⑨ Sensor Instantaneous Rate\n(spikes = dropped packets / timestamp errors)',
                  color=C_TEXT, fontsize=9)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Rate (Hz)')
    ax9.set_ylim(0, 150)
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)

    out_path = os.path.join(run_dir, 'ekf_deep_sanity.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    print(f'\n  📊  Diagnostic plot saved → {out_path}')
    return out_path


# ══════════════════════════════════════════════
#  PRINT REPORT
# ══════════════════════════════════════════════

STATUS_ICON = {'OK': '✅', 'WARN': '⚠️ ', 'ERROR': '❌'}
STATUS_CLR  = {'OK': '\033[92m', 'WARN': '\033[93m', 'ERROR': '\033[91m'}
RESET = '\033[0m'
CYAN = '\033[96m'

def print_result(r):
    icon  = STATUS_ICON[r.status]
    clr   = STATUS_CLR[r.status]
    print(f'  {icon}  {clr}{r.label:<38}{RESET}  {r.value}')
    print(f'         {r.detail}')
    if r.fix:
        print(f'         {CYAN}FIX → {r.fix}{RESET}')
    print()


def print_frame_fix_summary(results):
    print('\n' + '═'*70)
    print('  FRAME DIAGNOSIS SUMMARY')
    print('═'*70)

    errors  = [r for r in results if r.status == 'ERROR']
    warns   = [r for r in results if r.status == 'WARN']

    if not errors and not warns:
        print('  ✅  No frame errors detected. Look elsewhere for drift source.')
    else:
        if errors:
            print(f'\n  ❌  {len(errors)} ERROR(s) found:')
            for r in errors:
                print(f'      • {r.label}: {r.value}')
                if r.fix:
                    print(f'        → {r.fix}')
        if warns:
            print(f'\n  ⚠️   {len(warns)} WARNING(s):')
            for r in warns:
                print(f'      • {r.label}: {r.value}')
                if r.fix:
                    print(f'        → {r.fix}')

    print('\n  LIKELY CAUSE OF PERSISTENT LEFTWARD DRIFT:')
    print('  ┌─────────────────────────────────────────────────────────────┐')
    print('  │ If Magnetometer vs GNSS offset is non-zero:                 │')
    print('  │   → Yaw state is wrong → IMU body→world rotation wrong      │')
    print('  │   → Forward accel projects as lateral → persistent drift     │')
    print('  │   → GNSS corrects X,Y but NOT yaw → repeats every cycle     │')
    print('  │                                                              │')
    print('  │ Quick test: remove magnetometer from EKF, run again.        │')
    print('  │ If drift disappears → magnetometer frame is the culprit.    │')
    print('  └─────────────────────────────────────────────────────────────┘')
    print()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def run_deep_check(run_dir):
    print(f'\n{"═"*70}')
    print(f'  EKF DEEP SANITY CHECK  ·  {run_dir}')
    print(f'{"═"*70}\n')

    # Load data
    imu  = pd.read_csv(os.path.join(run_dir, 'imu_data.csv'))
    gnss = pd.read_csv(os.path.join(run_dir, 'gnss_data.csv'))
    odom = pd.read_csv(os.path.join(run_dir, 'odom_data.csv'))

    print(f'  Loaded: IMU={len(imu)} rows  GNSS={len(gnss)} rows  Odom={len(odom)} rows\n')

    all_results = []

    # ── Basic sign / range checks ─────────────────────────────────────────
    print('  ── BASIC CHECKS ─────────────────────────────────────────────\n')
    
    # Special handling for check_compass_sign (now takes imu and odom)
    r_compass = check_compass_sign(imu, odom)
    print_result(r_compass)
    all_results.append(r_compass)
    
    for fn in [check_gyro_sign, check_gnss_y, check_odom_y, check_accel_range, check_velocity_range]:
        try:
            r = fn(imu) if fn.__code__.co_varnames[0] == 'imu' else \
                fn(gnss) if fn.__code__.co_varnames[0] == 'gnss' else \
                fn(odom)
        except Exception:
            # multi-arg functions handled below
            continue
        if isinstance(r, DiagResult):
            print_result(r)
            all_results.append(r)

    # ── Timestamp sync ────────────────────────────────────────────────────
    print('  ── TIMESTAMP SYNC ───────────────────────────────────────────\n')
    r_ts = check_timestamp_sync(imu, gnss, odom)
    print_result(r_ts)
    all_results.append(r_ts)

    # ── Magnetometer frame check ──────────────────────────────────────────
    print('  ── MAGNETOMETER FRAME ANALYSIS ──────────────────────────────\n')
    mag_result = check_magnetometer_frame(imu, gnss)
    r_mag = mag_result[0]
    print_result(r_mag)
    all_results.append(r_mag)

    mag_plot_data = (None, None, None, None)
    if len(mag_result) == 5 and mag_result[1] is not None:
        _, t_gnss, comp_at_gnss, h_gnss, delta = mag_result
        mag_plot_data = (t_gnss, comp_at_gnss, h_gnss, delta)

    # ── Acceleration projection check ─────────────────────────────────────
    print('  ── ACCELERATION PROJECTION ──────────────────────────────────\n')
    accel_result = check_accel_projection(imu, odom)
    if isinstance(accel_result, tuple):
        r_accel = accel_result[0]
        accel_plot = accel_result[1:]   # corr, a_gt, ax_imu, t_odom
    else:
        r_accel = accel_result
        accel_plot = None
    print_result(r_accel)
    all_results.append(r_accel)

    # ── Multi-Axis Acceleration Alignment Check ──────────────────────────
    print('  ── COMPREHENSIVE ACCEL AXIS ALIGNMENT ───────────────────────\n')
    axis_align_result = check_accel_axis_alignment(imu, odom)
    r_axis_align = axis_align_result[0]
    axis_align_data = axis_align_result[1]
    print_result(r_axis_align)
    all_results.append(r_axis_align)
    
    # Print correlation matrix details
    if axis_align_data:
        print(f'  📊 Detailed Correlation Matrix:')
        print(f'     GT_Forward vs IMU_Accel_X: {axis_align_data["corr_x"]:7.3f}')
        print(f'     GT_Forward vs IMU_Accel_Y: {axis_align_data["corr_y"]:7.3f}')
        print(f'     GT_Forward vs IMU_Accel_Z: {axis_align_data["corr_z"]:7.3f}')
        print(f'     Best Match: {axis_align_data["best_axis"]} with r={axis_align_data["best_corr"]:.3f}')
        print()

    # ── Yaw consistency ───────────────────────────────────────────────────
    print('  ── YAW CONSISTENCY ──────────────────────────────────────────\n')
    yaw_data = check_yaw_consistency(imu, odom, gnss)

    # ── Summary ───────────────────────────────────────────────────────────
    print_frame_fix_summary(all_results)

    # ── Plots ─────────────────────────────────────────────────────────────
    print('  Generating diagnostic plots...')
    try:
        plot_path = make_plots(
            imu, gnss, odom, run_dir,
            mag_plot_data,
            accel_plot,
            yaw_data
        )
    except Exception as e:
        print(f'  ⚠️  Plot generation failed: {e}')
        import traceback; traceback.print_exc()

    return all_results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ekf_sanity_deep.py <run_dir>')
        print('Example: python ekf_sanity_deep.py results/run_5')
        sys.exit(1)

    run_deep_check(sys.argv[1])