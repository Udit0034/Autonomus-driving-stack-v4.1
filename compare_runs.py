import os
import json

def compare_runs():
    print("\n=== Simulation Metrics Comparison ===")
    
    if not os.path.exists('results'):
        print("No results directory found.")
        return

    # Find all run folders and sort them by number
    existing_runs = [d for d in os.listdir('results') if d.startswith('run_')]
    existing_runs.sort(key=lambda x: int(x.split('_')[1]))

    if len(existing_runs) < 2:
        print("Need at least 2 runs to compare.")
        return

    # Grab the last two runs
    run2_name = existing_runs[-1]
    run1_name = existing_runs[-2]
    
    try:
        with open(f'results/{run1_name}/metrics.json', 'r') as f:
            m1 = json.load(f)
        with open(f'results/{run2_name}/metrics.json', 'r') as f:
            m2 = json.load(f)
    except FileNotFoundError:
        print("Metrics file missing in one of the runs.")
        return

    print(f"Comparing {run1_name} (Previous) to {run2_name} (Current)\n")
    print(f"{'Metric':<20} | {'Current Value':<15} | {'vs Previous'}")
    print("-" * 60)

    def format_row(name, key, unit=""):
        v1, v2 = m1[key], m2[key]
        diff = v2 - v1
        pct = (diff / v1) * 100 if v1 != 0 else 0.0
        
        # Determine arrow (Down is usually better for errors and jerk)
        if pct < -1.0: arrow = "↓" 
        elif pct > 1.0: arrow = "↑"
        else: arrow = "~"

        print(f"{name:<20} | {v2:>7.3f} {unit:<6} | {arrow} {abs(pct):.1f}%")

    format_row("EKF RMSE", "rmse_pos", "m")
    format_row("EKF Mean Error", "mean_error", "m")
    format_row("EKF Max Error", "max_error", "m")
    format_row("Avg Speed", "avg_speed", "m/s")
    format_row("Avg |jerk|", "avg_jerk", "m/s³")
    format_row("Max |jerk|", "max_jerk", "m/s³")
    format_row("RMS jerk", "rms_jerk", "m/s³")
    
    print("-" * 60)
    print(f"Duration: {m2['duration_s']:.1f}s | Samples: {m2['samples']}")

if __name__ == "__main__":
    compare_runs()