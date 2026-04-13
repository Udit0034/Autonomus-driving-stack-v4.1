import math
import carla

class PIDController:
    """PID controller with integral anti-windup clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_clamp: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_clamp = integral_clamp
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # Integrate and clamp (anti-windup)
        self._integral += error * dt
        self._integral = max(-self.integral_clamp, min(self.integral_clamp, self._integral))

        if self._first:
            derivative = 0.0
            self._first = False
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        
        # Output saturation
        output = max(self.output_min, min(self.output_max, output))
        return output

class LongitudinalController:
    """Longitudinal controller: Speed tracking + Jerk limiter."""
    def __init__(self, kp=0.6, ki=0.2, kd=0.15, max_accel=2.5, max_decel=3.0,
                 max_jerk=1.0, max_jerk_emergency=3.0, velocity_filter_alpha=0.3):
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_jerk = max_jerk
        self.max_jerk_emergency = max_jerk_emergency
        self.velocity_filter_alpha = velocity_filter_alpha

        self._prev_accel = 0.0
        self._filtered_speed = 0.0
        self._prev_target_speed = None 
        self._emergency = False

        self._pid = PIDController(kp, ki, kd, output_min=-max_decel, output_max=max_accel)

    def compute(self, current_speed: float, target_speed: float, dt: float) -> float:
        # 1. Velocity low-pass filter (Smooths out sensor noise)
        self._filtered_speed = (self.velocity_filter_alpha * current_speed + 
                                (1.0 - self.velocity_filter_alpha) * self._filtered_speed)

        # 2. Check for emergency stop (target_speed == 0)
        self._emergency = (target_speed < 0.1)

        # 3. Target speed rate limiter (prevents abrupt jumps)
        if self._prev_target_speed is not None:
            rate = 10.0 if self._emergency else 2.0
            max_change = rate * dt
            target_speed = max(self._prev_target_speed - max_change,
                               min(self._prev_target_speed + max_change, target_speed))

        # 4. Feed-forward acceleration
        a_ff = 0.0 if self._prev_target_speed is None else (target_speed - self._prev_target_speed) / dt if dt > 0.0 else 0.0
        a_ff = max(-self.max_decel, min(self.max_accel, a_ff))
        self._prev_target_speed = target_speed

        # v4.1 FIX: Gain scheduling for speed-dependent control
        # Scale down PID gains at low speeds to prevent oscillation
        base_kp = 0.6  # Original kp value
        if current_speed < 5.0:  # Below 5 m/s, reduce gain
            speed_scale = 0.4 + (current_speed / 5.0) * 0.6  # Interpolate from 0.4 to 1.0
            self._pid.kp = base_kp * speed_scale
        else:
            self._pid.kp = base_kp  # Full gain at operational speed

        # 5. PID on speed error
        error = target_speed - self._filtered_speed
        accel_pid = self._pid.compute(error, dt)

        # 6. Combined command
        accel = a_ff + accel_pid
        accel = max(-self.max_decel, min(self.max_accel, accel))

        # 7. Jerk limiter (Comfort constraint)
        if dt > 0.0:
            jerk_limit = self.max_jerk_emergency if self._emergency else self.max_jerk
            max_change = jerk_limit * dt
            accel = max(self._prev_accel - max_change, min(self._prev_accel + max_change, accel))

        self._prev_accel = accel
        return accel

class LateralController:
    """Lateral controller: Heading error tracking (Bicycle Model approximation)."""
    def __init__(self, kp=1.0, ki=0.05, kd=0.1):
        self._pid = PIDController(kp, ki, kd, output_min=-1.0, output_max=1.0)

    def compute(self, ego_transform: carla.Transform, target_waypoint: carla.Waypoint, dt: float) -> float:
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        
        target_loc = target_waypoint.transform.location
        
        # Calculate angle from car to target waypoint
        v_x = target_loc.x - ego_loc.x
        v_y = target_loc.y - ego_loc.y
        target_yaw = math.atan2(v_y, v_x)
        
        # Calculate heading error
        error = target_yaw - ego_yaw
        
        # Normalize error to [-pi, pi] to prevent windup on full circles
        while error > math.pi: error -= 2.0 * math.pi
        while error < -math.pi: error += 2.0 * math.pi

        # Compute PID steering command
        steer = self._pid.compute(error, dt)
        return steer