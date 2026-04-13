import carla
import random
import pygame
import math
from agents.navigation.basic_agent import BasicAgent
from PID_controller import LongitudinalController, LateralController

class VehicleController:
    def __init__(self, vehicle, world, mode='auto'):
        self.vehicle = vehicle
        self.world = world
        self.mode = mode
        
        if self.mode == 'manual':
            pygame.init()
            self.display = pygame.display.set_mode((300, 200))
            pygame.display.set_caption("CARLA Control Window")
            print("Manual Control Engaged.")

        elif self.mode == 'auto':
            # BasicAgent acts as our High-Level Planner / Hazard Detector
            self.target_speed_kmh = 60.0
            self.agent = BasicAgent(self.vehicle, target_speed=self.target_speed_kmh)
            
            spawn_points = self.world.get_map().get_spawn_points()
            destination = random.choice(spawn_points).location
            self.agent.set_destination(destination)
            
            # Initialize our Custom PID Controllers
            self.long_controller = LongitudinalController()
            self.lat_controller = LateralController()
            
            print(f"Auto Mode Engaged (Custom PID). Navigating to: {destination}")

    def process_control(self, dt: float, est_x: float, est_y: float, est_v: float, est_yaw: float):
        """Call this inside your main loop to apply controls using EKF State."""
        if self.mode == 'manual':
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return False
            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()
            if keys[pygame.K_w]: control.throttle = 1.0
            if keys[pygame.K_s]: control.brake = 1.0
            if keys[pygame.K_a]: control.steer = -0.3
            elif keys[pygame.K_d]: control.steer = 0.3
            if keys[pygame.K_r]: 
                control.reverse = True
                control.throttle = 0.5 
            if keys[pygame.K_SPACE]: control.hand_brake = True
            self.vehicle.apply_control(control)

        elif self.mode == 'auto':
            if self.agent.done():
                print("Destination reached! Calculating a new route...")
                spawn_points = self.world.get_map().get_spawn_points()
                self.agent.set_destination(random.choice(spawn_points).location)
            
            # 1. Let BasicAgent check for hazards and route
            agent_control = self.agent.run_step()
            
            # 2. Cap your cruise speed to the road limit!
            # This prevents the BasicAgent from using its brakes to regulate speed.
            road_speed_limit_kmh = self.vehicle.get_speed_limit()
            if road_speed_limit_kmh > 0:
                cruise_speed_kmh = min(self.target_speed_kmh, road_speed_limit_kmh)
            else:
                cruise_speed_kmh = self.target_speed_kmh
            
            # 3. Highly sensitive Hazard Detection
            # Because cruise_speed prevents overspeeding, ANY brake command from 
            # BasicAgent now guarantees it is looking at a real hazard (red light/car).
            if agent_control.hand_brake or agent_control.brake > 0.3:
                target_speed_ms = 0.0
            else:
                target_speed_ms = cruise_speed_kmh / 3.6
                
            # 2. Build Estimated Transform from EKF
            estimated_transform = carla.Transform(
                carla.Location(x=est_x, y=est_y, z=0.0),
                carla.Rotation(pitch=0.0, yaw=math.degrees(est_yaw), roll=0.0)
            )
            current_speed_ms = est_v
            
            # 3. Get Target Waypoint from Planner
            plan = self.agent.get_local_planner().get_plan()
            target_wp = plan[0][0] if len(plan) > 0 else self.world.get_map().get_waypoint(estimated_transform.location)

            # 4. Run Custom Controllers using ESTIMATED state
            accel = self.long_controller.compute(current_speed_ms, target_speed_ms, dt)
            steer = self.lat_controller.compute(estimated_transform, target_wp, dt)
            
            # 5. Map Acceleration back to CARLA
            control = carla.VehicleControl()
            control.steer = steer
            if accel >= 0.0:
                control.throttle = min(1.0, accel / self.long_controller.max_accel)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(1.0, abs(accel) / self.long_controller.max_decel)

            self.vehicle.apply_control(control)
            
        return True