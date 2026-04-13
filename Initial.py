import carla

def setup_environment():
    """
    Connects to CARLA, loads the map, spawns the ego vehicle, 
    attaches sensors, and returns the key actors.
    """
    # 1. Client & World Setup
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)
    world = client.load_world('Town01')
    blueprint_library = world.get_blueprint_library()

    # 2. Spawn Ego Vehicle (Tesla Model 3)
    tesla_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]
    ego_vehicle = world.spawn_actor(tesla_bp, spawn_point)

    # 3. Setup Sensor Blueprints & Frequencies
    imu_bp = blueprint_library.find('sensor.other.imu')
    gnss_bp = blueprint_library.find('sensor.other.gnss')
   

    imu_bp.set_attribute('sensor_tick', '0.01')
    gnss_bp.set_attribute('sensor_tick', '1.0') # Changed to '1.0' for float consistency

    # 4. Spawn & Attach Sensors
    sensor_transform = carla.Transform(carla.Location(x=0, z=0))
    imu_sensor = world.spawn_actor(imu_bp, sensor_transform, attach_to=ego_vehicle)
    gnss_sensor = world.spawn_actor(gnss_bp, sensor_transform, attach_to=ego_vehicle)


    # 5. Initial Spectator Setup
    spectator = world.get_spectator()
    transform = ego_vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
    )

    # 6. Return all actors packed in a dictionary for easy access in main.py
    return {
        'world': world,
        'ego_vehicle': ego_vehicle,
        'imu_sensor': imu_sensor,
        'gnss_sensor': gnss_sensor,
        'spectator': spectator
    }