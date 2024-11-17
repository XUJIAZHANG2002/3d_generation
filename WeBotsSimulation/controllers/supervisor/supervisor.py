from controller import Robot
import numpy as np
import os
import math

# Initialize the Robot
robot = Robot()
object_name = "apple"
# Time step and motor setup
timestep = int(robot.getBasicTimeStep())
max_speed = 6.28

# Get device handles
left_motor = robot.getDevice('motor_1')
right_motor = robot.getDevice('motor_2')
camera = robot.getDevice('camera')
rf = robot.getDevice('range-finder')
lidar = robot.getDevice('lidar')
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')
imu = robot.getDevice('imu')

# Enable devices
camera.enable(timestep)
rf.enable(timestep)
lidar.enable(timestep)
lidar.enablePointCloud()
gps.enable(timestep)
compass.enable(timestep)
imu.enable(timestep)

# Motor setup
left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

# Directory setup for saving data
path = f"../../../dataset/WeBotsDataset/{object_name}"
directories = ['d', 'rgb', 'point_cloud']
for directory in directories:
    directory_path = os.path.join(path, directory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Helper functions
def get_robot_pose(gps, imu):
    """
    Retrieves the robot's position (GPS) and orientation (IMU) in the world frame.
    Args:
        gps: Webots GPS device.
        imu: Webots IMU device.
    Returns:
        position: A numpy array [x, y, z].
        quaternion: A list [qx, qy, qz, qw] representing the orientation.
    """
    # Get position from GPS
    position = np.array(gps.getValues())  # [x, y, z]

    # Get quaternion from IMU
    quaternion = imu.getQuaternion()  # [qx, qy, qz, qw]
    
    return position, quaternion

    
def quaternion_to_rotation_matrix(quaternion):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    Args:
        quaternion: A list [qw, qx, qy, qz].
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    qx, qy, qz, qw = quaternion
    # Compute the rotation matrix elements
    rotation_matrix = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return rotation_matrix

def transform_point_cloud(local_point_cloud, position, quaternion):
    """
    Transforms a point cloud from the robot's local frame to the world frame using axis-angle rotation.
    Args:
        local_point_cloud: List of LidarPoint objects.
        position: A numpy array [x, y, z] representing the robot's position.
        quaternion: A list [qw, qx, qy, qz] representing the robot's orientation.
    Returns:
        A numpy array of transformed points in the world frame.
    """
    # Normalize quaternion to avoid invalid values
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    rotated_points = []
    for point in local_point_cloud:
        # Extract x, y, z from LidarPoint and create a NumPy array
        local_point = np.array([point.x, point.y, point.z])
        rotated_point = np.dot(rotation_matrix, local_point)
        rotated_point += position
        rotated_points.append(rotated_point)
    
    return np.array(rotated_points)

def save_point_cloud_as_xyz(transformed_cloud, filename):
    """
    Saves a point cloud to an XYZ file.
    """
    with open(filename, 'a') as f:
        # Write each point to the file
        for point in transformed_cloud:
            if all(not (math.isnan(coord) or math.isinf(coord)) for coord in point):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                
                
def main():
    # Main loop
    start_time = robot.getTime()
    last_save_time = start_time
    world_point_cloud = []
    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        elapsed_time = current_time - start_time
        time_since_last_save = current_time - last_save_time
    
        # Stop simulation after 22 seconds
        if elapsed_time >= 23:
            print("Stopping simulation after 22 seconds.")
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            return 0
    
        # Set motor speeds
        left_speed = 0.825 * max_speed
        right_speed = 1 * max_speed
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
    
        # Perform periodic saving
        if time_since_last_save >= 1:
            # Get robot pose using GPS and Compass
            position, quaternion = get_robot_pose(gps, imu)
            local_point_cloud = lidar.getPointCloud()
            
            if local_point_cloud:
                # Transform point cloud to world coordinates
                transformed_cloud = transform_point_cloud(local_point_cloud, position, quaternion)
                world_point_cloud.extend(transformed_cloud)
    
                # Save the point cloud to a file
                filename = os.path.join(path, 'point_cloud', f"point_cloud.xyz")
                save_point_cloud_as_xyz(transformed_cloud, filename)
    
            # Save depth and RGB images
            rf.saveImage(os.path.join(path, 'd', f"{int(elapsed_time)}_depth.png"), quality=100)
            camera.saveImage(os.path.join(path, 'rgb', f"{int(elapsed_time)}_rgb.png"), quality=100)
    
            # Update the last save time
            last_save_time = current_time

main()