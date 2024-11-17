from controller import Supervisor
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# Initialize Supervisor (no Robot instance here)
supervisor = Supervisor()

# Time step and motor setup
timestep = int(supervisor.getBasicTimeStep())
max_speed = 6.28

# Get robot node (Supervisor node)
robot_node = supervisor.getFromDef("robot123")  # Replace "robot" with your robot's DEF name

if robot_node is None:
    print("Error: Robot node not found. Make sure the DEF name is correct in the world file.")
    supervisor.simulationQuit(1)

# Motor setup
left_motor = supervisor.getDevice('motor_1')
right_motor = supervisor.getDevice('motor_2')
left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

# Device setup
camera = supervisor.getDevice('camera')
camera.enable(timestep)
rf = supervisor.getDevice('range-finder')
rf.enable(timestep)
lidar = supervisor.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

gps = supervisor.getDevice('gps')
gps.enable(timestep)
compass = supervisor.getDevice('compass')
compass.enable(timestep)

# Directories for saving data
path = "../../../dataset/WeBotsDataset/"
directories = ['d', 'rgb', 'point_cloud']
for directory in directories:
    directory_path = os.path.join(path, directory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Helper functions
def get_robot_pose(robot_node):
    """
    Retrieves the robot's position and orientation (rotation matrix) in the world frame.
    """
    position = robot_node.getPosition()  # [x, y, z]
    orientation_matrix = robot_node.getOrientation()  # 3x3 rotation matrix
    return np.array(position), np.array(orientation_matrix)

def transform_point_cloud(local_point_cloud, position, orientation_matrix):
    """
    Transforms a point cloud from the robot's local frame to the world frame.
    """
    transformed_points = []
    for point in local_point_cloud:
        world_point = np.dot(orientation_matrix, point) + position
        transformed_points.append(world_point)
    return np.array(transformed_points)

def save_point_cloud_as_xyz(point_cloud, filename):
    """
    Saves a point cloud to an XYZ file.
    """
    np.savetxt(filename, point_cloud, fmt="%.6f", header="x y z", comments='')

# Main loop
start_time = supervisor.getTime()
last_save_time = start_time
world_point_cloud = []

while supervisor.step(timestep) != -1:
    current_time = supervisor.getTime()
    elapsed_time = current_time - start_time
    time_since_last_save = current_time - last_save_time

    # Stop simulation after 22 seconds
    if elapsed_time >= 22:
        print("Stopping simulation after 22 seconds.")
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break

    # Set motor speeds
    left_speed = 0.825 * max_speed
    right_speed = 1 * max_speed
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # Perform periodic saving
    if time_since_last_save >= 1:
        # Get robot pose
        position, orientation_matrix = get_robot_pose(robot_node)

        # Get local point cloud from LiDAR
        local_point_cloud = lidar.getPointCloud()
        if local_point_cloud:
            # Transform point cloud to world coordinates
            transformed_cloud = transform_point_cloud(local_point_cloud, position, orientation_matrix)
            world_point_cloud.extend(transformed_cloud)

            # Save the point cloud to a file
            filename = os.path.join(path, 'point_cloud', f"apple_pc_{int(elapsed_time)}.xyz")
            save_point_cloud_as_xyz(transformed_cloud, filename)

        # Save depth and RGB images
        rf.saveImage(os.path.join(path, 'd', f"{int(elapsed_time)}_depth.png"), quality=100)
        camera.saveImage(os.path.join(path, 'rgb', f"{int(elapsed_time)}_rgb.png"), quality=100)

        # Update the last save time
        last_save_time = current_time
