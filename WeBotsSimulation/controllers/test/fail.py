import numpy as np
import os
from controller import Robot
from controller import Camera
from controller import Supervisor


# def save_point_cloud_as_xyz(point_cloud, filename="point_cloud.xyz"):
    # with open(filename, "a") as f:
        # for point in point_cloud:
            # f.write(f"{point[0]} {point[1]} {point[2]}\n")
    # print(f"Point cloud saved as {filename}")
def save_point_cloud_as_xyz(point_cloud, filename):
    """
    Saves a point cloud to an XYZ file.
    """
    np.savetxt(filename, point_cloud, fmt="%.6f", header="x y z", comments='')

def get_robot_pose(robotNode):
    # Get position
    position = robot_node.getPosition()  # Returns [x, y, z] in world coordinates
    
    # Get orientation
    orientation = robot_node.getOrientation()  # Returns a 3x3 rotation matrix
    
    return position, orientation
    
def center_point_cloud(point_cloud):
    """
    Centers the given point cloud around (0, 0, 0).

    Args:
        point_cloud: List of tuples [(x, y, z), ...]

    Returns:
        centered_cloud: List of centered points [(x, y, z), ...]
    """
     # Convert point cloud to numpy array for easier computation
    point_cloud = np.array(point_cloud)
    
    # Remove points with NaN or inf values
    valid_points = point_cloud[~np.isnan(point_cloud).any(axis=1) & ~np.isinf(point_cloud).any(axis=1)]
    
    if len(valid_points) == 0:
        print("No valid points to center.")
        return valid_points  # Return empty if all points are invalid
    
    # Compute the centroid of the valid points
    centroid = np.mean(valid_points, axis=0)
    
    # Center the valid points by subtracting the centroid
    centered_point_cloud = valid_points - centroid
    
    return centered_point_cloud

def transform_to_world_frame(local_x, local_y, local_z, robot_x, robot_y, robot_theta):
    if any(np.isnan(val) or np.isinf(val) for val in [local_x, local_y, local_z, robot_x, robot_y, robot_theta]):
        return float('inf'),float('inf'),float('inf')  # Return None to indicate invalid data
    # if robot_x > 0:
        # world_x = robot_x - local_y
    # elif robot_x < 0:
        # world_x = robot_x + local_y
    # if robot_y > 0:    
        # world_y = robot_y - local_x
    # elif robot_y < 0:
        # world_y = robot_y + local_x
    
    # world_z = local_z  # If you are in 2D, the Z value can be constant (or omitted)
    
    # return world_x, world_y, world_z
    # Rotation matrix components
    cos_theta = np.cos(robot_theta)
    sin_theta = np.sin(robot_theta)

     # Rotate local coordinates
    rotated_x = cos_theta * local_x - sin_theta * local_y
    rotated_y = sin_theta * local_x + cos_theta * local_y

    # Translate to world frame
    world_x = rotated_x - robot_y
    world_y = rotated_y - robot_x
    # Z-coordinate (no transformation if working in 2D)
    world_z = local_z  

    return world_x, world_y, world_z
    
 def transform_point_cloud(local_point_cloud, position, orientation_matrix):
    """
    Transforms a point cloud from the robot's local frame to the world frame.
    """
    transformed_points = []
    for point in local_point_cloud:
        world_point = np.dot(orientation_matrix, point) + position
        transformed_points.append(world_point)
    return np.array(transformed_points)
    
robot = Robot()

timestep = int(robot.getBasicTimeStep())
max_speed =6.28
left_motor = robot.getDevice('motor_1')
right_motor = robot.getDevice('motor_2')
camera = robot.getDevice('camera')
camera.enable(64)
rf = robot.getDevice('range-finder')
rf.enable(64)
lidar = robot.getDevice('lidar')
lidar.enable(64)
lidar.enablePointCloud()

gps = robot.getDevice('gps')
gps.enable(timestep)
compass = robot.getDevice('compass')
compass.enable(timestep)

left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)

right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

supervisor = Supervisor()
robot_node = supervisor.getFromDef("robot")  # Replace "ROBOT_NAME" with your robot's DEF name

point_cloud = []
# if not os.path.exists("images"):
    # os.makedirs("images")
path = "../../../dataset/WeBotsDataset/"
directories = ['d','rgb','point_cloud']
for directory in directories:
    directory=path+directory
    if not os.path.exists(directory):
        os.makedirs(directory)

start_time = robot.getTime()
last_save_time = start_time  # Variable to track the last saved time

while robot.step(timestep) != 1:
    current_time = robot.getTime()
    elapsed_time = current_time - start_time
    time_since_last_save = current_time - last_save_time
    if elapsed_time >= 22:
        print("Stopping simulation after 32 seconds.")
        # robot.simulationQuit(0)  # This stops the simulation
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break
    left_speed = 0.825 * max_speed
    right_speed = 1 * max_speed
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    # Main loop
    point_cloud = []  # To accumulate points
    world_point_cloud = []
    if time_since_last_save >= 1:
        position, orientation_matrix = get_robot_pose(robot_node)
        # if elapsed_time >= 21:
        raw_points = lidar.getPointCloud()
        transformed_cloud = transform_point_cloud(local_point_cloud, position, orientation_matrix)
        world_point_cloud.append(transformed_cloud)

        # Save the point cloud to a file
        if len(transformed_point_cloud) > 0:
            save_point_cloud_as_xyz(transformed_point_cloud, filename=path + 'point_cloud/' + 'apple_pc.xyz')
            point_cloud = []  # Reset the point cloud after saving
                
        rf.saveImage(path + 'd/'+ str(int(elapsed_time)) + "_depth.png",quality=100)
        camera.saveImage(path + 'rgb/' + str(int(elapsed_time)) + "_rgb.png",quality=100)

        # Save the point cloud
        last_save_time = current_time
    # raw_points = lidar.getPointCloud()
    # for point in raw_points:
        # local_x, local_y, local_z = point.x, point.y, point.z  # Local frame coordinates
        # print("LIDAR", local_x, local_y, local_z)
        # print("ROBOT", robot_x, robot_y)
        # world_x, world_y, world_z = transform_to_world_frame(local_x, local_y, local_z, robot_x, robot_y, robot_theta)
        # if np.isinf(world_x) or np.isinf(world_y) or np.isinf(world_z):
            # continue
        # point_cloud.append((world_x, world_y, world_z))
    # if len(point_cloud) > 0:
            # save_point_cloud_as_xyz(point_cloud, filename=path + 'point_cloud/' + 'apple_pc.xyz' )
            # point_cloud = []
        