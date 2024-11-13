import numpy as np
import os
from controller import Robot
from controller import Camera


def save_point_cloud_as_xyz(point_cloud, filename="point_cloud.xyz"):
    with open(filename, "a") as f:
        for point in point_cloud:
            # Access the x, y, z attributes of each LidarPoint object
            # f.write(f"{point.x} {point.y} {point.z}\n")
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"Point cloud saved as {filename}")

def get_robot_pose(gps, compass):
    # Get the position (x, y) from the GPS
    position = gps.getValues()
    x, y = position[0], position[1]
    
    # Get orientation from compass (returns a vector)
    north = compass.getValues()
    theta = np.arctan2(north[0], north[2]-1.57)  # Orientation in radians
    # print(x,y,theta)
    
    return x, y, theta

def transform_to_world_frame(local_x, local_y, local_z, robot_x, robot_y, robot_theta):
    if any(np.isnan(val) or np.isinf(val) for val in [local_x, local_y, local_z, robot_x, robot_y, robot_theta]):
        return float('inf'),float('inf'),float('inf')  # Return None to indicate invalid data
    
    world_x = robot_x - local_x
    world_y = robot_y - local_y
    world_z = local_z  # If you are in 2D, the Z value can be constant (or omitted)
    
    return world_x, world_y, world_z
    
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
point_cloud = []
# if not os.path.exists("images"):
    # os.makedirs("images")
path = "../../dataset/WeBotsDatset/"
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
    if elapsed_time >= 32:
        print("Stopping simulation after 32 seconds.")
        # robot.simulationQuit(0)  # This stops the simulation
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break
    left_speed = 0.88 * max_speed
    right_speed = 1 * max_speed
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    
    
    robot_x, robot_y, robot_theta = get_robot_pose(gps, compass)
    
    # Save RGB image after every second
    if time_since_last_save >= 1:
        raw_points = lidar.getPointCloud()
        for point in raw_points:
        # print(point.x,point.y,point.z)
            local_x, local_y, local_z = point.x, point.y, point.z  # Local frame coordinates
            world_x, world_y, world_z = transform_to_world_frame(local_x, local_y, local_z, robot_x, robot_y, robot_theta)
            if np.isinf(world_x) or np.isinf(world_y) or np.isinf(world_z):
                continue
            point_cloud.append((world_x, world_y, world_z))
            
        rf.saveImage(path + 'd/'+ str(int(elapsed_time)) + "_depth.png",quality=100)
        camera.saveImage(path + 'rgb/' + str(int(elapsed_time)) + "_rgb.png",quality=100)

        # Save the point cloud
        if len(point_cloud) > 0:
            save_point_cloud_as_xyz(point_cloud, filename=path + 'point_cloud/' + 'apple_pc.xyz' )
            point_cloud = []
        last_save_time = current_time
        