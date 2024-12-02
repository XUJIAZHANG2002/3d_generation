import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxel_map(voxel_1d_array, visualize = False, save = False, save_dir = None):
    voxel_1d_array += 1
    voxel_1d_array *=0.5
    binary_data = (voxel_1d_array > 0.7).astype(int)


    # Parameters for voxel map
    voxel_size = 0.25
    grid_size = 64

    # Reshape binary data into 8x8x8 by averaging blocks of 4x4x4
    reshaped_data = binary_data.reshape(grid_size, 1,grid_size, 1, grid_size, 1).mean(axis=(1, 3, 5))
    voxel_data = (reshaped_data > 0.5).astype(int)  # Convert to binary based on average

    # Prepare the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D grid for the voxel map (dimensions + 1 to align with voxel corners)
    x, y, z = np.indices((grid_size + 1, grid_size + 1, grid_size + 1)) * voxel_size

    # Display voxels
    filled_voxels = (voxel_data == 1)
    
    ax.voxels(x, y, z, filled_voxels, 
            facecolors="blue", edgecolors="black", alpha=0.7)

    # Set labels and aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('auto')
    plt.title("8x8x8 Voxel Map")
    if visualize:
        # Show plot
        plt.show()
    if save:
        plt.savefig(save_dir)