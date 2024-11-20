from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

object_name = "valve"
point_cloud = o3d.io.read_point_cloud(f"../../../dataset/WeBotsDataset/{object_name}/point_cloud/point_cloud.xyz", format='xyz')
print("Loaded point cloud:", point_cloud)

voxel_size = 0.01  # Set the voxel size (adjust as needed)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

o3d.io.write_voxel_grid(f"../../../dataset/WeBotsDataset/{object_name}/point_cloud/voxel_grid.ply", voxel_grid)

# o3d.visualization.draw_geometries([voxel_grid])