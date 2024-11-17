import open3d as o3d
object_name = "apple"
pcd = o3d.io.read_point_cloud(f"../../../dataset/WeBotsDataset/{object_name}/point_cloud/point_cloud.xyz", format='xyz')
o3d.visualization.draw_geometries([pcd])
