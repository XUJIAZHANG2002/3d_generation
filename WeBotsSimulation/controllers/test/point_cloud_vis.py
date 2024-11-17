import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# # Load point cloud data
# point_cloud = np.loadtxt("../../../dataset/WeBotsDataset/point_cloud/apple_pc.xyz")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
# plt.show()


import open3d as o3d
pcd = o3d.io.read_point_cloud("../../../dataset/WeBotsDataset/point_cloud/apple_pc.xyz", format='xyz')
o3d.visualization.draw_geometries([pcd])
