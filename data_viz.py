# import open3d as o3d

# # Function to visualize the point cloud (blocking until window is closed)
# def visualize_pcd(pcd, window_name="Open3D", width=800, height=600):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name, width=width, height=height)
#     vis.add_geometry(pcd)
#     vis.run()  # Will block until the window is closed
#     vis.destroy_window()

# # Load the raw point cloud (ground truth) file
# pcd_raw = o3d.io.read_point_cloud("C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds\\train\\point_cloud_ground_truth_0000.pcd")
# print(f"Number of points in raw point cloud: {len(pcd_raw.points)}")

# # Load the point cloud with labels file
# pcd_with_labels = o3d.io.read_point_cloud("C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds\\train\\point_cloud_with_labels_0000.pcd")
# print(f"Number of points in point cloud with labels: {len(pcd_with_labels.points)}")

# # Visualize the raw point cloud first
# visualize_pcd(pcd_raw, "Raw Point Cloud")

# # Visualize the point cloud with labels after the raw point cloud is closed
# visualize_pcd(pcd_with_labels, "Point Cloud with Labels")


import open3d as o3d
import numpy as np

# Function to visualize the point cloud (blocking until window is closed)
def visualize_pcd(pcd, window_name="Open3D", width=800, height=600):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    vis.add_geometry(pcd)
    vis.run()  # Will block until the window is closed
    vis.destroy_window()

# Load the raw point cloud (ground truth) file
pcd_raw = o3d.io.read_point_cloud("C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\train_gt\\point_cloud_ground_truth_0000.pcd")
points_raw = np.asarray(pcd_raw.points)  # Convert points to NumPy array
print(f"Number of points in raw point cloud: {len(points_raw)}")
print(f"Shape of raw point cloud points: {points_raw.shape}")  # Shape of points

# Load the point cloud with labels file
pcd_with_labels = o3d.io.read_point_cloud("C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds\\train\\point_cloud_with_labels_0000.pcd")
points_with_labels = np.asarray(pcd_with_labels.points)  # Convert points to NumPy array
print(f"Number of points in point cloud with labels: {len(points_with_labels)}")
print(f"Shape of point cloud with labels: {points_with_labels.shape}")  # Shape of points

# Check if the point cloud has colors (used as labels)
if pcd_with_labels.has_colors():
    labels = np.asarray(pcd_with_labels.colors)  # Convert colors to NumPy array (used as labels)
    print(f"Shape of labels in point cloud with labels (from colors): {labels.shape}")
else:
    print("Point cloud with labels has no label information (colors).")

# Visualize the raw point cloud first
visualize_pcd(pcd_raw, "Raw Point Cloud")

# Visualize the point cloud with labels after the raw point cloud is closed
visualize_pcd(pcd_with_labels, "Point Cloud with Labels")
