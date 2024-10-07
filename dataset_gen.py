import numpy as np
import h5py
import open3d as o3d
import os

# Create a directory to store the point clouds if it doesn't exist
output_dir = "output_point_clouds"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
with h5py.File('nyu_depth_v2_labeled.mat', 'r') as f:
    rgb_images = np.array(f['images'])  # Adjust the key name if necessary
    depth_maps = np.array(f['depths'])  # Adjust the key name if necessary
    segmentation_labels = np.array(f['labels'])  # Adjust the key name if necessary

# Print shapes for debugging
print("RGB Images Shape:", rgb_images.shape)
print("Depth Maps Shape:", depth_maps.shape)
print("Segmentation Labels Shape:", segmentation_labels.shape)

# Define a color map for segmentation labels
def label_to_color(label):
    color_map = {
    0: [0, 0, 0],          # unlabeled (black) - position 0
    1: [165, 42, 42],      # book (brown) - position 1
    2: [0, 255, 127],      # bottle (spring green) - position 2
    3: [255, 105, 180],    # cabinet (hot pink) - position 3
    4: [0, 128, 255],      # ceiling (light blue) - position 4
    5: [255, 255, 0],      # chair (yellow) - position 5
    7: [173, 255, 47],     # counter (green-yellow) - position 7
    8: [128, 0, 128],      # dishwasher (purple) - position 8
    9: [70, 130, 180],     # faucet (steel blue) - position 9
    11: [0, 255, 0],       # floor (green) - position 11
    12: [255, 228, 181],   # garbage bin (moccasin) - position 12
    13: [255, 0, 255],     # microwave (magenta) - position 13
    15: [0, 128, 0],       # paper (dark green) - position 15
    16: [255, 20, 147],    # pot (deep pink) - position 16
    17: [255, 69, 0],      # refrigerator (red-orange) - position 17
    18: [30, 144, 255],    # stove burner (dodger blue) - position 18
    19: [0, 0, 255],       # table (blue) - position 19
    21: [255, 0, 0],       # wall (red) - position 21
    22: [255, 222, 173],   # bowl (navajo white) - position 22
    23: [255, 215, 0],     # magnet (gold) - position 23
    24: [0, 128, 0],       # sink (dark green) - position 24
    27: [255, 165, 0],     # door (orange) - position 27
    42: [255, 99, 71],     # shelves (tomato) - position 42
    57: [75, 0, 130],      # green screen (indigo) - position 57
    61: [255, 20, 147],    # bed (deep pink) - position 61
    62: [255, 69, 0],      # light (orange-red) - position 62
    74: [135, 206, 235],   # fan (sky blue) - position 74
    75: [128, 0, 128],     # bookshelf (purple) - position 75
    77: [139, 69, 19],     # sofa (saddle brown) - position 77
    83: [244, 164, 96],    # sofa (sandy brown) - position 83
    124: [0, 255, 255],    # toilet (cyan) - position 100
    122: [128, 128, 128],  # lamp (gray) - position 122
    154: [34, 139, 34],   # placemat (chocolate) - position 154
    172: [255, 239, 213],  # television (papaya whip) - position 172
    292: [255, 182, 193],  # tablecloth (light pink) - position 292
    28:  [210, 105, 30],     # door (forest green) - position 28
    477: [60, 179, 113],   # window frame (medium sea green) - position 477 
    49: [100, 149, 237],   # monitor (cornflower blue) - position 49 
    475: [46, 139, 87],    # desk drawer (sea green) - position 475
    59: [64, 224, 208],    # window (turquoise) - position 59
    174: [255, 215, 185],  # drawer (peach puff) - position 174
    136: [192, 192, 192],  # bathtub (silver) - position 136
    139: [255, 250, 240],  # toilet paper (floral white) - position 139
    143: [107, 142, 35],   # floor mat (olive drab) - position 143
    159: [219, 112, 147],  # deodorant (pale violet red) - position 159
    565: [255, 228, 225],  # toilet bowl brush (misty rose) - position 565
    123: [186, 85, 211],   # shower curtain (medium orchid) - position 123
    89: [72, 209, 204],    # curtain (medium turquoise) - position 89
    80: [255, 160, 122],   # blinds (light salmon) - position 80
    32: [0, 0, 128]        # telephone (navy) - position 32
}

    return np.array(color_map.get(label, [255, 255, 255])) / 255.0  # Default to white for unknown labels and normalize

# Convert RGB-D to 3D Point Cloud
def rgbd_to_point_cloud(rgb_image, depth_image, segmentation_image):
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    
    height, width = depth_image.shape
    points = []
    colors = []

    # Iterate through each pixel in the depth image
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u]
            if z > 0:  # Check if depth is greater than zero
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append((x, y, z))

                # Get the color for the segmentation label
                color = label_to_color(segmentation_image[v, u])
                colors.append(color)

    return np.array(points), np.array(colors)

# Create and save point cloud
def save_point_cloud(points, colors, file_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Colors are based on segmentation labels
    o3d.io.write_point_cloud(file_name, pcd)

# Create and save raw point cloud without labels
def save_ground_truth_point_cloud(points, file_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_name, pcd)

# Process each image and save the corresponding point clouds
for idx in range(rgb_images.shape[0]):
    # Access the RGB, depth, and segmentation images
    rgb_image = rgb_images[idx]
    depth_image = depth_maps[idx]
    segmentation_image = segmentation_labels[idx]

    # Convert to point cloud
    points, colors = rgbd_to_point_cloud(rgb_image, depth_image, segmentation_image)

    # Create a unique file name for each point cloud with semantic labels
    file_name_with_labels = os.path.join(output_dir, f"point_cloud_with_labels_{idx:04d}.pcd")
    
    # Save the point cloud in PCD format with semantic labels
    save_point_cloud(points, colors, file_name_with_labels)

    # Create a unique file name for each ground truth point cloud without labels
    file_name_ground_truth = os.path.join(output_dir, f"point_cloud_ground_truth_{idx:04d}.pcd")

    # Save the point cloud in PCD format without labels (just geometry)
    save_ground_truth_point_cloud(points, file_name_ground_truth)

    print(f"Saved point cloud {idx} to {file_name_with_labels} and {file_name_ground_truth}")

print("All point clouds have been processed and saved.")
