import torch
import open3d as o3d
import numpy as np
from pointnet_arch import PointNetSegHead
from data_loader_sampling_remap import PCDDataset

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
NUM_CLASSES = 50  
NUM_POINTS = 10000  
MODEL_PATH = "path_to_trained_model.pth"  
PCD_FILE = "path_to_test_point_cloud.pcd"  

# Load trained model
model = PointNetSegHead(m=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load and preprocess point cloud
def preprocess_point_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # Normalize point cloud
    mean = np.mean(points, axis=0)
    points -= mean
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= scale

    # Convert to tensor and add batch dimension
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(2, 1).to(DEVICE)
    return points_tensor

# Perform inference
def run_inference(points_tensor):
    with torch.no_grad():
        preds, _, _ = model(points_tensor)
        pred_labels = torch.softmax(preds, dim=2).argmax(dim=2).cpu().numpy()
    return pred_labels

# Visualize results
def visualize_inference(points_tensor, pred_labels):
    points_np = points_tensor.squeeze().transpose(1, 0).cpu().numpy()  # Convert back to numpy array
    pred_labels_np = pred_labels.squeeze()

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Define a color map for visualization
    color_map = np.array([[0, 0, 0], [165, 42, 42], [0, 255, 127], [255, 105, 180], [0, 128, 255], 
                          [255, 255, 0], [173, 255, 47], [128, 0, 128], [70, 130, 180], [0, 255, 0], 
                          [255, 228, 181], [255, 0, 255], [0, 128, 0], [255, 20, 147], [255, 69, 0], 
                          [30, 144, 255], [0, 0, 255], [255, 0, 0], [255, 222, 173], [255, 215, 0], 
                          [0, 128, 0], [255, 165, 0], [255, 99, 71], [75, 0, 130], [255, 20, 147], 
                          [255, 69, 0], [135, 206, 235], [128, 0, 128], [139, 69, 19], [244, 164, 96], 
                          [0, 255, 255], [128, 128, 128], [34, 139, 34], [255, 239, 213], [255, 182, 193],
                          [210, 105, 30], [60, 179, 113], [100, 149, 237], [46, 139, 87], [64, 224, 208],
                          [255, 215, 185], [192, 192, 192], [255, 250, 240], [107, 142, 35], [219, 112, 147],
                          [255, 228, 225], [186, 85, 211], [72, 209, 204], [255, 160, 122], [0, 0, 128]]) / 255.0

    # Assign colors based on predicted labels
    colors = np.array([color_map[label] for label in pred_labels_np])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Main inference workflow
def main():
    points_tensor = preprocess_point_cloud(PCD_FILE)
    pred_labels = run_inference(points_tensor)
    visualize_inference(points_tensor, pred_labels)

if __name__ == "__main__":
    main()
