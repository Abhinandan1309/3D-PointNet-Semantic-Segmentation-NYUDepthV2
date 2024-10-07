import os
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np


class PCDDataset(Dataset):
    def __init__(self, point_cloud_dir, transform=None, downsample=None, num_samples=None, voxel_size=None):
        """
        Args:
            point_cloud_dir (string): Directory with all the .pcd files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            downsample (str, optional): Downsampling method ('random' or 'voxel').
            num_samples (int, optional): Number of points to sample (for random downsampling).
            voxel_size (float, optional): Voxel size (for voxel grid downsampling).
        """
        self.point_cloud_dir = point_cloud_dir
        self.transform = transform
        self.downsample = downsample
        self.num_samples = num_samples
        self.voxel_size = voxel_size
        self.pcd_files = [f for f in os.listdir(point_cloud_dir) if f.endswith('.pcd')]

        # Define a mapping from sparse class labels to continuous class IDs
        self.sparse_to_continuous_map = {
            0: 0,          # unlabeled
            1: 1,          # book
            2: 2,          # bottle
            3: 3,          # cabinet
            4: 4,          # ceiling
            5: 5,          # chair
            7: 6,          # counter
            8: 7,          # dishwasher
            9: 8,          # faucet
            11: 9,         # floor
            12: 10,        # garbage bin
            13: 11,        # microwave
            15: 12,        # paper
            16: 13,        # pot
            17: 14,        # refrigerator
            18: 15,        # stove burner
            19: 16,        # table
            21: 17,        # wall
            22: 18,        # bowl
            23: 19,        # magnet
            24: 20,        # sink
            27: 21,        # door
            42: 22,        # shelves
            57: 23,        # green screen
            61: 24,        # bed
            62: 25,        # light
            74: 26,        # fan
            75: 27,        # bookshelf
            77: 28,        # sofa
            83: 29,        # sofa
            124: 30,       # toilet
            122: 31,       # lamp
            154: 32,       # placemat
            172: 33,       # television
            292: 34,       # tablecloth
            28: 35,        # door
            477: 36,       # window frame
            49: 37,        # monitor
            475: 38,       # desk drawer
            59: 39,        # window
            174: 40,       # drawer
            136: 41,       # bathtub
            139: 42,       # toilet paper
            143: 43,       # floor mat
            159: 44,       # deodorant
            565: 45,       # toilet bowl brush
            123: 46,       # shower curtain
            89: 47,        # curtain
            80: 48,        # blinds
            32: 49         # telephone
        }
    
    def __len__(self):
        return len(self.pcd_files)
    
    def _load_pcd(self, file_path):
        """
        Load a point cloud from a .pcd file and return the points and labels (if available).
        """
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        
        # If the pcd file has colors, treat them as labels. Map them to class IDs.
        if pcd.has_colors():
            labels = np.asarray(pcd.colors)
            labels = self._map_colors_to_class_ids(labels)  # Convert RGB to class IDs
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int64)  # Default to class 0 if no labels available

        # Downsample the point cloud (if applicable)
        if self.downsample == 'random' and self.num_samples:
            points, labels = self.random_sampling(points, labels, self.num_samples)
        elif self.downsample == 'voxel' and self.voxel_size:
            points, labels = self.voxel_grid_downsampling(points, labels, self.voxel_size)

        return points, labels

    def random_sampling(self, points, labels, num_samples):
        """Randomly sample a fixed number of points from the point cloud."""
        assert len(points) == len(labels), "Points and labels should have the same length"
        indices = np.random.choice(len(points), num_samples, replace=False)
        return points[indices], labels[indices]

    def voxel_grid_downsampling(self, points, labels, voxel_size):
        """Downsample a point cloud using voxel grid filtering."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        downsampled_points = np.asarray(pcd.points)
        indices = [np.argmin(np.linalg.norm(points - dp, axis=1)) for dp in downsampled_points]
        downsampled_labels = labels[indices]
        return downsampled_points, downsampled_labels

    def _map_colors_to_class_ids(self, colors):
        """Map RGB colors to class IDs and remap them to continuous class IDs."""
        color_to_class = {
            (0, 0, 0): 0,          # unlabeled (black)
            (165, 42, 42): 1,      # book (brown)
            (0, 255, 127): 2,      # bottle (spring green)
            (255, 105, 180): 3,    # cabinet (hot pink)
            (0, 128, 255): 4,      # ceiling (light blue)
            (255, 255, 0): 5,      # chair (yellow)
            (173, 255, 47): 7,     # counter (green-yellow)
            (128, 0, 128): 8,      # dishwasher (purple)
            (70, 130, 180): 9,     # faucet (steel blue)
            (0, 255, 0): 11,       # floor (green)
            (255, 228, 181): 12,   # garbage bin (moccasin)
            (255, 0, 255): 13,     # microwave (magenta)
            (0, 128, 0): 15,       # paper (dark green)
            (255, 20, 147): 16,    # pot (deep pink)
            (255, 69, 0): 17,      # refrigerator (red-orange)
            (30, 144, 255): 18,    # stove burner (dodger blue)
            (0, 0, 255): 19,       # table (blue)
            (255, 0, 0): 21,       # wall (red)
            (255, 222, 173): 22,   # bowl (navajo white)
            (255, 215, 0): 23,     # magnet (gold)
            (0, 128, 0): 24,       # sink (dark green)
            (255, 165, 0): 27,     # door (orange)
            (255, 99, 71): 42,     # shelves (tomato)
            (75, 0, 130): 57,      # green screen (indigo)
            (255, 20, 147): 61,    # bed (deep pink)
            (255, 69, 0): 62,      # light (orange-red)
            (135, 206, 235): 74,   # fan (sky blue)
            (128, 0, 128): 75,     # bookshelf (purple)
            (139, 69, 19): 77,     # sofa (saddle brown)
            (244, 164, 96): 83,    # sofa (sandy brown)
            (0, 255, 255): 124,    # toilet (cyan)
            (128, 128, 128): 122,  # lamp (gray)
            (34, 139, 34): 154,    # placemat (chocolate)
            (255, 239, 213): 172,  # television (papaya whip)
            (255, 182, 193): 292,  # tablecloth (light pink)
            (210, 105, 30): 28,    # door (forest green)
            (60, 179, 113): 477,   # window frame (medium sea green)
            (100, 149, 237): 49,   # monitor (cornflower blue)
            (46, 139, 87): 475,    # desk drawer (sea green)
            (64, 224, 208): 59,    # window (turquoise)
            (255, 215, 185): 174,  # drawer (peach puff)
            (192, 192, 192): 136,  # bathtub (silver)
            (255, 250, 240): 139,  # toilet paper (floral white)
            (107, 142, 35): 143,   # floor mat (olive drab)
            (219, 112, 147): 159,  # deodorant (pale violet red)
            (255, 228, 225): 565,  # toilet bowl brush (misty rose)
            (186, 85, 211): 123,   # shower curtain (medium orchid)
            (72, 209, 204): 89,    # curtain (medium turquoise)
            (255, 160, 122): 80,   # blinds (light salmon)
            (0, 0, 128): 32        # telephone (navy)
        }
        
        labels = []
        for color in colors:
            color_tuple = tuple((color * 255).astype(int))  # Convert to 0-255 RGB
            class_id = color_to_class.get(color_tuple, 0)  # Default to class 0 if not found
            
            # Remap to continuous labels
            remapped_class_id = self.sparse_to_continuous_map.get(class_id, 0)  # Default to remapped class 0
            labels.append(remapped_class_id)

        return np.array(labels, dtype=np.int64)
    
    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
        
    #     pcd_file = os.path.join(self.point_cloud_dir, self.pcd_files[idx])
    #     points, labels = self._load_pcd(pcd_file)
        
    #     sample = {'points': points, 'labels': labels}
        
    #     if self.transform:
    #         sample = self.transform(sample)
        
    #     return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pcd_file = os.path.join(self.point_cloud_dir, self.pcd_files[idx])
        points, labels = self._load_pcd(pcd_file)
        
        # Convert points and labels to PyTorch tensors
        points = torch.tensor(points, dtype=torch.float32)  # Assuming points are floats
        labels = torch.tensor(labels, dtype=torch.long)  # Assuming labels are integers
        
        sample = {'points': points, 'labels': labels}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class NormalizePointCloud:
    """Normalize the point cloud to have zero mean and unit variance."""
    
    def __call__(self, sample):
        points, labels = sample['points'], sample['labels']
        
        # Normalize points to zero mean and unit variance
        mean = torch.mean(points, axis=0)
        points -= mean
        scale = torch.max(torch.norm(points, dim=1))
        points /= scale
        
        return {'points': points, 'labels': labels}


def get_dataloader(point_cloud_dir, batch_size=32, shuffle=True, num_workers=2, downsample=None, num_samples=None, voxel_size=None):
    """
    Utility function to create a DataLoader for point cloud data with downsampling options.

    Args:
        point_cloud_dir (str): Directory where .pcd files are stored.
        batch_size (int): Number of point clouds per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        downsample (str): Downsampling method ('random' or 'voxel').
        num_samples (int): Number of points to sample for random downsampling.
        voxel_size (float): Voxel size for voxel grid downsampling.

    Returns:
        DataLoader: PyTorch DataLoader for the point cloud data.
    """
    dataset = PCDDataset(point_cloud_dir=point_cloud_dir, transform=NormalizePointCloud(), downsample=downsample, num_samples=num_samples, voxel_size=voxel_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def visualize_sample(dataloader):
    """ Utility function to visualize a sample from the dataset """
    import open3d as o3d
    for batch in dataloader:
        points = batch['points'][0].cpu().numpy()  # Convert to numpy array
        labels = batch['labels'][0].cpu().numpy()  # Convert to numpy array

        # Convert points and labels to Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Visualize the point cloud with semantic labels
        o3d.visualization.draw_geometries([pcd])
        break  # Visualize only the first sample
