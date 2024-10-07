import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the directories
data_dir = 'C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all .pcd files (both ground truth and with labels)
all_files = [f for f in os.listdir(data_dir) if f.endswith('.pcd')]

# Separate ground truth files and label files
ground_truth_files = [f for f in all_files if 'ground_truth' in f]
label_files = [f for f in all_files if 'with_labels' in f]

# Ensure the lengths match (for each ground truth, there should be a corresponding label)
assert len(ground_truth_files) == len(label_files), "Mismatch between ground truth and label files!"

# Sort both lists to ensure corresponding files are aligned
ground_truth_files.sort()
label_files.sort()

# Split into train, validation, and test sets (e.g., 70% train, 15% val, 15% test)
train_gt, temp_gt, train_labels, temp_labels = train_test_split(ground_truth_files, label_files, test_size=0.3, random_state=42)
val_gt, test_gt, val_labels, test_labels = train_test_split(temp_gt, temp_labels, test_size=0.5, random_state=42)

# Helper function to move files
def move_files(file_list, dest_dir):
    for file in file_list:
        shutil.move(os.path.join(data_dir, file), os.path.join(dest_dir, file))

# Move the files into their respective directories
move_files(train_gt + train_labels, train_dir)
move_files(val_gt + val_labels, val_dir)
move_files(test_gt + test_labels, test_dir)

print(f"Data split complete. Train: {len(train_gt)} Validation: {len(val_gt)} Test: {len(test_gt)}")
