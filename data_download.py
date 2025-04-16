import kagglehub
import os
import shutil

# Create the data directory path
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)

# Download latest version of the dataset
print("Downloading dataset...")
temp_path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")
print(f"Downloaded to temporary path: {temp_path}")

# Move files to the data directory
for file in os.listdir(temp_path):
    src_file = os.path.join(temp_path, file)
    dst_file = os.path.join(data_dir, file)
    shutil.copy(src_file, dst_file)
    print(f"Copied {file} to {dst_file}")

print(f"Dataset files are now available in: {data_dir}")