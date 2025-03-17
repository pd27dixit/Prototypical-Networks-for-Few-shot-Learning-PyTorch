import os
import re
import shutil

# Define source directories
source_dirs = [
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part1",
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part2"
]

# Define destination base directory
dest_base_dir = "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2"

# Ensure the destination base directory exists
os.makedirs(dest_base_dir, exist_ok=True)

# Regular expression to extract class, side, and image index
pattern = re.compile(r"C(\d+)_S(\d+)_I(\d+)\.tiff")

# Process files
for src_dir in source_dirs:
    if os.path.exists(src_dir):
        for filename in os.listdir(src_dir):
            match = pattern.match(filename)
            if match:
                class_label = int(match.group(1))  # Extract class label
                side = int(match.group(2))  # Extract side
                image_index = int(match.group(3))  # Extract image index

                # Convert class label to 3-digit format (001, 002, ..., 522)
                class_dir = os.path.join(dest_base_dir, f"{class_label:03}")
                os.makedirs(class_dir, exist_ok=True)

                # Convert side: 1 -> L, 2 -> R
                side_letter = "L" if side == 1 else "R"

                # Convert image index to 2-digit format
                new_filename = f"{image_index:02}_{side_letter}.tiff"

                # Copy and rename the file
                src_path = os.path.join(src_dir, filename)
                dest_path = os.path.join(class_dir, new_filename)
                shutil.copy2(src_path, dest_path)

print("File copying and renaming completed successfully!")
