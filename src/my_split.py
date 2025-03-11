import os
import shutil
import random

# Set paths
source_root = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database"
destination_root = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database"

# Train-Val-Test split ratios
train_ratio = 0.6
val_ratio = 0.1
test_ratio = 0.3

# Ensure train, val, test directories exist
os.makedirs(os.path.join(destination_root, "train"), exist_ok=True)
os.makedirs(os.path.join(destination_root, "val"), exist_ok=True)
os.makedirs(os.path.join(destination_root, "test"), exist_ok=True)

# Get all class folders, ignoring "Normalized_Images"
classes = [cls for cls in os.listdir(source_root)
           if os.path.isdir(os.path.join(source_root, cls)) and cls != "Normalized_Images"]

for cls in classes:
    class_path = os.path.join(source_root, cls)
    images = [img for img in os.listdir(class_path) if img.endswith(('.bmp', '.png', '.jpg', '.jpeg'))]

    # Shuffle for randomness
    random.shuffle(images)

    # Split data
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create class directories inside train, val, test
    os.makedirs(os.path.join(destination_root, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(destination_root, "val", cls), exist_ok=True)
    os.makedirs(os.path.join(destination_root, "test", cls), exist_ok=True)

    # Copy images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(destination_root, "train", cls, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(destination_root, "val", cls, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(destination_root, "test", cls, img))

    print(f"Class '{cls}' - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

print("Dataset split complete! ")
