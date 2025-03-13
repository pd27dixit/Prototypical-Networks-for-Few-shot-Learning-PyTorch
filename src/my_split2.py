import os
import shutil
import random

# Set paths
source_root = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database"
destination_root = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database"


# Ensure train, val, and test directories exist
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(destination_root, split), exist_ok=True)

# Get all class folders, ignoring "Normalized_Images"
classes = [cls for cls in os.listdir(source_root)
           if os.path.isdir(os.path.join(source_root, cls)) and cls != "Normalized_Images"]

# Shuffle and split the class folders
random.shuffle(classes)
train_classes = classes[:134]
val_classes = classes[134:179]
test_classes = classes[179:]

# Function to copy class folders
def copy_classes(class_list, split):
    for cls in class_list:
        src = os.path.join(source_root, cls)
        dst = os.path.join(destination_root, split, cls)
        shutil.copytree(src, dst)

# Copy folders to respective splits
copy_classes(train_classes, "train")
copy_classes(val_classes, "val")
copy_classes(test_classes, "test")

print("Dataset split completed successfully!")