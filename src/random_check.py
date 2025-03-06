import os
import argparse

def compute_default_classes_per_it(dataset_root):
    """
    Computes default values for classes_per_it_tr and classes_per_it_val 
    based on the number of unique classes in the dataset.
    
    Args:
        dataset_root (str): Path to the dataset directory where each class has its own folder.
    
    Returns:
        tuple: (default_classes_per_it_tr, default_classes_per_it_val)
    """
    # Get all subdirectories (assuming each class has a folder)
    class_folders = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    class_folders = [cls for cls in class_folders if cls.isnumeric()]  # Only numeric class folders

    num_classes = len(class_folders)

    if num_classes == 0:
        raise ValueError("No class directories found in the dataset path.")

    # Compute defaults based on dataset size
    default_classes_per_it_tr = min(max(num_classes // 3, 5), 50)  # Between 5 and 50
    default_classes_per_it_val = min(max(num_classes // 5, 2), 10)  # Between 2 and 10

    return default_classes_per_it_tr, default_classes_per_it_val


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute default values for dataset.")
parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset directory')

args = parser.parse_args()

# Compute defaults
default_tr, default_val = compute_default_classes_per_it(args.dataset_root)

# # Add to argument parser
# parser.add_argument('-cVa', '--classes_per_it_val',
#                     type=int,
#                     default=default_val,
#                     help=f'Number of random classes per episode for validation (default={default_val})')

# parser.add_argument('-cTr', '--classes_per_it_tr',
#                     type=int,
#                     default=default_tr,
#                     help=f'Number of random classes per episode for training (default={default_tr})')

print(f"Computed defaults -> classes_per_it_tr: {default_tr}, classes_per_it_val: {default_val}")

"""
python random_check.py --dataset
_root '/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database'
Computed defaults -> classes_per_it_tr: 50, classes_per_it_val: 10
"""
