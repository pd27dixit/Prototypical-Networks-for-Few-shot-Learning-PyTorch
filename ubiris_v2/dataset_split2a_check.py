import os

# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/test"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Test Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)


# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/train"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Train Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)

# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/val"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Val Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)



'''
Test Set - Number of immediate directories: 103
Train Set - Number of immediate directories: 312
Val Set - Number of immediate directories: 103
'''

