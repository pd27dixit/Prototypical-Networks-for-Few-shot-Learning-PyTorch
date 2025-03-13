import os

# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database/test"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Test Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)


# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database/train"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Train Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)

# Define the target directory
target_directory = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database/val"

# List only immediate directories (not internal subdirectories)
immediate_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]

# Print the count and directories
print(f"Val Set - Number of immediate directories: {len(immediate_dirs)}")
# print("Directories:", immediate_dirs)



'''
Test Set - Number of immediate directories: 45
Train Set - Number of immediate directories: 134
Val Set - Number of immediate directories: 45
'''

