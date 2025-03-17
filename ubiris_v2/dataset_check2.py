import os
import re
from collections import defaultdict

# Define the directories
dirs = [
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part1",
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part2"
]

# Regular expression pattern to extract class label and side
pattern = re.compile(r"C(\d+)_S(\d+)_I\d+\.tiff")

# Dictionary to store counts for (class_label, side)
class_side_counts = defaultdict(int)

# Iterate through directories
for dir_path in dirs:
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            match = pattern.match(filename)
            if match:
                class_label = int(match.group(1))  # Extract class label
                side = int(match.group(2))  # Extract side
                class_side_counts[(class_label, side)] += 1

# Print the results
print("Class Label | Side | Image Count")
print("---------------------------------")
for (class_label, side), count in sorted(class_side_counts.items()):
    print(f"{class_label:11} | {side:4} | {count:5}")
    
'''
        503 |    1 |    12
        503 |    2 |    15
        504 |    1 |    12
        504 |    2 |    15
'''
