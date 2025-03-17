import os
import re
from collections import defaultdict

# Define the directories
dirs = [
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part1",
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part2"
]

# Regular expression pattern to extract class label
pattern = re.compile(r"C(\d+)_S\d+_I\d+\.tiff")

# Dictionary to store counts for each class label
class_counts = defaultdict(int)

# Iterate through directories
for dir_path in dirs:
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            match = pattern.match(filename)
            if match:
                class_label = int(match.group(1))  # Extract class label
                class_counts[class_label] += 1  # Increment count

# Compute the number of classes with the same image count
image_count_distribution = defaultdict(int)
classes_with_specific_counts = {14: [], 27: [], 31: [], 33: []}

for class_label, count in class_counts.items():
    image_count_distribution[count] += 1
    if count in classes_with_specific_counts:
        classes_with_specific_counts[count].append(class_label)

# Print class-wise image count
print("Class Label | Total Images")
print("--------------------------")
for class_label, count in sorted(class_counts.items()):
    print(f"{class_label:11} | {count:5}")

# Print image count distribution
print("\nImage Count Distribution:")
print("Number of Images | Number of Classes")
print("------------------------------------")
for num_images, num_classes in sorted(image_count_distribution.items()):
    print(f"{num_images:15} | {num_classes:5}")

# Print specific class labels for counts 14, 27, 31, 32
print("\nClass Labels for Specific Image Counts:")
for count in classes_with_specific_counts:
    if classes_with_specific_counts[count]:
        print(f"Classes with {count} images: {sorted(classes_with_specific_counts[count])}")
        
        
'''
Image Count Distribution:
Number of Images | Number of Classes
------------------------------------
             14 |     3
             15 |   293
             27 |     2
             30 |   214
             31 |     4
             33 |     2

Class Labels for Specific Image Counts:
Classes with 14 images: [198, 321, 322]
Classes with 27 images: [503, 504]
Classes with 31 images: [147, 148, 247, 248]
Classes with 33 images: [251, 252]
'''
