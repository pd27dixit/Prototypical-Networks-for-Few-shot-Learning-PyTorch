import os
import re
from PIL import Image

# Define the directories
dirs = [
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part1",
    "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/CLASSES_400_300_Part2"
]

# Regular expression pattern to extract class label
pattern = re.compile(r"C(\d+)_S\d+_I\d+\.tiff")

# Set to store unique class labels
class_labels = set()
total_images = 0
first_image_dimensions = None

# Iterate through directories
for dir_path in dirs:
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            match = pattern.match(filename)
            if match:
                class_labels.add(int(match.group(1)))  # Convert to int for sorting
                total_images += 1  # Count images
                
                # Get image dimensions of the first image
                if first_image_dimensions is None:
                    image_path = os.path.join(dir_path, filename)
                    with Image.open(image_path) as img:
                        first_image_dimensions = img.size  # (width, height)

# Expected class labels from 1 to 522
expected_classes = set(range(1, 523))

# Find missing classes
missing_classes = expected_classes - class_labels

# Print the results
print("Number of unique classes found:", len(class_labels))
print("Number of expected classes:", len(expected_classes))
print("Missing classes:", sorted(missing_classes))
print("Total number of images:", total_images)
if first_image_dimensions:
    print("Image dimensions (Width x Height):", first_image_dimensions)
else:
    print("No images found.")




'''
Number of unique classes found: 518
Number of expected classes: 522
Missing classes: [407, 408, 409, 410]
Total number of images: 11101
Image dimensions (Width x Height): (400, 300)
'''