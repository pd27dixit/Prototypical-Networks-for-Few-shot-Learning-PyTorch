import os

# Define dataset directories
dataset_dirs = {
    "Test": "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/test",
    "Train": "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/train",
    "Val": "/old/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2/val",
}

# Check minimum number of images per class for each set
print("\n=== Minimum Images Per Class in Each Dataset ===\n")

for dataset_name, dataset_path in dataset_dirs.items():
    if os.path.exists(dataset_path):
        image_counts = []
        
        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                image_counts.append((class_dir, num_images))
        
        # Find the minimum count
        if image_counts:
            min_images = min(image_counts, key=lambda x: x[1])[1]
            min_classes = [class_label for class_label, count in image_counts if count == min_images]

            print(f"{dataset_name} Set:")
            print(f"   → Minimum images per class: **{min_images}**")
            print(f"   → Classes with this count: {', '.join(min_classes)}\n")
        else:
            print(f"{dataset_name} Set: No class directories found.\n")
            
            
            
'''
=== Minimum Images Per Class in Each Dataset ===

Test Set:
   → Minimum images per class: **15**
   → Classes with this count: 113, 473, 182, 375, 471, 350, 146, 374, 265, 058, 249, 055, 423, 220, 038, 470, 497, 464, 419, 070, 430, 219, 123, 347, 400, 203, 230, 384, 486, 299, 429, 028, 302, 209, 513, 415, 214, 051, 292, 098, 063, 076, 142, 373, 454, 078, 457, 476, 137, 217, 447, 283, 500, 399, 353, 068, 288, 397, 169, 161, 431, 155

Train Set:
   → Minimum images per class: **14**
   → Classes with this count: 198, 321, 322

Val Set:
   → Minimum images per class: **15**
   → Classes with this count: 158, 120, 494, 031, 030, 487, 241, 398, 499, 284, 290, 295, 011, 221, 327, 305, 059, 065, 021, 272, 296, 160, 342, 004, 438, 211, 492, 145, 212, 072, 263, 282, 135, 223, 170, 312, 016, 204, 333, 479, 159, 054, 179, 315, 180, 232, 075, 153, 477, 194, 273, 334, 291, 141, 437, 411, 193
'''
