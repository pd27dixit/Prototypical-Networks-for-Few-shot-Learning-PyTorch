import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
class IITDelhiDataset(Dataset):
    def __init__(self, mode='train', root='dataset', transform=None, target_transform=None):
        '''
        IIT Delhi Dataset structured similarly to OmniglotDataset.
        - root: dataset directory
        - mode: 'train', 'val', or 'test'
        - transform: input image transformation
        - target_transform: label transformation
        '''
        super(IITDelhiDataset, self).__init__()
        self.root = os.path.join(root, mode) 
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        # Prepare dataset (find images and labels)
        self.classes = self._get_classes()
        self.all_items = self._find_items()
        self.idx_classes = self._index_classes()

        # Get paths and labels
        paths, self.y = zip(*[self.get_path_label(i) for i in range(len(self))])

        # Lazily load images using map
        self.x = map(self._load_img, paths, range(len(paths)))
        self.x = list(self.x)

        print(f"[DEBUG] Loaded {len(self.x)} images for {self.mode} mode.")

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def get_path_label(self, index):
        """Retrieve image path and corresponding label."""
        filename, class_name, root = self.all_items[index]
        img_path = os.path.join(root, filename)
        target = self.idx_classes[class_name]

        if self.target_transform:
            target = self.target_transform(target)

        return img_path, target

    def _load_img(self, path, idx):
        """Load an image and apply necessary transformations."""
        img = Image.open(path).convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 like Omniglot
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        return img

    def _get_classes(self):
        """Retrieve sorted class directories."""
        if not os.path.exists(self.root):
            raise ValueError(f"Dataset directory '{self.root}' not found!")

        classes = sorted([
            cls for cls in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, cls)) and cls.isdigit()
        ])
        print(f"[DEBUG] Found {len(classes)} class directories in {self.mode} mode.")
        return classes
    
    def _find_items(self):
        """Find all image files and associate them with their class labels."""
        items = []
        for cls in self.classes:
            class_path = os.path.join(self.root, cls)
            images = [
                (img, cls, class_path) for img in os.listdir(class_path) if img.endswith('.bmp')
            ]
            items.extend(images)
        print(f"[DEBUG] Found {len(items)} images.")
        return items

    def _index_classes(self):
        """Create a mapping from class names to integer labels."""
        idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"[DEBUG] Indexed {len(idx)} unique classes.")
        return idx
