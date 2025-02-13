import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

class IITDelhiDataset(Dataset):
    def __init__(self, mode='train', root='dataset'):
        self.root = root
        self.mode = mode
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.root))  # Each class folder (001, 002, ..., 224)
        label_map = {cls: idx for idx, cls in enumerate(classes)}

        all_data = []
        for cls in classes:
            class_path = os.path.join(self.root, cls)
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.bmp')]
            all_data.append((images, label_map[cls]))

        train_data, temp_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        dataset_splits = {'train': train_data, 'val': val_data, 'test': test_data}
        
        for images, label in dataset_splits[self.mode]:
            for img_path in images:
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.labels[index]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, label
