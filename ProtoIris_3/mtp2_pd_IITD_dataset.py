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
        # self.idx_classes = self._index_classes()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        # self.all_items = self._find_items()
        self._index_classes() 

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
        # target = self.idx_classes[class_name]
        target = self.class_to_idx[class_name]

        if self.target_transform:
            target = self.target_transform(target)

        return img_path, target

    def _load_img(self, path, idx):
        """Load an image and apply necessary transformations."""
        img = Image.open(path).convert("L")  # Convert to grayscale
        # img = img.resize((28, 28))  # Resize to 28x28 like Omniglot
        img = img.resize((85, 85))  # Resize to 28x28 like Omniglot
        
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

    # def _index_classes(self):
    #     """Create a mapping from class names to integer labels."""
    #     idx = {cls: i for i, cls in enumerate(self.classes)}
    #     print(f"[DEBUG] Indexed {len(idx)} unique classes.")
    #     return idx

    def _index_classes(self):
        ### OBJECTIVE  
        ### Map person_id to sector_id and build sector-classlist mapping using indices
        self.class_to_sector = {}  ### person_id → sector_id

        for person_id in self.classes:
            sector_id = self._get_sector_for_person(person_id) ### extracts the sector by taking the first two characters of the person ID.
            self.class_to_sector[person_id] = sector_id

        self.sectors = sorted(list(set(self.class_to_sector.values()))) ### extracts all unique sector IDs from the dictionary
        self.sector_to_classlist = {sector: [] for sector in self.sectors} ### maps sector_id-> list of person_ids, init each sector_id to []

        for person_id, sector_id in self.class_to_sector.items():
            class_idx = self.class_to_idx[person_id] ### self.class_to_idx maps person_id → internal index
            self.sector_to_classlist[sector_id].append(class_idx) ### For each person, we append the index to the correct sector's list.

    def _get_sector_for_person(self, person_id):
        # Define logic to extract sector — using first 2 characters here
        return person_id[:2] #logic assigns sector by taking person_id[:2] (e.g., "01A23" → "01").






# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from torchvision import transforms
# from PIL import Image
    
# from sklearn.cluster import KMeans
# import torch
# from PIL import Image
# import torchvision.transforms as transforms



# class IITDelhiDataset(Dataset):
#     def __init__(self, mode='train', root='dataset', transform=None, target_transform=None, model=None, device='cpu'):
#         '''
#         IIT Delhi Dataset structured similarly to OmniglotDataset.
#         - root: dataset directory
#         - mode: 'train', 'val', or 'test'
#         - transform: input image transformation
#         - target_transform: label transformation
#         '''
#         super(IITDelhiDataset, self).__init__()
#         self.root = os.path.join(root, mode) 
#         self.mode = mode
#         self.transform = transform
#         self.target_transform = target_transform
        
#         self.model = model
#         self.device = device

#         # Prepare dataset (find images and labels)
#         self.classes = self._get_classes()
#         self.all_items = self._find_items()
#         # self.idx_classes = self._index_classes()
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
#         # self.all_items = self._find_items()
#         self._index_classes() 

#         # Get paths and labels
#         paths, self.y = zip(*[self.get_path_label(i) for i in range(len(self))])

#         # Lazily load images using map
#         self.x = map(self._load_img, paths, range(len(paths)))
#         self.x = list(self.x)

#         print(f"[DEBUG] Loaded {len(self.x)} images for {self.mode} mode.")

#     def __len__(self):
#         return len(self.all_items)

#     def __getitem__(self, idx):
#         x = self.x[idx]
#         if self.transform:
#             x = self.transform(x)
#         return x, self.y[idx]



#     def get_path_label(self, index):
#         """Retrieve image path and corresponding label."""
#         filename, class_name, root = self.all_items[index]
#         img_path = os.path.join(root, filename)
#         # target = self.idx_classes[class_name]
#         target = self.class_to_idx[class_name]

#         if self.target_transform:
#             target = self.target_transform(target)

#         return img_path, target

#     def _load_image(self, path):

#         transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
#         return transform(Image.open(path))

#     def _get_classes(self):
#         """Retrieve sorted class directories."""
#         if not os.path.exists(self.root):
#             raise ValueError(f"Dataset directory '{self.root}' not found!")

#         classes = sorted([
#             cls for cls in os.listdir(self.root)
#             if os.path.isdir(os.path.join(self.root, cls)) and cls.isdigit()
#         ])
#         print(f"[DEBUG] Found {len(classes)} class directories in {self.mode} mode.")
#         return classes
    
#     def _find_items(self):
#         """Find all image files and associate them with their class labels."""
#         items = []
#         for cls in self.classes:
#             class_path = os.path.join(self.root, cls)
#             images = [
#                 (img, cls, class_path) for img in os.listdir(class_path) if img.endswith('.bmp')
#             ]
#             items.extend(images)
#         print(f"[DEBUG] Found {len(items)} images.")
#         return items
    

#     def _index_classes(self):
#         print("Building class-to-sector mapping using visual clustering...")

#         # Step 1: Extract one embedding per person
#         person_embeddings = []
#         person_ids = []

#         self.model.eval()  # Ensure model is in eval mode

#         with torch.no_grad():
#             for person_id in self.classes:
#                 # Load a sample image for this person
#                 sample_path = self._get_one_sample_path(person_id)
#                 image_tensor = self._load_image(sample_path).unsqueeze(0).to(self.device)  # shape: [1, 1, H, W]

#                 # Extract embedding
#                 embedding = self.model(image_tensor).squeeze().cpu().numpy()
#                 person_embeddings.append(embedding)
#                 person_ids.append(person_id)

#         # Step 2: Perform KMeans clustering
#         num_sectors = 10  # You can choose based on your data size
#         kmeans = KMeans(n_clusters=num_sectors, random_state=42)
#         sector_labels = kmeans.fit_predict(person_embeddings)

#         # Step 3: Map person → sector
#         self.class_to_sector = {pid: f"sector_{label}" for pid, label in zip(person_ids, sector_labels)}

#         # Step 4: Build sector to class list
#         self.sectors = sorted(list(set(self.class_to_sector.values())))
#         self.sector_to_classlist = {sector: [] for sector in self.sectors}

#         for person_id, sector_id in self.class_to_sector.items():
#             class_idx = self.class_to_idx[person_id]
#             self.sector_to_classlist[sector_id].append(class_idx)
            
            
#     def _get_one_sample_path(self, person_id):
#         return self.person_to_samples[person_id][0]  # just grab the first sample
        