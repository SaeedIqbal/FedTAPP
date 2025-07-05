import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageFolder
from PIL import Image
import pandas as pd
import numpy as np


class NIH_ChestXray14(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = os.path.join(root, 'nih_chestxray14')
        self.transform = transform
        self.train = train
        # Dummy implementation: Replace with real labels file
        self.image_paths = [os.path.join(self.root, f"{i}.png") for i in range(1000)]  # Example
        self.labels = np.random.randint(0, 14, len(self.image_paths))  # Replace with actual labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class ODIR_100K(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = os.path.join(root, 'odir_100k')
        self.transform = transform
        self.train = train
        self.image_paths = [os.path.join(self.root, f"{i}.jpg") for i in range(1000)]  # Example
        self.labels = np.random.randint(0, 70, len(self.image_paths))  # Replace with actual multi-labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataset_loaders(dataset_name, root='/home/phd/datasets/', batch_size=32, num_clients=5, split='non-iid'):
    """
    Returns list of DataLoaders per client for continual learning.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == 'CIFAR-100':
        dataset = CIFAR100(root=root, train=True, download=True, transform=transform)
        class_per_client = 40
        private_classes = 15
        task_size = 8
        clients = []

        for c in range(num_clients):
            selected_classes = np.random.choice(np.setdiff1d(np.arange(100), []), class_per_client, replace=False)
            indices = np.isin(dataset.targets, selected_classes)
            subset = torch.utils.data.Subset(dataset, np.where(indices)[0])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            clients.append(loader)

    elif dataset_name == 'ImageNet-R':
        path = os.path.join(root, 'imagenet-r')
        dataset = ImageFolder(path, transform=transform)
        class_per_client = 40
        clients = []
        all_classes = np.unique(dataset.targets)
        for c in range(num_clients):
            selected_classes = np.random.choice(all_classes, class_per_client, replace=False)
            indices = np.isin(dataset.targets, selected_classes)
            subset = torch.utils.data.Subset(dataset, np.where(indices)[0])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            clients.append(loader)

    elif dataset_name == 'NIH-ChestX-ray14':
        dataset = NIH_ChestXray14(root=root, transform=transform)
        # Implement non-IID or Dirichlet sampling here
        clients = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for _ in range(num_clients)]

    elif dataset_name == 'ODIR-100K':
        dataset = ODIR_100K(root=root, transform=transform)
        # Implement non-IID or Dirichlet sampling here
        clients = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for _ in range(num_clients)]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return clients