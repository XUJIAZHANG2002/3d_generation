import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
class VoxelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing `.npy` files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.transform = transforms.Normalize([0.5],[0.5])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        # voxel_map = np.load(file_path).astype(np.float16)  # Convert to float32 for PyTorch compatibility
        voxel_map = np.load(file_path).astype(np.float32)  # Convert to float32 for PyTorch compatibility
        voxel_map = torch.tensor(voxel_map)
    
        voxel_map = self.transform(voxel_map)
        return voxel_map
