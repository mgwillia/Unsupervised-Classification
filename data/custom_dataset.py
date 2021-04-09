"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        print('anchor transform', self.anchor_transform)
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output

class SCANFDataset(Dataset):
    def __init__(self, dataset, neighbor_indices, stranger_indices, num_neighbors=None, num_strangers=None):
        super(SCANFDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
            self.stranger_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
            self.stranger_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.neighbor_indices = neighbor_indices
        self.stranger_indices = stranger_indices
        if num_neighbors is not None:
            self.neighbor_indices = self.neighbor_indices[:, :num_neighbors+1]
        if num_strangers is not None:
            self.stranger_indices = self.stranger_indices[:, :num_strangers+1]
        assert(self.neighbor_indices.shape[0] == len(self.dataset))
        assert(self.stranger_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.neighbor_indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)
        
        stranger_index = np.random.choice(self.stranger_indices[index], 1)[0]
        stranger = self.dataset.__getitem__(stranger_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        stranger['image'] = self.stranger_transform(stranger['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['stranger'] = stranger['image']
        output['possible_neighbors'] = torch.from_numpy(self.neighbor_indices[index])
        output['possible_strangers'] = torch.from_numpy(self.stranger_indices[index])
        output['target'] = anchor['target']
        
        return output
