"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import random

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


""" 
    TeachersDataset
    Returns an image with its cluster prediction.
"""
class TeachersDataset(Dataset):
    def __init__(self, dataset, cluster_preds_path):
        super(TeachersDataset, self).__init__()
        self.dataset = dataset
        self.cluster_preds = torch.load(cluster_preds_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = self.dataset.__getitem__(index)
        output['cluster_pred'] = self.cluster_preds[index]
        
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

class SCANCDataset(Dataset):
    def __init__(self, dataset, medoid_indices, neighbor_indices, num_neighbors=None):
        super(SCANCDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.medoid_indices = medoid_indices
        self.neighbor_indices = neighbor_indices
        if num_neighbors is not None:
            self.neighbor_indices = self.neighbor_indices[:, :num_neighbors+1]

        assert(self.neighbor_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.neighbor_indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        random_index = random.choice(range(len(self.medoid_indices)))
        random_medoid_index = self.medoid_indices[random_index]
        random_medoid = self.dataset.__getitem__(random_medoid_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        random_medoid['image'] = self.anchor_transform(random_medoid['image'])


        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.neighbor_indices[index])
        output['target'] = anchor['target']
        output['random_medoid_image'] = random_medoid['image']
        output['random_medoid_label'] = random_index

        #print(type(output['anchor']))
        #print(type(output['neighbor']))
        #print(type(output['possible_neighbors']))
        #print(type(output['target']))
        #print(type(output['random_medoid_image']))
        #print(type(output['random_medoid_label']))
        
        return output

class OldSCANCDataset(Dataset):
    def __init__(self, dataset, centroid_indices, neighbor_indices, num_neighbors=None):
        super(SCANCDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.centroid_indices = centroid_indices
        self.neighbor_indices = neighbor_indices
        if num_neighbors is not None:
            self.neighbor_indices = self.neighbor_indices[:, :num_neighbors+1]

        self.is_centroid = []
        self.centroid_labels = []
        counter = 0
        for i in range(len(self.dataset)):
            if i in centroid_indices:
                self.is_centroid.append(True)
                self.centroid_labels.append(counter)
                counter += 1
            else:
                self.is_centroid.append(False)
                self.centroid_labels.append(counter)

        self.is_centroid = torch.tensor(self.is_centroid)
        self.centroid_labels = torch.tensor(self.centroid_labels)

        assert(self.neighbor_indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.neighbor_indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.neighbor_indices[index])
        output['target'] = anchor['target']
        output['is_centroid'] = self.is_centroid[index]
        output['centroid_label'] = self.centroid_labels[index]
        
        return output