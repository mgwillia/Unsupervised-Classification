"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
from scipy.spatial.distance import pdist, squareform
from sklearn import cluster
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import numpy as np
import torch
import os


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_cluster_centroids(self, num_clusters):
        features = self.features.cpu().numpy()

        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1, algorithm='full').fit(features)
        cluster_centers= kmeans.cluster_centers_
        similarities = np.matmul(features, np.transpose(cluster_centers))
        print('Similarites shape:', similarities.shape)
        centroid_indices = []
        for i in range(num_clusters):
            centroid_indices.append(np.argmin(similarities[:,i]))

        """
        similarities = squareform(pdist(features))  
        kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', init='k-medoids++').fit(similarities)
        centroid_indices = kmedoids.medoid_indices_
        """
        
        targets = self.targets.cpu().numpy()
        represented_targets = []
        for index in centroid_indices:
            represented_targets.append(targets[index])
        
        num_targets_represented = len(list(set(represented_targets)))

        return centroid_indices, num_targets_represented

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        #features = self.features.cpu().numpy()
        #print(features.shape)
        #similarities = squareform(pdist(features))        
        #indices = np.argpartition(similarities, topk)[:,:topk]

        import faiss
        features = self.features.cpu().numpy()
        _, dim = features.shape[0], features.shape[1]
        #index = faiss.GpuIndexFlatIP(dim)
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index, ngpu=len(os.environ['CUDA_VISIBLE_DEVICES']))
        print(index)
        index.add(features)
        _, indices = index.search(features, topk+1) # Sample itself is included

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            #neighbor_targets = np.take(targets, indices[:,:], axis=0) # Exclude sample itself for eval
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def mine_farthest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk farthest strangers for every sample
        features = self.features.cpu().numpy()
        #normalized_features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
        #similarities = normalized_features.dot(normalized_features.T)
        #similarities = features.dot(features.T)
        similarities = squareform(pdist(features))
        middle_index = int(features.shape[0] / 2)
        indices = np.argpartition(similarities, middle_index)[:,middle_index:middle_index + topk]
        print(indices.shape)       
 
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            stranger_targets = np.take(targets, indices[:,:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(stranger_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
