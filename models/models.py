"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out


class HierarchicalClusteringModel(nn.Module):
    def __init__(self, backbone, nbranches, nclusters):
        super(HierarchicalClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        #assert(isinstance(self.nheads, int))
        #assert(self.nheads > 0)
        self.cluster_head = nn.Linear(self.backbone_dim, nclusters)
        self.branch_head = nn.Linear(nclusters, nbranches)
        ## TODO: allow for multiple heads

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            cluster_out = self.cluster_head(features)
            branch_out = self.branch_head(cluster_out)
            out = {'features': features, 'branch_output': branch_out, 'cluster_output': cluster_out}
        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [self.cluster_head(features)]}
        elif forward_pass == 'cluster':
            features = self.backbone(x)
            out = self.cluster_head(features)

        return out


class LinearModel(nn.Module):
    def __init__(self, backbone, nclasses):
        super(LinearModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.fc = nn.Linear(self.backbone_dim, nclasses)

    def forward(self, x, forward_pass='default'):
        features = self.backbone(x)
        out = self.fc(features)
        return out