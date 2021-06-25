"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        #print(max_prob)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SCANKLLoss(nn.Module):
    def __init__(self, entropy_weight = 0.0, kl_weight = 1.0):
        super(SCANKLLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
        self.kl_weight = kl_weight

    def forward(self, anchors, neighbors, anchor_embeddings, neighbor_embeddings):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
            - anchor_embeddings: embeddings from SimCLR for anchor images w/ shape [b, feature_dim]
            - neighbor_embeddings: embeddings from SimCLR for neighbor images w/ shape [b, feature_dim]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        probs_similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(probs_similarity)
        consistency_loss = self.bce(probs_similarity, ones)
        
        anchors_similarities = torch.matmul(anchors, neighbors.T)
        embeddings_similarities = torch.matmul(anchor_embeddings, neighbor_embeddings.T)

        #print(anchors_similarities.mean(), embeddings_similarities.mean())
        
        soft_anchor_similarities = F.log_softmax(anchors_similarities, dim=1)
        soft_embeddings_similarities = self.softmax(embeddings_similarities)

        kl_loss = F.kl_div(soft_anchor_similarities, soft_embeddings_similarities, reduction='batchmean')
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss +  self.kl_weight * kl_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, kl_loss, entropy_loss


class SCANFLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANFLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors, strangers):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
        negatives_prob = self.softmax(strangers)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Dissimilarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), negatives_prob.view(b, n, 1)).squeeze()
        zeros = torch.zeros_like(similarity)
        stranger_loss = self.bce(similarity, zeros)
        
        #Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        #total_loss = consistency_loss + stranger_loss
        total_loss = consistency_loss + stranger_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, stranger_loss, entropy_loss


class SCANHLoss(nn.Module):
    def __init__(self, overcluster_weight = 1.0, entropy_weight = 2.0, medoid_weight = 0.1):
        super(SCANHLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.overcluster_weight = overcluster_weight
        self.entropy_weight = entropy_weight # Default = 2.0
        self.medoid_weight = medoid_weight


    def forward(self, anchors, neighbors, medoids, medoid_labels):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        a_cluster_output = anchors['cluster_output']
        a_overcluster_output = anchors['overcluster_output']

        n_cluster_output = neighbors['cluster_output']
        n_overcluster_output = neighbors['overcluster_output']

        # Softmax
        b, num_clusters = a_cluster_output.size()
        _, num_overclusters = a_overcluster_output.size()
        a_clusters_prob = self.softmax(a_cluster_output)
        n_clusters_prob = self.softmax(n_cluster_output)

        a_overcluster_prob = self.softmax(a_overcluster_output)
        n_overcluster_prob = self.softmax(n_overcluster_output)
       
        # Similarity in output space
        cluster_similarity = torch.bmm(a_clusters_prob.view(b, 1, num_clusters), n_clusters_prob.view(b, num_clusters, 1)).squeeze()
        overcluster_similarity = torch.bmm(a_overcluster_prob.view(b, 1, num_overclusters), n_overcluster_prob.view(b, num_overclusters, 1)).squeeze()
        ones = torch.ones_like(cluster_similarity)
        cluster_consistency_loss = self.bce(cluster_similarity, ones)
        overcluster_consistency_loss = self.bce(overcluster_similarity, ones)

        # Medoid loss
        medoid_loss = F.cross_entropy(medoids, medoid_labels) * self.medoid_weight
        
        #Entropy loss
        cluster_entropy_loss = entropy(torch.mean(a_clusters_prob, 0), input_as_probabilities = True)
        overcluster_entropy_loss = entropy(torch.mean(a_overcluster_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = (cluster_consistency_loss - self.entropy_weight * cluster_entropy_loss + medoid_loss) + self.overcluster_weight * (overcluster_consistency_loss - self.entropy_weight * overcluster_entropy_loss)

        return total_loss, cluster_consistency_loss, overcluster_consistency_loss, cluster_entropy_loss, overcluster_entropy_loss, medoid_loss


class SCANCLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0, medoid_weight = 0.1):
        super(SCANCLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
        self.medoid_weight = medoid_weight


    def forward(self, anchors, neighbors, medoids, medoid_labels):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """

        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        neighbors_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), neighbors_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Medoid loss
        medoid_loss = F.cross_entropy(medoids, medoid_labels) * self.medoid_weight
        
        #Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss + medoid_loss
        
        return total_loss, consistency_loss, entropy_loss, medoid_loss


class OldSCANCLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(OldSCANCLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors, is_centroid, centroid_labels):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        #Overwrite positives_probs for centroids
        positives_prob[torch.nonzero(is_centroid).reshape(-1)] = torch.eye(200)[centroid_labels[torch.nonzero(is_centroid).reshape(-1)]]
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        #Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss


class SimCLRDistillLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature, distill_alpha):
        super(SimCLRDistillLoss, self).__init__()
        self.temperature = temperature
        self.distill_alpha = distill_alpha

    
    def forward(self, features: torch.Tensor, clusters: torch.Tensor):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
            - clusters: soft label vectors from cluster teacher of shape [b, 2, num_clusters]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0) #features but shape [bx2, dim]
        anchor = features[:,0]

        cluster_features = torch.cat(torch.unbind(clusters, dim=1), dim=0)
        cluster_anchors = clusters[:,0]
        cluster_similarities = torch.matmul(cluster_anchors, cluster_features.T) / self.temperature
        #cluster_similarities = torch.matmul(clusters, clusters.T) / self.temperature
        #anchor_similarities = torch.matmul(anchor, anchor.T) / self.temperature

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        #soft_anchor_similarities = F.log_softmax(anchor_similarities, dim=1)
        soft_anchor_similarities = F.log_softmax(dot_product, dim=1)
        soft_cluster_similarities = F.softmax(cluster_similarities, dim=1)

        distill_loss = F.kl_div(soft_anchor_similarities, soft_cluster_similarities, reduction='batchmean')
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1,2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        print('simclr loss:\t', loss, ';\tdistill loss:\t', distill_loss)

        return loss + distill_loss * self.distill_alpha