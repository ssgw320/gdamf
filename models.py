#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary


# In[2]:


class GDAMF(nn.Module):
    """
    @param
    num_labels : int, number of class labels
    num_domains : int, number of domains, ex. source/inter/target -> 3
    input_dim : int -> fully connect, tuple -> Conv2D
    hidden_dim : int, latent feature dim
    """

    def __init__(self, num_labels, num_domains, input_dim, hidden_dim):
        super(GDAMF, self).__init__()
        self.network = nn.ModuleList()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        for i in range(num_domains):
            if i == 0:
                self.network.append(MLP(num_labels, input_dim, hidden_dim, source=True))
            else:
                self.network.append(MLP(num_labels, input_dim, hidden_dim, source=False))

    def forward(self, x, domain):
        for i in range(domain):
            if i == 0 :
                feature, pred_y = self.network[i](x)
            else:
                feature, pred_y = self.network[i](feature, pred_y)
        return feature, pred_y


class MLP(nn.Module):
    def __init__(self, num_labels, input_dim, hidden_dim, source=False):
        super(MLP, self).__init__()
        if isinstance(input_dim, tuple):
            # image data
            num_conv = 3
            conv_dim = np.rint(np.array(input_dim) / 2**num_conv)
            latent_dim = int(conv_dim[0] * conv_dim[1] * hidden_dim)
            if source:
                conv_settings = dict(kernel_size=5, stride=2, padding=2)
                self.fc = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=hidden_dim, **conv_settings),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, **conv_settings),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, **conv_settings),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.5),
                                        nn.BatchNorm2d(num_features=hidden_dim),
                                        nn.Flatten(),)
                self.pred = nn.Linear(latent_dim, num_labels)
            else:
                self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=latent_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.BatchNorm1d(num_features=latent_dim))
                self.pred = nn.Linear(latent_dim+num_labels, num_labels)
        else:
            # table data
            seq = [nn.ReLU(), nn.Dropout(p=0.2), nn.BatchNorm1d(num_features=hidden_dim)]
            if source:
                seq.insert(0, nn.Linear(input_dim, hidden_dim))
                self.fc = nn.Sequential(*seq)
                self.pred = nn.Linear(hidden_dim, num_labels)
            else:
                seq.insert(0, nn.Linear(hidden_dim, hidden_dim))
                self.fc = nn.Sequential(*seq)
                self.pred = nn.Linear(hidden_dim+num_labels, num_labels)

    def forward(self, x, pseudoY=None):
        feature = self.fc(x)
        if pseudoY is None:
            pred_y = self.pred(feature)
        else:
            # Add the pseudo-labels
            pred_feature = torch.cat([feature, pseudoY], dim=1)
            pred_y = self.pred(pred_feature)
        return feature, pred_y


# In[12]:


class AuxiliaryModel(nn.Module):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    """

    def __init__(self, num_labels, input_dim, hidden_dim):
        super(AuxiliaryModel, self).__init__()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.network = nn.ModuleList()
        self.network.append(MLP(num_labels, input_dim, hidden_dim, source=True))

    def forward(self, x, domain=None):
        """ domain args does not need. fix for other utils """
        _, pred_y = self.network[0](x)
        return _, pred_y

    def get_prediction_with_uniform_prior(self, soft_prediction):
        soft_prediction_uniform = soft_prediction / soft_prediction.sum(0, keepdim=True).pow(0.5)
        soft_prediction_uniform /= soft_prediction_uniform.sum(1, keepdim=True)
        return soft_prediction_uniform

    def classifier_prediction(self, x_source:torch.Tensor):
        with torch.no_grad():
            _, pred_network = self.forward(x_source)
            pred_network = F.softmax(pred_network, dim=1)
        pred_network = self.get_prediction_with_uniform_prior(pred_network)
        pseudo_y = pred_network.argmax(dim=1)
        return pred_network, pseudo_y

    def ensemble_prediction(self, x_source:torch.Tensor, y_source:torch.Tensor, x_target:torch.Tensor):
        """ use only for self train """
        pred_network, _, = self.classifier_prediction(x_target)
        pred_kmeans = self.get_labels_from_kmeans(x_source, y_source, x_target)
        pred_lp = self.get_labels_from_lp(x_source, y_source, x_target)
        pred_kmeans = self.get_prediction_with_uniform_prior(pred_kmeans)
        pred_lp = self.get_prediction_with_uniform_prior(pred_lp)
        pred_ensemble = (pred_network + pred_kmeans + pred_lp) / 3
        pseudo_y = pred_ensemble.argmax(dim=1)
        return pred_ensemble, pseudo_y

    def get_labels_from_kmeans(self, x_source:torch.Tensor, y_source:torch.Tensor, x_target:torch.Tensor):
        with torch.no_grad():
            z_source, _ = self.network[0](x_source)
            z_target, _ = self.network[0](x_target)
        z_source_array = z_source.numpy()
        y_source_array = y_source.numpy()
        z_target_array = z_target.numpy()
        init = np.vstack([z_source_array[y_source_array==i].mean(axis=0) for i in np.unique(y_source_array)])
        kmeans = KMeans(n_clusters=self.num_labels, init=init, n_init=1, random_state=0).fit(z_target_array)
        centers = kmeans.cluster_centers_  # num_category * feature_dim
        centers_tensor = torch.from_numpy(centers)
        centers_tensor_unsq = torch.unsqueeze(centers_tensor, 0)
        target_u_feature_unsq = torch.unsqueeze(z_target, 1)
        L2_dis = ((target_u_feature_unsq - centers_tensor_unsq)**2).mean(2)
        soft_label_kmeans = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
        return soft_label_kmeans

    def get_labels_from_lp(self, x_source:torch.Tensor, y_source:torch.Tensor, x_target:torch.Tensor):
        """ label propagation """
        graphk = 20
        alpha = 0.75
        with torch.no_grad():
            labeled_features, _ = self.network[0](x_source)
            unlabeled_features, _ = self.network[0](x_target)
        labeled_onehot_gt = F.one_hot(y_source, num_classes=self.num_labels)

        num_labeled = labeled_features.size(0)
        if num_labeled > 100000:
            print('too many labeled data, randomly select a subset')
            indices = torch.randperm(num_labeled)[:10000]
            labeled_features = labeled_features[indices]
            labeled_onehot_gt  = labeled_onehot_gt[indices]
            num_labeled = 10000

        num_unlabeled = unlabeled_features.size(0)
        num_all = num_unlabeled + num_labeled
        all_features = torch.cat((labeled_features, unlabeled_features), dim=0)
        unlabeled_zero_gt = torch.zeros(num_unlabeled, self.num_labels)
        all_gt = torch.cat((labeled_onehot_gt, unlabeled_zero_gt), dim=0)
        ### calculate the affinity matrix
        all_features = F.normalize(all_features, dim=1, p=2)
        weight = torch.matmul(all_features, all_features.transpose(0, 1))
        weight[weight < 0] = 0
        values, indexes = torch.topk(weight, graphk)
        weight[weight < values[:, -1].view(-1, 1)] = 0
        weight = weight + weight.transpose(0, 1)
        weight.diagonal(0).fill_(0)  ## change the diagonal elements with inplace operation.
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, num_all)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(num_all, 1)
        S = D1 * weight * D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
        pred_all = torch.matmul(torch.inverse(torch.eye(num_all) - alpha * S + 1e-8), all_gt)
        del weight
        pred_unl = pred_all[num_labeled:, :]
        #### add a fix value
        min_value = torch.min(pred_unl, 1)[0]
        min_value[min_value > 0] = 0
        pred_unl = pred_unl - min_value.view(-1, 1)
        pred_unl = pred_unl / pred_unl.sum(1).view(-1, 1)
        soft_label_lp = pred_unl
        return soft_label_lp


