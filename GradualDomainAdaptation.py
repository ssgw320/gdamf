# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:.conda-torch-env]
#     language: python
#     name: conda-env-.conda-torch-env-py
# ---

import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# my module
from util import torch_to, preprocess_input


# +
class MLP(nn.Module):
    def __init__(self, num_labels, input_dim, hidden_dim):
        super(MLP, self).__init__()
        """ in the case of GIFT and two-moon data, nn.Linear is only one """
        """ in the case of Cover Type, hidden_dim=32 is recommend """
        self.num_labels = num_labels
        self.input_dim = input_dim
        # tabular
        if isinstance(input_dim, int):
            self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.BatchNorm1d(num_features=hidden_dim))
            self.pred = nn.Linear(hidden_dim, num_labels)
        # image, input_dim -> ex. (28, 28)
        else:
            num_conv = 3
            conv_dim = np.rint(np.array(input_dim) / 2**num_conv)
            latent_dim = int(conv_dim[0] * conv_dim[1] * hidden_dim)
            conv_settings = dict(kernel_size=5, stride=2, padding=2)
            self.fc = nn.Sequential(nn.Conv2d(1, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, **conv_settings), nn.ReLU(),
                                    nn.Dropout2d(p=0.5),
                                    nn.BatchNorm2d(num_features=hidden_dim),
                                    nn.Flatten())
            self.pred = nn.Linear(latent_dim, num_labels)

    def forward(self, x):
        feature = self.fc(x)
        pred_y = self.pred(feature)
        return pred_y


class AuxiliaryModel(MLP):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    """

    def get_prediction_with_uniform_prior(self, soft_prediction):
        soft_prediction_uniform = soft_prediction / soft_prediction.sum(0, keepdim=True).pow(0.5)
        soft_prediction_uniform /= soft_prediction_uniform.sum(1, keepdim=True)
        return soft_prediction_uniform

    def classifier_prediction(self, x_source):
        with torch.no_grad():
            pred_network = self.forward(x_source)
            pred_network = nn.functional.softmax(pred_network, dim=1)
        pred_network = self.get_prediction_with_uniform_prior(pred_network)
        pseudo_y = pred_network.argmax(dim=1)
        return pred_network, pseudo_y

    def ensemble_prediction(self, x_source, y_source, x_target):
        """ use only for self train """
        pred_network, _, = self.classifier_prediction(x_target)
        pred_kmeans = self.get_labels_from_kmeans(x_source, y_source, x_target)
        pred_lp = self.get_labels_from_lp(x_source, y_source, x_target)
        pred_kmeans = self.get_prediction_with_uniform_prior(pred_kmeans)
        pred_lp = self.get_prediction_with_uniform_prior(pred_lp)
        pred_ensemble = (pred_network + pred_kmeans + pred_lp) / 3
        pseudo_y = pred_ensemble.argmax(dim=1)
        return pred_ensemble, pseudo_y

    def get_labels_from_kmeans(self, x_source, y_source, x_target):
        with torch.no_grad():
            z_source = self.fc(x_source)
            z_target = self.fc(x_target)
        z_source_array, y_source_array, z_target_array = z_source.numpy(), y_source.numpy(), z_target.numpy()
        init = np.vstack([z_source_array[y_source_array==i].mean(axis=0) for i in np.unique(y_source_array)])
        kmeans = KMeans(n_clusters=self.num_labels, init=init, n_init=1, random_state=0).fit(z_target_array)
        centers = kmeans.cluster_centers_  # num_category * feature_dim
        centers_tensor = torch.from_numpy(centers)
        centers_tensor_unsq = torch.unsqueeze(centers_tensor, 0)
        target_u_feature_unsq = torch.unsqueeze(z_target, 1)
        L2_dis = ((target_u_feature_unsq - centers_tensor_unsq)**2).mean(2)
        soft_label_kmeans = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
        return soft_label_kmeans

    def get_labels_from_lp(self, x_source, y_source, x_target):
        """ label propagation """
        graphk = 20
        alpha = 0.75
        with torch.no_grad():
            labeled_features = self.fc(x_source)
            unlabeled_features = self.fc(x_target)
        labeled_onehot_gt = nn.functional.one_hot(y_source, num_classes=self.num_labels)

        num_labeled = labeled_features.size(0)
        if num_labeled > 100000:
            print('too many labeled data, randomly select a subset')
            indices = torch.randperm(num_labeled)[:10000]
            labeled_features = labeled_features[indices]
            labeled_onehot_gt = labeled_onehot_gt[indices]
            num_labeled = 10000

        num_unlabeled = unlabeled_features.size(0)
        num_all = num_unlabeled + num_labeled
        all_features = torch.cat((labeled_features, unlabeled_features), dim=0)
        unlabeled_zero_gt = torch.zeros(num_unlabeled, self.num_labels)
        all_gt = torch.cat((labeled_onehot_gt, unlabeled_zero_gt), dim=0)
        ### calculate the affinity matrix
        all_features = nn.functional.normalize(all_features, dim=1, p=2)
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


# +
def acquisition_function(model, x):
    dataset = preprocess_input(x)
    model, _x = torch_to(model, dataset.tensors[0])
    with torch.no_grad():
        pred = model(_x)
    pred = np.array(torch.Tensor.cpu(nn.functional.softmax(pred, dim=1)))
    unc = 1 - pred.max(axis=1)
    return unc


def multi_fidelity(x_all, models: list, budget: int, cost: list):
    """
    @memo
    Calculate the optimal number of queries with considering cost and correlation
    We will not query form Source
    @param
    cost: list, ex. [1, 5, 8, 10]
    """
    assert len(x_all)-1 == len(cost)
    input_dim = _check_input_dim(x_all[0])
    if isinstance(input_dim, tuple):
        h, w = input_dim
        size = (10000, 1, h, w)
    else:
        size = (10000, input_dim)
    X = np.vstack(x_all)
    z = np.random.uniform(low=np.min(X), high=np.max(X), size=size).astype(np.float32)
    # calc each f(Z)
    fz = []
    z = torch_to(torch.tensor(z))
    with torch.no_grad():
        for model in models:
            model = torch_to(model)
            out = model(z)
            out = np.array(torch.Tensor.cpu(out))
            fz.append(out)
    # calc corr f_i(Z) vs f_target(Z)
    corr = np.array([])
    for fi in fz[1:-1]:  # corr of source vs. target and target vs. target are not need
        corr_each_label = [np.corrcoef(fi[:,col], fz[-1][:,col])[0,1] for col in range(model.num_labels)]
        corr = np.append(corr, np.mean(corr_each_label))
    # calc r
    p1, c_hi = deepcopy(corr[-1]), deepcopy(cost[-1])
    rK = 1  # rK = 1 / c_hi
    r = np.array([])
    for i, c in enumerate(cost[:-1]):
        pi = corr[i]
        pii = 0 if i ==0 else corr[i-1]
        r = np.append(r, np.sqrt(abs((c_hi*(pi**2 - pii**2)) / (c*(1 - p1**2)))))
    r = np.append(r, rK)  # r -> [r1, ..., rk-1, rK]
    m_hi = budget / (r @ np.array(cost))
    opt_eval_num = np.rint(r * m_hi).astype(int)
    return opt_eval_num


def get_pseudo_y(model: nn.Module, x: torch.Tensor, confidence_q: float=0.1, GIFT: bool=False) -> (np.ndarray, np.ndarray):
    """ remove less confidence sample """
    dataset = preprocess_input(x)
    model, _x = torch_to(model, dataset.tensors[0])
    with torch.no_grad():
        logits = model(_x) if not GIFT else model.pred(_x)
        logits = nn.functional.softmax(logits, dim=1)
        confidence = np.array(torch.Tensor.cpu(logits.amax(dim=1) - logits.amin(dim=1)))
        alpha = np.quantile(confidence, confidence_q)
        conf_index = np.argwhere(confidence >= alpha)[:,0]
        pseudo_y = logits.argmax(dim=1)
    return pseudo_y.detach().cpu().numpy(), conf_index


def mask_true_label(y_all: list, num_keep_labels: int):
    """ y_all[0] = source """
    masked_y_all = [deepcopy(y_all[0])]
    for i, y in enumerate(y_all[1:]):
        size = y.size
        np.random.seed(1234)
        idx = np.random.choice(np.arange(size), size-num_keep_labels, replace=False)
        masked_y = deepcopy(y)
        masked_y[idx] = -1
        masked_y_all.append(masked_y)
    return masked_y_all


def calc_accuracy(model, x, y):
    dataset = preprocess_input(x)
    with torch.no_grad():
        model, _x = torch_to(model, dataset.tensors[0])
        pred = model(_x)
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.array(torch.Tensor.cpu(pred.argmax(dim=1)))
    return accuracy_score(y, pred.squeeze())


def _check_input_dim(x):
    if np.ndim(x) == 4:
        input_dim = x.shape[-2:]
    else:
        input_dim = x.shape[1]
    return input_dim


def train_classifier(clf, x, y, n_epochs=100, weight_decay=1e-3, GIFT=False, drop_last=False):
    model = deepcopy(clf)
    model = torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()
    batch_size = 1024
    dataset = preprocess_input(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    loss_history = []
    for e in range(n_epochs):
        running_loss = 0
        for x_sample, y_sample in train_loader:
            x_sample, y_sample = torch_to(x_sample, y_sample)
            optimizer.zero_grad()
            y_pred = model(x_sample) if not GIFT else model.pred(x_sample)
            loss = loss_f(y_pred, y_sample)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / batch_size
        loss_history.append(running_loss)
    return model, loss_history


def train_classifier_with_weight(clf, x1, y1, x2, y2, n_epochs=100, weight_decay=1e-3):
    """ use for semi-supervised learning """
    model = deepcopy(clf)
    model = torch_to(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()
    batch_size = 1024
    weight = y1.size / y2.size
    dataset1 = preprocess_input(x1, y1)
    dataset2 = preprocess_input(x2, y2)
    train_loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
    loss_history = []
    for e in range(n_epochs):
        running_loss = 0
        for (x_sample1, y_sample1), (x_sample2, y_sample2) in zip(train_loader1, train_loader2):
            x_sample1, y_sample1, x_sample2, y_sample2 = torch_to(x_sample1, y_sample1, x_sample2, y_sample2)
            optimizer.zero_grad()
            y_pred1 = model(x_sample1)
            y_pred2 = model(x_sample2)
            loss1 = loss_f(y_pred1, y_sample1)
            loss2 = loss_f(y_pred2, y_sample2)
            loss = loss1 + weight * loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / batch_size
        loss_history.append(running_loss)
    return model, loss_history


def gradual_train(source_model, x_all, y_all, ssl=False, ws=True, n_epochs=100, weight_decay=1e-3):
    """
    gradual learning
    @param
    y_all: list, it contains partially labels and lacked one described by -1
    ssl: bool, True -> semi-supervised, False -> supervised
    ws: bool/nn.Module, True -> warm start, nn.Module -> use input as initial model
    Note!! ssl=False & wa=True is not equal to TargetOnly
    @return
    all_model: list, models for each domain \theta^{(j)}
    """
    all_model = []
    student_model = deepcopy(source_model)
    teacher_model = deepcopy(source_model)
    all_model.append(student_model)  # source modle \theta^{(0)}
    for x, y in tqdm(zip(x_all[1:], y_all[1:]), total=len(x_all[1:]), disable=True):
        teacher_model.load_state_dict(student_model.state_dict())
        pseudo_y, conf_idx = get_pseudo_y(teacher_model, x)
        true_idx = np.argwhere(y != -1).flatten()
        if isinstance(ws, nn.Module):
            student_model = deepcopy(ws)
        if ssl:
            student_model, _ = train_classifier_with_weight(student_model, x[true_idx], y[true_idx],
                                                            x[conf_idx], pseudo_y[conf_idx], n_epochs, weight_decay)
        else:
            student_model, _ = train_classifier(student_model, x[true_idx], y[true_idx], n_epochs, weight_decay)
        all_model.append(student_model)
    return all_model


# +
def GDAMF(x_all, y_all, num_init_labels, budget, cost, AL=True, ssl=False, ws=True, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    @param
    num_init_labels: int, initial sample size
    budget: int, control the amount of query
    cost: list, ex [1, 2, 3, 4]
    AL: bool, True -> active learning, False -> random sampling
    ssl: bool, True -> semi-supervised, False -> supervised
    ws: bool, warm start True/False
    @return
    all_model: list, models for each domain
    masked_y_all: list, queried labels from each domain
    """
    # train source model
    initial_budget = deepcopy(budget)
    input_dim = _check_input_dim(x_all[0])
    source_model = MLP(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    initialize_model = deepcopy(source_model)
    source_model, _ = train_classifier(source_model, x_all[0], y_all[0], n_epochs, weight_decay)
    masked_y_all = mask_true_label(y_all, num_init_labels)
    queried_num = np.zeros_like(cost)
    while budget > 0:
        ws_input = True if ws else initialize_model
        all_model = gradual_train(source_model, x_all, masked_y_all, ssl, ws_input, n_epochs, weight_decay)
        opt_eval_num = multi_fidelity(x_all, all_model, initial_budget, cost)
        no_query_count = 0
        for i, c in enumerate(cost):  # len(x_all) -1 = len(cost)
            if (queried_num[i] < opt_eval_num[i]) & (budget > c):
                candidate = np.argwhere(masked_y_all[i+1] == -1).flatten()
                if AL:
                    unc = acquisition_function(all_model[i+1], x_all[i+1][candidate])
                    idx = candidate[unc.argmax()]
                else:
                    # random sampling
                    idx = np.random.choice(candidate, 1)[0]
                # query!!
                masked_y_all[i+1][idx] = deepcopy(y_all[i+1][idx])
                queried_num[i] += 1
                budget -= c
            else:
                no_query_count += 1
        print(f"\rRest of budget {budget}, opt eval num{opt_eval_num}", end="")
        if no_query_count == len(cost):
            print('\nno query!')
            return all_model, queried_num
    return all_model, queried_num


def directGDAMF(x_all, y_all, num_init_labels, budget, cost, AL=True, ssl=False, ws=True, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    @memo
    for ablation study
    apply domain adaptation without intermediate domains
    """
    # train source model
    input_dim = _check_input_dim(x_all[0])
    source_model = MLP(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    initialize_model = deepcopy(source_model)
    source_model, _ = train_classifier(source_model, x_all[0], y_all[0], n_epochs, weight_decay)
    # prepare for direct update
    x_subset, y_subset = [x_all[i].copy() for i in [0, -1]], [y_all[i].copy() for i in [0, -1]]
    masked_y_subset = mask_true_label(y_subset, num_init_labels)
    queried_num = 0
    while budget > 0:
        print(f"\rRest of budget {budget}", end="")
        ws_input = True if ws else initialize_model
        all_model = gradual_train(source_model, x_subset, masked_y_subset, ssl, ws_input, n_epochs, weight_decay)
        candidate = np.argwhere(masked_y_subset[-1] == -1).flatten()
        if AL:
            unc = acquisition_function(all_model[-1], x_subset[-1][candidate])
            idx = candidate[unc.argmax()]
        else:
            # random sampling
            idx = np.random.choice(candidate, 1)[0]
        if budget > cost[-1]:
            # query
            masked_y_subset[-1][idx] = deepcopy(y_subset[-1][idx])
            budget -= cost[-1]
            queried_num += 1
        else:
            print('\nno query!')
            return all_model, queried_num
    return all_model, queried_num


def GradualSelfTrain(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    @retrun
    student_model: nn.Module, target model \theta^{(K)}
    reults: dict, model and items used for training
    """
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    student_model = deepcopy(model)
    teacher_model = deepcopy(model)
    student_model, loss_history = train_classifier(student_model, x_all[0], y_all[0], n_epochs, weight_decay)
    all_model = [student_model]
    for j, x in enumerate(tqdm(x_all[1:])):
        teacher_model.load_state_dict(student_model.state_dict())
        pseudo_y, conf_index = get_pseudo_y(teacher_model, x)
        student_model, loss_history = train_classifier(student_model, x[conf_index], pseudo_y[conf_index], n_epochs, weight_decay)
        all_model.append(student_model)
    return all_model, None


def DSAODA(x_all, y_all, num_init_labels, budget, cost, b, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    Rai proposed
    Domain Adaptation meets Active Learning
    http://users.umiacs.umd.edu/~hal/docs/daume10daal.pdf
    We modify this metod for gradual domain adaptation
    @param
    num_init_labels: int, initial sample size
    budget: int, control the amount of query
    cost: list, ex [1, 2, 3, 4]
    b: int, it control the frequecy of sampling, Ex. r=0.5, b=5 -> p=0.9 
    """
    masked_y_all = mask_true_label(y_all, num_init_labels)
    all_model = []
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    model, loss_history = train_classifier(model, x_all[0], masked_y_all[0], n_epochs, weight_decay)
    all_model.append(model)
    queried_num = np.zeros_like(cost)
    for i, (x, y) in enumerate(zip(x_all, y_all)):

        if budget <= 0:
            print('\nno budget')
            return all_model, queried_num

        if i+1 < len(x_all):
            x_next, y_next, masked_y_next = x_all[i+1].copy(), y_all[i+1].copy(), masked_y_all[i+1].copy()
            # train domain separator
            x_sep = np.vstack([x, x_next])
            y_sep = np.array([0] * x.shape[0] + [1] * x_next.shape[0])  # 0 -> source, 1-> target
            input_dim = _check_input_dim(x_sep)
            separator = MLP(num_labels=2, input_dim=input_dim, hidden_dim=hidden_dim)
            separator, _ = train_classifier(separator, x_sep, y_sep, n_epochs, weight_decay)
            # show x one by one to the classifier and determine query or not query
            separator.eval()
            for idx, (_x, _y) in enumerate(zip(x_next, y_next)):
                model.eval()
                with torch.no_grad():
                    _x_input = torch_to(torch.tensor(np.expand_dims(_x, axis=0)).float())
                    r = separator(_x_input)
                    y_pred = model(_x_input)
                    y_pred = nn.functional.softmax(y_pred, dim=1)
                    y_pred = torch.Tensor.cpu(y_pred.argmax(dim=1))
                r = torch.Tensor.cpu(nn.functional.softmax(r, dim=1)[:,1])
                prob = b / (b + r)
                sample = np.random.binomial(n=1, p=prob, size=1)[0]
                # DO NOT query the index which has TRUE label.
                no_label = True if masked_y_next[idx] == -1 else False
                if (r.item() > 0.5) & (sample == 1) & (_y != y_pred.item()) & (budget > cost[i]) & (no_label):
                    # query
                    masked_y_next[idx] = _y
                    queried_num[i] += 1
                    budget -= cost[i]
                    # udate classifier
                    train_idx = np.argwhere(masked_y_next != -1).flatten()
                    model.train()
                    model, _ = train_classifier(model, x_next[train_idx], y_next[train_idx], n_epochs//2, weight_decay)
            print(f"Rest of budget {budget}")
            all_model.append(model)
    return all_model, queried_num


def AuxSelfTrain(x_all, y_all, num_inter, hidden_dim=32, n_epochs=100, weight_decay=1e-3, drop_last=False):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    @param
    num_inter: int, control the number of steps for adaptation
    """
    input_dim = _check_input_dim(x_all[0])
    model = AuxiliaryModel(num_labels=np.unique(y_all[0]).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)

    x_source, y_source, x_target = x_all[0].copy(), y_all[0].copy(), np.vstack(x_all[1:]).copy()
    num_source = x_source.shape[0]
    num_target = x_target.shape[0]
    num_labels = np.unique(y_source).size

    def get_index_each_label(num_labels: int, num_sample: int, pred_soft: torch.Tensor, pseudo_y: torch.Tensor):
        conf_index = []
        for l in range(num_labels):
            idx = np.arange(pseudo_y.numpy().shape[0])
            l_idx = idx[pseudo_y == l]
            l_idx_sorted = np.argsort(pred_soft.amax(dim=1)[l_idx].numpy())[::-1]
            top = num_sample // num_labels
            l_idx = l_idx[l_idx_sorted[:top]]
            conf_index.append(l_idx)
        return np.hstack(conf_index)

    model, _ = train_classifier(model, x_source, y_source, n_epochs, weight_decay)
    all_model = [model]
    for m in range(1, num_inter):
        top_s = int(((num_inter - m - 1) * num_source) / num_inter)
        top_t = int(((m + 1) * num_target) / num_inter)
        if m == 1:
            x_input, y_input = torch.tensor(x_source).float(), torch.tensor(y_source).long()
        else:
            x_input, y_input = torch.tensor(x_inter).float(), torch.tensor(y_inter).long()
        model = model.to(torch.device('cpu'))
        pred_s, pseudo_ys = model.classifier_prediction(x_input)
        pred_t, pseudo_yt = model.ensemble_prediction(x_input, y_input, torch.tensor(x_target).float())
        # select the data with high confidence
        conf_index_s = get_index_each_label(num_labels, top_s, pred_s, pseudo_ys)
        conf_index_t = get_index_each_label(num_labels, top_t, pred_t, pseudo_yt)
        if m == 1:
            x_inter = np.vstack([x_source[conf_index_s], x_target[conf_index_t]])
            y_inter = np.hstack([y_source[conf_index_s], pseudo_yt[conf_index_t]])
        else:
            x_inter = np.vstack([x_inter[conf_index_s], x_target[conf_index_t]])
            y_inter = np.hstack([y_inter[conf_index_s], pseudo_yt[conf_index_t]])
        print(f'top_s {top_s}, top_t {top_t}, x_inter size {x_inter.shape[0]}')
        model, _ = train_classifier(model, x_inter, y_inter, n_epochs=n_epochs, weight_decay=weight_decay, drop_last=drop_last)
        all_model.append(model)
    return all_model, None


def GIFT(x_all, y_all, iters, adapt_lmbda=3, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    """
    Abnar proposed
    Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent
    https://arxiv.org/abs/2106.06080
    @memo
    two-moon dataset example needs StandardScaler to each domain and 1 hidden layer, 32 nodes.
    @param
    iters: int, how many times lambda update
    adapt_lmbda: int, how many times update student model for synthesis data
    """
    # GIFT does not need intermediate dataset
    x_source, y_source = x_all[0].copy(), y_all[0].copy()
    x_target = x_all[-1].copy()
    input_dim = _check_input_dim(x_all[0])
    model = MLP(num_labels=np.unique(y_source).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)

    def align(ys, yt):
        index_s = np.arange(ys.shape[0])
        index_t = []
        for i in index_s:
            indices = np.arange(yt.size)
            indices = np.random.permutation(indices)
            index = np.argmax(ys[i] == yt[indices])
            index_t.append(indices[index])
        index_t = np.array(index_t)
        return index_s, index_t

    teacher_model, _ = train_classifier(model, x_source, y_source, n_epochs, weight_decay)
    all_model = [teacher_model]

    for i in tqdm(range(1, iters+1)):
        lmbda = (1.0 / iters) * i
        student_model = deepcopy(teacher_model)
        for j in range(adapt_lmbda):
            with torch.no_grad():
                zs = student_model.fc(torch_to(torch.tensor(x_source).float()))
                zt = teacher_model.fc(torch_to(torch.tensor(x_target).float()))
                pred_yt = teacher_model.pred(zt)
                pred_yt = torch.Tensor.cpu(pred_yt.argmax(dim=1)).numpy()
            index_s, index_t = align(y_source, pred_yt)
            zi = torch.vstack([(1.0 - lmbda) * zs[i] + lmbda * zt[j] for i,j in zip(index_s, index_t)])
            # update student model with pseudo label
            pseudo_y, conf_index = get_pseudo_y(teacher_model, zi, GIFT=True)
            student_model, _ = train_classifier(student_model, zi[conf_index], pseudo_y[conf_index], n_epochs, weight_decay, GIFT=True)
        teacher_model = deepcopy(student_model)
        all_model.append(teacher_model)
    return all_model, None


def SourceOnly(x_all, y_all, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    x, y = x_all[0].copy(), y_all[0].copy()
    input_dim = _check_input_dim(x)
    model = MLP(num_labels=np.unique(y).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    model, _ = train_classifier(model, x, y, n_epochs, weight_decay)
    return [model], None


def TargetOnly(x_all, y_all, num_init_labels, budget, cost, AL=False, hidden_dim=32, n_epochs=100, weight_decay=1e-3):
    x, y = x_all[-1].copy(), y_all[-1].copy()
    input_dim = _check_input_dim(x)
    model = MLP(num_labels=np.unique(y).shape[0], input_dim=input_dim, hidden_dim=hidden_dim)
    if AL:
        query_index = np.random.choice(np.arange(y.size), num_init_labels, replace=False)
        while budget > 0:
            print(f"\rRest of budget {budget}", end="")
            model, _ = train_classifier(model, x[query_index], y[query_index], n_epochs, weight_decay)
            candidate = np.delete(np.arange(y.size), query_index)
            unc = acquisition_function(model, x[candidate])
            idx = candidate[unc.argmax()]
            if budget > cost[-1]:
                query_index = np.append(query_index, idx)
                budget -= cost[-1]
            else:
                print('no query')
                return [model], query_index
    else:
        sample_size = num_init_labels + budget // cost[-1]
        query_index = np.random.choice(np.arange(y.size), sample_size, replace=False)
        model, _ = train_classifier(model, x[query_index], y[query_index], n_epochs, weight_decay)
    return [model], query_index