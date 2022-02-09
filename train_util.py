#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import models


# # model train util

# In[2]:


def torch_to(*args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return [arg.to(device) for arg in args] if len(args) > 1 else args[0].to(device)


def get_pseudo_y(model:nn.Sequential, x:torch.Tensor, confidence_q:float=0.1, GIFT=False) -> (np.ndarray, np.ndarray):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.array(x).astype(np.float32))
    with torch.no_grad():
        logits = model(x, 1)[1] if not GIFT else model.network[0].pred(x)
        logits = nn.functional.softmax(logits, dim=1)
        confidence = np.array(torch.Tensor.cpu(logits.amax(dim=1) - logits.amin(dim=1)))
        alpha = np.quantile(confidence, confidence_q)
        conf_index = np.argwhere(confidence >= alpha)[:,0]
        pseudo_y = logits.argmax(dim=1)
    return pseudo_y, conf_index


def _preprocess_input(x, y) -> TensorDataset:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.array(x).astype(np.float32))
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(np.array(y).astype(int))
    dataset = TensorDataset(x, y)
    return dataset


def calc_accuracy(model, x, y, domain:int):
    dataset = _preprocess_input(x, y)
    with torch.no_grad():
        _x, model = torch_to(dataset.tensors[0], model)
        _, pred = model(_x, domain=domain)
        if pred.dim() == 3:
            pred = pred.squeeze()
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.array(torch.Tensor.cpu(pred.argmax(dim=1)))
    return accuracy_score(y, pred.squeeze())


def acquisition_function(model, x, domain:int):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.array(x, dtype=np.float32))
        x, model = torch_to(x, model)
    with torch.no_grad():
        _, pred = model(x, domain=domain)
        if pred.dim() == 3:
            pred = pred.squeeze()
    pred = np.array(torch.Tensor.cpu(nn.functional.softmax(pred, dim=1)))
    unc = 1 - pred.max(axis=1)
    return unc


def calc_fidelity_eval_num(model:nn.Module, x_all:list, budget:int, cost:list, rK:float=1.0):
    """
    @memo
    Calculate the optimal number of queries by considering cost and correlation
    We will not query form Source
    @param
    cost : ex. input; [1, 5, 8, 10]
    """
    assert len(x_all)-1 == len(cost)
    if isinstance(model.input_dim, tuple):
        h, w = model.input_dim
        z = np.random.uniform(low=np.min(x_all), high=np.max(x_all), size=(10000, 1, h, w)).astype(np.float32)
    else:
        z = np.random.uniform(low=np.min(x_all), high=np.max(x_all), size=(10000, model.input_dim)).astype(np.float32)
    # calc each f(Z)
    fz = []
    z = torch_to(torch.tensor(z))
    with torch.no_grad():
        for i in range(len(x_all)):
            _, out = model(z, domain=i+1)
            out = np.array(torch.Tensor.cpu(out))
            fz.append(out)
    # calc corr f_i(Z) vs f_target(Z)
    corr = np.array([])
    for fi in fz[1:-1]:  # source vs target and target vs target corr does not need
        corr_each_label = [np.corrcoef(fi[:,col], fz[-1][:,col])[0,1] for col in range(model.num_labels)]
        corr = np.append(corr, np.mean(corr_each_label))
    # calc r
    p1, c_hi = deepcopy(corr[-1]), deepcopy(cost[-1])
    r = np.array([])
    for i, c in enumerate(cost[:-1]):
        pi = corr[i]
        pii = 0 if i ==0 else corr[i-1]
        r = np.append(r, np.sqrt(abs((c_hi*(pi**2 - pii**2)) / (c*(1 - p1**2)))))
    r = np.append(r, rK)  # r -> [r1, ..., rk-1, rK]
    m_hi = budget / (r @ np.array(cost))
    opt_eval_num = np.rint(r * m_hi).astype(int)
    return opt_eval_num


# # model train

# In[25]:


def trainGDAMF(model:nn.Module, x_all:list, y_all:list, n_epochs:int, weight_decay:float=0, tqdm_disable:bool=False, GIFT:bool=False):
    model = torch_to(model)
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()
    loss_history = []
    for e in tqdm(range(n_epochs), disable=tqdm_disable):
        running_loss = 0
        for d in range(len(x_all)):
            dataset = _preprocess_input(x_all[d], y_all[d])
            d_size = dataset.tensors[0].shape[0]
            batch_size = 128 if d_size > 1000 else d_size
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x_sample, y_sample in train_loader:
                x_sample, y_sample = torch_to(x_sample, y_sample)
                optimizer.zero_grad()
                if not GIFT:
                    feature, y_pred = model(x_sample, d+1)
                else:
                    y_pred = model.network[0].pred(x_sample)
                loss = loss_f(y_pred, y_sample)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        loss_history.append(running_loss)
    return model, loss_history


def train_oracle(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                 samples:list, n_epochs:int, num_repeats:int, weight_decay:float=0):
    accuracy = []
    target_x, target_y = x_all[-1].copy(), y_all[-1].copy()
    candidate = np.arange(target_y.size)
    for sample in tqdm(samples):
        loop_acc = []
        for i in range(num_repeats):
            index = np.random.choice(candidate, sample, replace=False)
            oracle = deepcopy(model)
            oracle, loss_history = trainGDAMF(oracle, [target_x[index,:]], [target_y[index]], n_epochs, weight_decay, True)
            acc = calc_accuracy(oracle, x_eval, y_eval, 1)
            loop_acc.append(acc)
        accuracy.append(loop_acc)
    return oracle, accuracy


def train_oracle_with_budget(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                             budgets:list, cost:list, n_epochs:int, num_repeats:int, weight_decay:float=0, tqdm_disable:bool=True):
    assert len(x_all)-1 == len(cost)
    all_accuracy, all_sampled_index = list(), list()
    target_x, target_y = x_all[-1].copy(), y_all[-1].copy()
    candidate = np.arange(target_y.size)
    for budget in tqdm(budgets, disable=tqdm_disable):
        loop_acuuracy, loop_sampled_index = list(), list()
        for rep in range(num_repeats):
            num_query = budget // cost[-1]
            sampled_index = np.random.choice(candidate, num_query, replace=False)
            oracle = deepcopy(model)
            oracle, loss_history = trainGDAMF(oracle, [target_x[sampled_index,:]], [target_y[sampled_index]], n_epochs, weight_decay, True)
            acc = calc_accuracy(oracle, x_eval, y_eval, 1)
            loop_acuuracy.append(acc)
            loop_sampled_index.append(sampled_index)
        all_accuracy.append(loop_acuuracy)
        all_sampled_index.append(loop_sampled_index)
    return oracle, all_accuracy, all_sampled_index


def train_random(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                 samples:list, n_epochs:int, num_repeats:int, weight_decay:float=0):
    all_accuracy, all_sampled_index = list(), list()
    steps = len(x_all)
    candidate = np.arange(y_all[0].size)
    for sample in tqdm(samples):
        loop_acc, loop_sampled_index = list(), list()
        for i in range(num_repeats):
            sampled_index = [candidate] + [np.random.choice(candidate, sample, replace=False) for i in range(steps-1)]
            x = [_x[idx] for _x, idx in zip(x_all, sampled_index)]
            y = [_y[idx] for _y, idx in zip(y_all, sampled_index)]
            gdamf = deepcopy(model)
            gdamf, loss_history = trainGDAMF(gdamf, x, y, n_epochs, weight_decay,True)
            acc = calc_accuracy(gdamf, x_eval, y_eval, steps)
            loop_acc.append(acc)
            loop_sampled_index.append(sampled_index)
        all_accuracy.append(loop_acc)
        all_sampled_index.append(loop_sampled_index)
    return gdamf, all_accuracy, all_sampled_index


def train_random_with_budget(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                             budgets:list, cost:list, n_epochs:int, num_repeats:int, weight_decay:float=0, tqdm_disable:bool=True):
    all_accuracy, all_sampled_index = list(), list()
    steps = len(x_all)
    assert steps-1 == len(cost)
    candidate = np.arange(y_all[0].size)
    for budget in tqdm(budgets, disable=tqdm_disable):
        loop_acuuracy, loop_sampled_index = list(), list()
        for rep in range(num_repeats):
            sample_domain = np.random.choice(range(steps-1), size=budget, replace=True)
            idx = np.cumsum([cost[i] for i in sample_domain]) <= budget
            _, num_sample_each_domain = np.unique(sample_domain[idx], return_counts=True)
            sampled_index = [candidate] + [np.random.choice(candidate, n, replace=False) for n in num_sample_each_domain]
            x = [_x[idx] for _x, idx in zip(x_all, sampled_index)]
            y = [_y[idx] for _y, idx in zip(y_all, sampled_index)]
            gdamf = deepcopy(model)
            gdamf, loss_history = trainGDAMF(gdamf, x, y, n_epochs, weight_decay, True)
            acc = calc_accuracy(gdamf, x_eval, y_eval, steps)
            loop_acuuracy.append(acc)
            loop_sampled_index.append(sampled_index)
        all_accuracy.append(loop_acuuracy)
        all_sampled_index.append(loop_sampled_index)
    return gdamf, all_accuracy, all_sampled_index


def train_al(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
             initial_samples:int, budgets:list, cost:list, n_epochs:int, num_repeats:int, weight_decay:float=0, rK:float=1.0):
    all_accuracy, all_sampled_index = list(), list()
    steps = len(x_all)
    candidate = np.arange(y_all[0].size)
    for budget in budgets:
        loop_accuracy, loop_sampled_index = list(), list()
        for i in range(num_repeats):
            gdamfAL = deepcopy(model)
            # initial sampling
            sampled_index = [candidate] + [np.random.choice(candidate, initial_samples, replace=False) for i in range(steps-1)]
            # Remove the budget for the initial sampled.
            _budget = budget - (np.array(cost) * initial_samples).sum()
            # initial training
            x = [_x[idx] for _x, idx in zip(x_all, sampled_index)]
            y = [_y[idx] for _y, idx in zip(y_all, sampled_index)]
            gdamfAL, loss_history = trainGDAMF(gdamfAL, x, y, n_epochs, weight_decay, True)
            # compute optimal number of queries from each pool dataset
            opt_sample_num = calc_fidelity_eval_num(gdamfAL, x_all, _budget, cost, rK)
            # active learning with mini-model
            for d, num in enumerate(opt_sample_num):
                domain = d + 1
                x, y = x_all[domain][sampled_index[domain]], y_all[domain][sampled_index[domain]]
                mini_model = models.GDAMF(num_labels=gdamfAL.num_labels, num_domains=1,
                                          input_dim=gdamfAL.input_dim, hidden_dim=gdamfAL.hidden_dim)
                mini_model, lh = trainGDAMF(mini_model, [x], [y], n_epochs, weight_decay, True)
                for i in range(num):
                    unc = acquisition_function(mini_model, x_all[domain], 1)
                    sampled_index[domain] = np.append(sampled_index[domain], unc.argmax())
                    x, y = x_all[domain][sampled_index[domain]], y_all[domain][sampled_index[domain]]
                    mini_model, lh = trainGDAMF(mini_model, [x], [y], n_epochs, weight_decay, True)
            # final training
            x = [_x[idx] for _x, idx in zip(x_all, sampled_index)]
            y = [_y[idx] for _y, idx in zip(y_all, sampled_index)]
            gdamfAL, loss_history = trainGDAMF(gdamfAL, x, y, n_epochs, weight_decay, True)
            # evaluation
            acc = calc_accuracy(gdamfAL, x_eval, y_eval, steps)
            loop_accuracy.append(acc)
            loop_sampled_index.append(sampled_index)
        all_accuracy.append(loop_accuracy)
        all_sampled_index.append(loop_sampled_index)
    return gdamfAL, all_accuracy, all_sampled_index


def GradualSelfTrain(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                     n_epochs:int, num_repeats:int, weight_decay:float=0, tqdm_disable:bool=True):
    """
    Kumar proposed
    Understanding Self-Training for Gradual Domain Adaptation
    https://arxiv.org/abs/2002.11361
    """
    accuracy = []
    for rep in tqdm(range(num_repeats), disable=tqdm_disable):
        student_model = deepcopy(model)
        teacher_model = deepcopy(model)
        student_model, loss_history = trainGDAMF(student_model, [x_all[0]], [y_all[0]], n_epochs, weight_decay, True)
        for i, (x, y) in enumerate(zip(x_all[1:], y_all[1:])):
            param = student_model.state_dict()
            teacher_model.load_state_dict(param)
            pseudo_y, conf_index = get_pseudo_y(teacher_model, x)
            student_model, loss_history = trainGDAMF(student_model, [x[conf_index]], [pseudo_y[conf_index]], n_epochs, weight_decay, True)
        acc = calc_accuracy(student_model, x_eval, y_eval, 1)
        accuracy.append(acc)
    return student_model, accuracy


def AuxSelfTrain(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                 num_inter:int, n_epochs:int, num_repeats:int, weight_decay:float=0):
    """
    Zhang proposed
    Gradual Domain Adaptation via Self-Training of Auxiliary Models
    https://arxiv.org/abs/2106.09890
    https://github.com/YBZh/AuxSelfTrain
    """
    x_source, y_source, x_target = x_all[0].copy(), y_all[0].copy(), np.vstack(x_all[1:]).copy()
    num_source = x_source.shape[0]
    num_target = x_target.shape[0]
    num_labels = np.unique(y_source).size

    def get_index_each_label(num_labels:int, num_sample:int, pred_soft:torch.Tensor, pseudo_y:torch.Tensor):
        conf_index = []
        for l in range(num_labels):
            idx = np.arange(pseudo_y.numpy().shape[0])
            l_idx = idx[pseudo_y == l]
            l_idx_sorted = np.argsort(pred_soft.amax(dim=1)[l_idx].numpy())[::-1]
            top = num_sample // num_labels
            l_idx = l_idx[l_idx_sorted[:top]]
            conf_index.append(l_idx)
        return np.hstack(conf_index)

    accuracy = []
    for rep in range(num_repeats):
        aux = deepcopy(model)
        aux, loss_history = trainGDAMF(aux, [x_source], [y_source], n_epochs, weight_decay, True)
        for m in range(1, num_inter):
            top_s = int(((num_inter - m - 1) * num_source) / num_inter)
            top_t = int(((m + 1) * num_target) / num_inter)
            x_input, y_input = (torch.tensor(x_source), torch.tensor(y_source)) if m==1 else (torch.tensor(x_inter), torch.tensor(y_inter))
            aux = aux.to(torch.device('cpu'))
            pred_s, pseudo_ys = aux.classifier_prediction(x_input)
            pred_t, pseudo_yt = aux.ensemble_prediction(x_input, y_input, torch.tensor(x_target))
            # select the one with the highest confidence level for each class label
            conf_index_s = get_index_each_label(num_labels, top_s, pred_s, pseudo_ys)
            conf_index_t = get_index_each_label(num_labels, top_t, pred_t, pseudo_yt)
            if m == 1:
                x_inter = np.vstack([x_source[conf_index_s], x_target[conf_index_t]])
                y_inter = np.hstack([y_source[conf_index_s], pseudo_yt[conf_index_t]])
            else:
                x_inter = np.vstack([x_inter[conf_index_s], x_target[conf_index_t]])
                y_inter = np.hstack([y_inter[conf_index_s], pseudo_yt[conf_index_t]])
            #print(f'top_s {top_s}, top_t {top_t}, x_inter shape {x_inter.shape[0]}')
            aux = deepcopy(model)
            aux, loss_history = trainGDAMF(aux, [x_inter], [y_inter], n_epochs, weight_decay, True)
        acc = calc_accuracy(aux, x_eval, y_eval, 1)
        accuracy.append(acc)
    return aux, accuracy


def GIFT(model:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
         iters:int, adapt_lmbda:int, n_epochs:int, num_repeats:int, weight_decay:float=0):
    """
    Abnar proposed
    Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent
    https://arxiv.org/abs/2106.06080
    @memo
    moon toy data example needs StandardScaler to each domain
    @param
    iters : how many times lambda update
    adapt_lmbda : how many times update student model for synthesis data
    """
    # GIFT does not need intermediate data
    x_source, y_source = x_all[0].copy(), y_all[0].copy()
    x_target = x_all[-1].copy()

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

    accuracy = []
    for rep in range(num_repeats):
        teacher_model = deepcopy(model)
        teacher_model, loss_history = trainGDAMF(teacher_model, [x_source], [y_source], n_epochs, weight_decay, True)
        for i in range(1, iters+1):
            lmbda = (1.0 / iters) * i
            student_model = deepcopy(teacher_model)
            for j in range(adapt_lmbda):
                with torch.no_grad():
                    zs, _ = student_model(torch_to(torch.tensor(x_source)), 1)
                    zt, pred_yt = teacher_model(torch_to(torch.tensor(x_target)), 1)
                    pred_yt = torch.Tensor.cpu(pred_yt.argmax(dim=1)).numpy()
                index_s, index_t = align(y_source, pred_yt)
                zi = torch.vstack([(1.0 - lmbda) * zs[i] + lmbda * zt[j] for i,j in zip(index_s, index_t)])
                pred_yi, conf_index = get_pseudo_y(teacher_model, zi, GIFT=True)
                student_model, loss_history = trainGDAMF(student_model, [zi[conf_index]], [pred_yi[conf_index]], n_epochs, weight_decay, True, True)
            teacher_model = deepcopy(student_model)
        acc = calc_accuracy(teacher_model, x_eval, y_eval, 1)
        accuracy.append(acc)
    return teacher_model, accuracy


def DSAODA(aoda:nn.Module, separator:nn.Module, x_source:torch.Tensor, y_source:torch.Tensor, x_target:torch.Tensor, y_target:torch.Tensor,
           b:int, r_budget:int, n_epochs:int, weight_decay:float=0, train_classifer:bool=True):
    """
    Rai proposed
    Domain Adaptation meets Active Learning
    http://users.umiacs.umd.edu/~hal/docs/daume10daal.pdf
    @param
    b : hyper param for sampling, Ex. r=0.5, b=5 -> p=0.9 
    r_budget : remained budget, number of query samples
    """
    # train domain separator
    x_sep = np.vstack([x_source, x_target])
    y_sep = np.array([0] * x_source.shape[0] + [1] * x_target.shape[0])  # 0 -> source, 1-> target
    separator, loss_history = trainGDAMF(separator, [x_sep], [y_sep], n_epochs, weight_decay, True)
    # train classifer
    if train_classifer:
        aoda, loss_history = trainGDAMF(aoda, [x_source], [y_source], n_epochs, weight_decay, True)
    # check all x_target as online algorithm
    with torch.no_grad():
        _, r = separator(torch_to(torch.tensor(x_target)), 1)
        _, y_pred = aoda(torch_to(torch.tensor(x_target)), 1)
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = np.array(torch.Tensor.cpu(y_pred.argmax(dim=1)))
    r = np.array(torch.Tensor.cpu(nn.functional.softmax(r, dim=1)[:,1]))
    prob = b / (b + r)
    sample = np.array([np.random.binomial(n=1, p=p, size=1)[0] for p in prob])
    query_index = np.where((r > 0.5) & (sample == 1) & (y_target != y_pred))[0]
    np.random.shuffle(query_index)
    query_index = query_index[:r_budget]
    if query_index.size == 0:
        return aoda, query_index
    else:
        # update model
        aoda, loss_history = trainGDAMF(aoda, [x_target[query_index]], [y_target[query_index]], n_epochs, weight_decay, True)
    return aoda, query_index


def GradualDSAODA(model:nn.Module, separator:nn.Module, x_all:list, y_all:list, x_eval:torch.Tensor, y_eval:torch.Tensor,
                  b:int, budgets:list, cost:list, n_epochs:int, num_repeats:int, weight_decay:float=0, tqdm_disable:bool=True):
    assert len(x_all)-1 == len(cost)
    all_accuracy, all_sampled_index = list(), list()
    for budget in tqdm(budgets, disable=tqdm_disable):
        loop_accuracy, loop_sampled_index = list(), list()
        for rep in range(num_repeats):
            dsaoda = deepcopy(model)
            loop_budget = deepcopy(budget)
            sampled_index = []
            for i, (x, y) in enumerate(zip(x_all, y_all)):
                train_classifer = True if i == 0 else False
                if i < len(x_all) - 1:
                    sep = deepcopy(separator)
                    r_budget = loop_budget // cost[i]
                    dsaoda, query_index = DSAODA(dsaoda, sep, x, y, x_all[i+1], y_all[i+1], b, r_budget, n_epochs, weight_decay, train_classifer)
                    sampled_index.append(query_index)
                    loop_budget -= cost[i] * query_index.shape[0]
                    if loop_budget <= 0:
                        break
            loop_accuracy.append(calc_accuracy(dsaoda, x_eval, y_eval, 1))
            loop_sampled_index.append(sampled_index)
        all_accuracy.append(loop_accuracy)
        all_sampled_index.append(loop_sampled_index)
    return dsaoda, all_accuracy, all_sampled_index

