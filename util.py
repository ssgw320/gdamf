#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import ot
#import umap
#from umap.parametric_umap import ParametricUMAP
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
from torch.utils.data import TensorDataset


def torch_to(*args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return [arg.to(device) for arg in args] if len(args) > 1 else args[0].to(device)


def preprocess_input(*args) -> TensorDataset:
    all_tensors = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            # this is label data
            if arg.ndim == 1:
                arg = torch.tensor(np.array(arg).astype(int))
            # this is feature data
            else:
                arg = torch.tensor(np.array(arg).astype(np.float32))
        all_tensors.append(arg)
    dataset = TensorDataset(*all_tensors)
    return dataset


def get_expand_range(series:pd.Series, ratio:int=10) -> list:
    """ use for plot """
    d_min, d_max = series.min(), series.max()
    upper = d_max + (d_max * ratio / 100) if d_max > 0 else d_max - (d_max * ratio / 100)
    lower = d_min - (d_min * ratio / 100) if d_min > 0 else d_min + (d_min * ratio / 100)
    return [lower, upper]


def plot_gradual(x_all:list, y_all:list=None):
    # make data frame
    df = []
    for i, x in enumerate(x_all):
        _df = pd.DataFrame(x, columns=['x1', 'x2'])
        _df['frame'] = i
        if y_all != None:
            _df['y'] = y_all[i].astype(str)
        df.append(_df)
    df = pd.concat(df)
    # plot
    color = 'y' if y_all != None else None
    fig = px.scatter(data_frame=df, x='x1', y='x2', animation_frame='frame', color=color,
                     range_x=get_expand_range(df['x1']), range_y=get_expand_range(df['x2']), width=600, height=600)
    return fig


def visualize_predict(model, x, y, mesh_points=50) -> go.Figure:
    """ 2d and 2-class data only """
    x1_min, x1_max = get_expand_range(x[:,0])
    x2_min, x2_max = get_expand_range(x[:,1])
    x1range = np.linspace(x1_min, x1_max, mesh_points)
    x2range = np.linspace(x2_min, x2_max, mesh_points)
    x1x1, x2x2 = np.meshgrid(x1range, x2range)
    # estimate prob for all mesh points
    mesh = np.c_[x1x1.ravel(), x2x2.ravel()]
    dataset = preprocess_input(mesh)
    model, dataset = torch_to(model, dataset.tensors[0])
    with torch.no_grad():
        logits = model(dataset)
        logits = torch.nn.functional.softmax(logits, dim=1)
        #z = np.array(logits)[:,1]
        z = logits.cpu().detach().numpy()[:,1]
    z = z.reshape(x1x1.shape)
    # plot
    yA, yB = np.unique(y)
    fig = go.Figure(data=[go.Scatter(x=x[y==yA, 0], y=x[y==yA, 1], mode='markers')])
    fig.add_scatter(x=x[y==yB, 0], y=x[y==yB, 1], mode='markers')
    fig.update_layout(width=600, height=500, xaxis_title='x1', yaxis_title='x2',
                      margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    fig.add_trace(go.Contour(x=x1range, y=x2range, z=z, showscale=False, colorscale=['blue', 'white', 'red'], opacity=0.3))
    return fig


def wasserstein_distance(xa: np.ndarray, xb: np.ndarray, norm: int=1) -> float:
    """ norm : 1 -> W1 wasserstein, 2 -> W2 wasserstein """
    metric = 'euclidean' if norm == 1 else 'sqeuclidean'
    # uniform distribution on samples
    a_size, b_size = xa.shape[0], xb.shape[0]
    # image data
    if xa.ndim == 4:
        xa, xb = xa.reshape(a_size, -1), xb.reshape(b_size, -1)
    a = np.ones(a_size) / a_size
    b = np.ones(b_size) / b_size
    # loss matrix
    M = ot.dist(xa, xb, metric=metric)
    dist = ot.emd2(a, b, M, numItermax=10000000)
    return dist


def mpcWD(x_all, y_all):
    """ maximum per class wasserstein distance """
    all_class = np.unique(y_all[0])
    mpc_wd = []
    for i, (x, y) in enumerate(zip(x_all, y_all)):
        if i+1 < len(x_all):
            x_next, y_next = x_all[i+1].copy(), y_all[i+1].copy()
            wds = []
            for c in all_class:
                _x = x[np.argwhere(y==c).flatten()].copy()
                _x_next = x_next[np.argwhere(y_next==c).flatten()].copy()
                wds.append(wasserstein_distance(_x, _x_next))
            mpc_wd.append(max(wds))
    return mpc_wd


# def fit_umap(x_all, y_all, **umap_kwargs) -> list:
#     umap_settings = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
#     umap_settings.update(umap_kwargs)
#     X = np.vstack(x_all)
#     X = X.reshape(X.shape[0], -1)
#     # use source label as semi-superviesd UMAP
#     Y_semi_supervised = [np.full(shape=y.shape[0], fill_value=-1) for y in y_all]
#     Y_semi_supervised[0] = y_all[0].copy()
#     Y_semi_supervised = np.hstack(Y_semi_supervised)
#     # fit UMAP
#     encoder = umap.UMAP(random_state=1234, **umap_settings)
#     Z = encoder.fit_transform(X, Y_semi_supervised)
#     z_idx = np.cumsum([i.shape[0] for i in x_all])
#     z_all = np.vsplit(Z, z_idx)[:-1]
#     return z_all, encoder


# def fit_parametric_umap(x_all, y_all, metric, n_neighbors):
#     X = np.vstack(x_all)
#     X = X.reshape(X.shape[0], -1)
#     # use source label as semi-superviesd UMAP
#     Y_semi_supervised = [np.full(shape=y.shape[0], fill_value=-1) for y in y_all]
#     Y_semi_supervised[0] = y_all[0].copy()
#     Y_semi_supervised = np.hstack(Y_semi_supervised)
#     # fit UMAP
#     embedder = ParametricUMAP(parametric_embedding=False, verbose=False,
#                               metric=metric, n_neighbors=n_neighbors,
#                               n_components=2, random_state=np.random.RandomState(1234))
#     Z = embedder.fit_transform(X, Y_semi_supervised)
#     z_idx = np.cumsum([i.shape[0] for i in x_all])
#     z_all = np.vsplit(Z, z_idx)[:-1]
#     return z_all, embedder._history['loss']


def plot_loss_history(path: str):
    lh = pd.read_pickle(path)
    now = np.sum(~np.isnan(lh[0,:]))
    total = lh[0,:].shape[0]
    fig = px.scatter(title=f'{now}/{total} epochs')
    for i in range(lh.shape[0]):
        fig.add_scatter(y=lh[i,:], mode='markers', name=f'time={i+1}')
    fig.update_layout(margin=dict(t=30, b=30), xaxis_title='epochs', yaxis_title='loss')
    fig.show()
    return lh


# def parse_umap_result(path: str):
#     umap_res = pd.read_pickle(path)
#     keys = list(umap_res.keys())
#     min_idx = np.argmin([np.min(umap_res[k][-1]) for k in keys])
#     opt_key = keys[min_idx]
#     print(opt_key)
#     z_all, y_all, _ = umap_res[opt_key]
#     return z_all, y_all


def rounded_statistics(array, ndigits=3):
    m, s = round(np.nanmean(array), ndigits), round(np.nanstd(array), ndigits)
    return '{}±{}'.format(m, s)


def subset_domain(x_all, y_all, num_inter: int, seed: int):
    """
    make a subset of given domains
    @param
    num_inter: int, control the number of intermediate domains
    seed: int, random seed, intermediate domains are selected at random
    """
    np.random.seed(seed)
    inter_index = np.arange(1, len(x_all)-1)
    idx = sorted(np.random.choice(inter_index, num_inter, replace=False).tolist())
    idx = [0] + idx + [len(x_all)-1]  # source + inter + target
    x_subset = [x_all[i].copy() for i in idx]
    y_subset = [y_all[i].copy() for i in idx]
    return x_subset, y_subset


def query_settings(x_all, num_inter: int, cover_flag=False):
    """ return num_init_labels, cost, budgets """
    num_init_labels = x_all[0].shape[0] // 1000 if cover_flag else x_all[0].shape[0] // 100
    cost = np.arange(num_inter+2)[1:]
    max_budget = x_all[0].shape[0] // 100 if cover_flag else x_all[0].shape[0] // 10
    budgets = np.linspace(max_budget//5, max_budget, 5, dtype=int)
    return num_init_labels, cost, budgets
