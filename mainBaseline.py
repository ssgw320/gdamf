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

import sys
import torch
import numpy as np
import pandas as pd
import util
import datasets2
import GradualDomainAdaptation as G

# settings
rep = 20
# key: load function, num inter
settings = {'mnist': (datasets2.load_RotatedMNIST2, 3),
            'portraits': (datasets2.load_Portraits, 2),  # past -> future, 3
            'gas': (datasets2.load_GasSensor, 1),
            'cover': (datasets2.load_CoverType, 3),
            'rotmoon': (datasets2.make_gradual_data, 1)}
# key: train function, params
methods = {'gst': (G.GradualSelfTrain, [None]),
           'dsaoda': (G.DSAODA, [0.1, 1, 10]),
           'gift': (G.GIFT, [10, 20, 30]),
           'aux': (G.AuxSelfTrain, [10, 20, 30]),
           'sourceonly': (G.SourceOnly, [None])}

if __name__ == '__main__':
    key, m = sys.argv[1], sys.argv[2]  # ex. python mainBaseline.py mnist dsaoda
    print(f'{key}_{m}')
    # load data
    load_f, num_inter = settings[key]
    x_all, y_all = load_f()
    x_eval, y_eval = x_all.pop(), y_all.pop()
    # query settings
    flag = True if key == 'cover' else False
    num_init_labels, cost, budgets = util.query_settings(x_all, num_inter, flag)
    # results, accuracy with 3 level hyper harameters, low -> mid -> high
    func, params = methods[m]
    res = np.full(shape=(len(params), rep), fill_value=np.nan)
    for r in range(rep):
        np.random.seed(r)
        torch.manual_seed(r)
        x_subset, y_subset = util.subset_domain(x_all, y_all, num_inter, r)
        for i, p in enumerate(params):
            if (m == 'gst') or (m == 'sourceonly'):
                all_model, _ = func(x_subset, y_subset)
            elif m == 'dsaoda':
                all_model, _ = func(x_subset, y_subset, num_init_labels, budgets[-1], cost, p)
            else:
                if (m == 'aux') & (key == 'gas'):
                    all_model, _ = func(x_subset, y_subset, p, drop_last=True)
                else:
                    all_model, _ = func(x_subset, y_subset, p)
            res[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'acc': res}, f'./result/{key}_{m}.pkl')

