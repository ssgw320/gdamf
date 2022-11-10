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
# key: load_function, num inter
settings = {'mnist': (datasets2.load_RotatedMNIST2, 3),
            'portraits': (datasets2.load_Portraits, 2),
            'gas': (datasets2.load_GasSensor, 1),
            'cover': (datasets2.load_CoverType, 3)}

if __name__ == '__main__':
    key = sys.argv[1]
    print(key)
    # load data
    load_f, num_inter = settings[key]
    x_all, y_all = load_f()
    x_eval, y_eval = x_all.pop(), y_all.pop()
    # query settings
    flag = True if key == 'cover' else False
    num_init_labels, cost, budgets = util.query_settings(x_all, num_inter, flag)
    # results, accuracy and queried sample size
    gdamf = np.full(shape=(len(budgets), rep), fill_value=np.nan)  # prposed method
    gdamf_rnd = np.full_like(gdamf, fill_value=np.nan)  # ablation study1, withour AL
    gdamf_direct = np.full_like(gdamf, fill_value=np.nan)  # ablation study2, without intermediate
    gdamf_abl = np.full_like(gdamf, fill_value=np.nan)  # ablation study3, without AL, intermediate
    gdamf_ws = np.full_like(gdamf, fill_value=np.nan)  # ablation study 4, without warm start
    to = np.full_like(gdamf, fill_value=np.nan)  # ref, target only
    to_al = np.full_like(gdamf, fill_value=np.nan)  # ref2, target only with AL
    query = np.full(shape=(len(budgets), rep, num_inter+1), fill_value=np.nan)
    query_ws = np.full_like(query, fill_value=np.nan)  # ablation study 4, without warm start
    for i, budget in enumerate(budgets):
        for r in range(rep):
            np.random.seed(r)
            torch.manual_seed(r)
            x_subset, y_subset = util.subset_domain(x_all, y_all, num_inter, r)
            # proposed
            all_model, queried_num = G.GDAMF(x_subset, y_subset, num_init_labels, budget, cost, AL=True, ssl=False)
            gdamf[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': gdamf}, f'./result/{key}_gdamf.pkl')
            query[i, r, :] = queried_num
            pd.to_pickle({'budgets': budgets, 'query': query}, f'./result/query_{key}.pkl')
            # ablation study1, random sampling
            all_model, _ = G.GDAMF(x_subset, y_subset, num_init_labels, budget, cost, AL=False, ssl=False)
            gdamf_rnd[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': gdamf_rnd}, f'./result/{key}_gdamf-rnd.pkl')
            # ablation study2, without intermediate
            all_model, _ = G.directGDAMF(x_subset, y_subset, num_init_labels, budget, cost, AL=True, ssl=False)
            gdamf_direct[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': gdamf_direct}, f'./result/{key}_gdamf-direct.pkl')
            # ablation study3, without AL, intermediate
            all_model, _ = G.directGDAMF(x_subset, y_subset, num_init_labels, budget, cost, AL=False, ssl=False)
            gdamf_abl[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': gdamf_abl}, f'./result/{key}_gdamf-abl.pkl')
            # ablation study 4, without warm start
            all_model, queried_num = G.GDAMF(x_subset, y_subset, num_init_labels, budget, cost, AL=True, ssl=False, ws=False)
            gdamf_ws[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': gdamf_ws}, f'./result/{key}_gdamf-ws.pkl')
            query_ws[i, r, :] = queried_num
            pd.to_pickle({'budgets': budgets, 'query': query_ws}, f'./result/query-ws_{key}.pkl')
            # target only
            all_model, _ = G.TargetOnly(x_subset, y_subset, num_init_labels, budget, cost)
            to[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': to}, f'./result/{key}_targetonly.pkl')
            # target only with AL
            all_model, _ = G.TargetOnly(x_subset, y_subset, num_init_labels, budget, cost, AL=True)
            to_al[i, r] = G.calc_accuracy(all_model[-1], x_eval, y_eval)
            pd.to_pickle({'budgets': budgets, 'acc': to_al}, f'./result/{key}_targetonly-al.pkl')
