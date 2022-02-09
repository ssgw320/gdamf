#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
from pathlib import Path
# my module
import models
import train_util
import util
import datasets


def arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reslut_dir', default='./result/')
    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--num_repeats', default=1)
    return parser.parse_args()


def NumSampleExperiment():
    """
    4.3 Effect of Gradual Domain Adaptation
    """
    global settings, n_epochs, n_epochs_img, weight_decay, num_repeats
    samples = np.arange(10, 510, 10)
    for key in ['moon', 'gas']:
        x_all, y_all = settings[key]['f'](**settings[key]['data'])
        x_eval, y_eval = x_all.pop(-1), y_all.pop(-1)
        # oracle
        oracle = models.GDAMF(num_domains=1, **settings[key]['nn'])
        res1 = train_util.train_oracle(oracle, x_all, y_all, x_eval, y_eval,
                                       samples, n_epochs, num_repeats, weight_decay)
        util.save_result(res1, samples, 'sampling', key, 'oracle')
        # GDAMF+Random
        gdamf = models.GDAMF(num_domains=len(x_all), **settings[key]['nn'])
        res2 = train_util.train_random(gdamf, x_all, y_all, x_eval, y_eval,
                                       samples, n_epochs, num_repeats, weight_decay)
        util.save_result(res2, samples, 'sampling', key, 'gdamf+random')


def NumInterDomainExperiment():
    """
    4.4 Self Training vs. Query Label
    In this Experiment only, we set n_epochs = 100, because large number of intermediate domain
    """
    global settings, n_epochs, weight_decay, num_repeats
    x_all, y_all = settings['mnist']['f'](**settings['mnist']['data'])
    x_eval, y_eval = x_all.pop(-1), y_all.pop(-1)

    num_inter_domain = np.arange(settings['mnist']['data']['num_inter_domain']) + 1
    num_select = np.arange(settings['mnist']['data']['num_inter_domain']+1)
    samples = [200]
    all_random_accuracy, all_gst_accuracy = list(), list()
    for d in num_select:
        print(f'num of domain {d}')
        loop_rand_acc, loop_gst_acc = list(), list()
        for i in range(num_repeats):
            # interの選択
            idx = np.random.choice(num_inter_domain, d, replace=False)
            idx.sort()
            # sourceとtargetを追加
            idx = np.insert(idx, 0, 0)
            idx = np.append(idx, len(x_all)-1)
            x_subset = np.array(x_all)[idx]
            y_subset = np.array(y_all)[idx]
            # GDAMF
            gdamf = models.GDAMF(num_domains=len(x_subset), **settings['mnist']['nn'])
            gdamf, random_accuracy, _ = train_util.train_random(gdamf, x_subset, y_subset, x_eval, y_eval, samples, n_epochs*2, 1, weight_decay)
            loop_rand_acc.append(random_accuracy)
            # Gradaul Self Train
            gst = models.GDAMF(num_domains=1, **settings['mnist']['nn'])
            gst, gst_accuracy = train_util.GradualSelfTrain(gst, x_subset, y_subset, x_eval, y_eval, n_epochs*2, 1, weight_decay)
            loop_gst_acc.append(gst_accuracy)
        all_random_accuracy.append(loop_rand_acc)
        all_gst_accuracy.append(loop_gst_acc)
    res1 = [gdamf, np.array(all_random_accuracy).squeeze()]
    res2 = [gst, np.array(all_gst_accuracy).squeeze()]
    util.save_result(res1, num_select, 'domain', 'mnist', 'gdamf+random')
    util.save_result(res2, num_select, 'domain', 'mnist', 'gst')


def _prepareExperiment(key:str):
    """
    :return: x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples
    """
    global settings
    x_all, y_all = settings[key]['f'](**settings[key]['data'])
    x_eval, y_eval = x_all.pop(-1), y_all.pop(-1)
    budgets = settings[key]['cost']['budgets']
    initial_samples = settings[key]['cost']['initial_samples']
    idx = settings[key]['cost']['idx']
    x_all, y_all = np.array(x_all)[idx], np.array(y_all)[idx]
    #cost = np.arange(len(x_all))[1:]
    cost = np.array(settings[key]['cost']['cost'])
    return x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples


def CostExperiment(ExpTarget:list):
    """
    4.6 Comparison with base line method
    """
    global settings, n_epochs, weight_decay, num_repeats, rK

    exp_type = 'cost'

    for key in ExpTarget:
        x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples = _prepareExperiment(key)
        # Source Only(so)
        print(f'Source Only {key}')
        accuracy = list()
        for n in range(num_repeats):
            model = models.GDAMF(num_domains=1, **settings[key]['nn'])
            model, _ = train_util.trainGDAMF(model, [x_all[0]], [y_all[0]], n_epochs, weight_decay, True)
            acc = train_util.calc_accuracy(model, x_eval, y_eval, 1)
            accuracy.append(acc)
        util.save_result([model, accuracy], budgets, exp_type, key, 'so')

        # Oracle
        print(f'Oracle {key}')
        model = models.GDAMF(num_domains=1, **settings[key]['nn'])
        res = train_util.train_oracle_with_budget(model, x_all, y_all, x_eval, y_eval, budgets, cost, n_epochs, num_repeats, weight_decay)
        util.save_result(res, budgets, exp_type, key, 'oracle')

        # GDAMF+Random
        print(f'GDAMF+Random {key}')
        model = models.GDAMF(num_domains=len(x_all), **settings[key]['nn'])
        res = train_util.train_random_with_budget(model, x_all, y_all, x_eval, y_eval, budgets, cost, n_epochs, num_repeats, weight_decay)
        util.save_result(res, budgets, exp_type, key, 'gdamf+random')

        # GDAMF+AL
        print(f'GDAMF+AL {key}')
        model = models.GDAMF(num_domains=len(x_all), **settings[key]['nn'])
        res = train_util.train_al(model, x_all, y_all, x_eval, y_eval, initial_samples, budgets, cost, n_epochs, num_repeats, weight_decay, rK)
        util.save_result(res, budgets, exp_type, key, 'gdamf+al')

        # GradualSelfTrain
        print(f'GradualSelfTrain {key}')
        model = models.GDAMF(num_domains=1, **settings[key]['nn'])
        res = train_util.GradualSelfTrain(model, x_all, y_all, x_eval, y_eval, n_epochs, num_repeats, weight_decay)
        util.save_result(res, budgets, exp_type, key, 'gst')


def CostExperimentDS(ExpTarget:list):
    # DSAODA
    global settings, n_epochs, weight_decay, num_repeats
    for key in ExpTarget:
        print(f'DSOADA {key}')
        x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples = _prepareExperiment(key)

        for b, name in zip([0.1, 1, 10], ['low', 'mid', 'high']):
            model = models.GDAMF(num_domains=len(x_all), **settings[key]['nn'])
            separator = models.GDAMF(num_labels=2, num_domains=1,
                                     input_dim=settings[key]['nn']['input_dim'], hidden_dim=settings[key]['nn']['hidden_dim'])
            res = train_util.GradualDSAODA(model, separator, x_all, y_all, x_eval, y_eval,
                                           b, budgets, cost, n_epochs, num_repeats, weight_decay)
            util.save_result(res, budgets, 'cost', key, f'dsaoda-{name}')


def CostExperimentGIFT(ExpTarget:list):
    # GIFT
    global settings, n_epochs, weight_decay, num_repeats
    for key in ExpTarget:
        print(f'GIFT {key}')
        x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples = _prepareExperiment(key)

        for iters, name in zip([20, 40, 60], ['low', 'mid', 'high']):
            adapt_lmbda = 3
            model = models.GDAMF(num_domains=1, **settings[key]['nn'])
            res = train_util.GIFT(model, x_all, y_all, x_eval, y_eval, iters, adapt_lmbda, n_epochs, num_repeats, weight_decay)
            util.save_result(res, budgets, 'cost', key, f'gift-{name}')


def CostExperimentAux(ExpTarget:list):
    # Aux
    global settings, n_epochs, weight_decay, num_repeats
    for key in ExpTarget:
        print(f'Aux {key}')
        x_all, y_all, x_eval, y_eval, budgets, cost, initial_samples = _prepareExperiment(key)

        for num_iter, name in zip([10, 14, 18], ['low', 'mid', 'high']):   # param change 5, 15, 30 -> 10, 14, 18
            model = models.AuxiliaryModel(num_labels=settings[key]['nn']['num_labels'],
                                          input_dim=settings[key]['nn']['input_dim'],
                                          hidden_dim=settings[key]['nn']['hidden_dim'])
            res = train_util.AuxSelfTrain(model, x_all, y_all, x_eval, y_eval, num_iter, n_epochs, num_repeats, weight_decay)
            util.save_result(res, budgets, 'cost', key, f'aux-{name}')



# data settings
settings = {'moon': {'f': datasets.make_gradual_data,
                     'data': dict(steps=5, n_samples=2000, start=0, end=90, mode='moon'),
                     'nn': dict(num_labels=2, input_dim=2, hidden_dim=32),},

            'mnist': {'f': datasets.load_RotatedMNIST,
                      'data': dict(num_inter_domain=21,
                                   rot_kwargs={'source': [2000, 0, 5, False], 'inter': [42000, 5, 55, True],
                                               'target': [2000, 55, 60, False], 'eval': [2000, 55, 60, False]}),
                      'nn': dict(num_labels=10, input_dim=(28, 28), hidden_dim=32),
                      'cost': dict(idx=[0, 5, 10, 15, 22], initial_samples=20, budgets=np.arange(1280, 3300, 300),
                                   cost=[1, 4, 9, 50])},

            'portraits': {'f': datasets.load_Portraits,
                          'data': dict(num_inter_domain=14,
                                       num_domain_samples={'source': 1000, 'inter': 14000, 'target': 1000, 'eval': 1000}),
                          'nn': dict(num_labels=2, input_dim=(32, 32), hidden_dim=32),
                          'cost':dict(idx=[0, 4, 8, 12, 15], initial_samples=20, budgets=np.arange(1280, 3300, 300),
                                      cost=[1, 4, 9, 50])},

            'cover': {'f': datasets.load_CoverType,
                      'data': dict(num_inter_domain=30,
                                   num_domain_samples={'source': 10000, 'inter': 300000, 'target': 10000, 'eval': 10000}),
                      'nn': dict(num_labels=2, input_dim=52, hidden_dim=32),
                      'cost':dict(idx=[0, 10, 20, 31], initial_samples=20, budgets=np.arange(1280, 3300, 300),
                                  cost=[1, 4, 50])},

            'gas': {'f': datasets.load_GasSensor,
                    'data': dict(num_inter_domain=7, num_domain_samples={'source': 1000, 'inter': 7000, 'target': 1000, 'eval': 1000}),
                    'nn': dict(num_labels=5, input_dim=128, hidden_dim=32),
                    'cost': dict(idx=[0, 2, 4, 6, 8], initial_samples=20, budgets=np.arange(1280, 3300, 300),
                                 cost=[1, 4, 9, 50])},}


if __name__ == '__main__':

    args = arg_parser()
    ExpTarget = ['mnist', 'portraits', 'gas', 'cover']
    rK = 0.1
    n_epochs = 50
    weight_decay = 1e-3
    num_repeats = int(args.num_repeats)
    datasets.data_dir = Path(args.data_dir)
    util.result_dir = Path(args.reslut_dir)
#     print(f'{num_repeats}')
#     print(f'{datasets.data_dir.absolute()}')
#     print(f'{util.result_dir.absolute()}')

    NumSampleExperiment()
    NumInterDomainExperiment()
    CostExperiment(ExpTarget)
    CostExperimentDS(ExpTarget)
    CostExperimentGIFT(ExpTarget)
    CostExperimentAux(ExpTarget)

    util.plot_NumSampleExperiment()
    util.plot_NumInterDomainExperiment()
    util.plot_NumQuerySample()
#     util.plot_CompareBudget()
    util.plot_CostExperiment(num_repeats)

