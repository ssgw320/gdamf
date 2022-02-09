#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import torch

result_dir = Path('/home/')

data_label = ("Rotating MNIST", "Portraits", "Cover Type", "Gas Sensor")

plotly_rgb = [list(px.colors.hex_to_rgb(_hex)) for _hex in px.colors.qualitative.Plotly]

# label, color
plot_dict = {'gdamf+random': ('GDAMF+Random', plotly_rgb[0], 'lines+markers', 'solid'),
             'gdamf+al': ('GDAMF+AL', plotly_rgb[1], 'lines+markers', 'solid'),
             'so': ('SourceOnly', plotly_rgb[2], 'lines', 'dash'),
             'oracle': ('TargetOnly', plotly_rgb[3], 'lines+markers', 'solid'),
             'dsaoda': ('DS-AODA', plotly_rgb[4], 'lines+markers', 'solid'),
             'gst': ('GradualSelfTrain', plotly_rgb[5], 'lines', 'dash'),
             'gift': ('GIFT', plotly_rgb[6], 'lines', 'dash'),
             'aux': ('AuxSelfTrain', plotly_rgb[7], 'lines', 'dash')}
# row, col
pos_dict = {'mnist': (1, 1),
            'portraits': (1, 2),
            'cover': (1, 3),
            'gas': (1, 4)}


# In[2]:


def get_expand_range(series:pd.Series, ratio:int=10) -> list:
    d_min, d_max = series.min(), series.max()
    upper = d_max + (d_max * ratio / 100) if d_max > 0 else d_max - (d_max * ratio / 100)
    lower = d_min - (d_min * ratio / 100) if d_min > 0 else d_min + (d_min * ratio / 100)
    return [lower, upper]


def plot_gradual(x_all:list, y_all:list) -> go.Figure:
    df = pd.DataFrame(np.vstack(x_all), columns=['x1', 'x2'])
    df['y'] = np.hstack(y_all).astype(str)
    # As gradual shift setting, the size of dataset at each step is the same.
    domain_size = y_all[0].size
    partition = df.shape[0] // domain_size
    df['domain'] = ['source'] * domain_size + ['inter'] * domain_size * (partition - 2) + ['target'] * domain_size
    # add 'frame' for plotly animation
    split_partition = np.array_split(np.arange(df.shape[0]), partition)
    for i, idx in enumerate(split_partition):
        df.loc[idx, 'frame'] = int(i)
    # plot
    x, y = 'x1', 'x2'
    fig = px.scatter(data_frame=df, x=x, y=y, color='y', animation_frame='frame',
                     range_x=get_expand_range(df[x]), range_y=get_expand_range(df[y]), width=600, height=600)
    return fig


def visualize_predict(x:np.ndarray, y:np.ndarray, model:torch.nn.Sequential, domain:int, mesh_points:int=50) -> go.Figure:
    """
    x : numpy 2d array
    y : 2-class only
    """
    x1_min, x1_max = get_expand_range(x[:,0])
    x2_min, x2_max = get_expand_range(x[:,1])
    x1range = np.linspace(x1_min, x1_max, mesh_points)
    x2range = np.linspace(x2_min, x2_max, mesh_points)
    x1x1, x2x2 = np.meshgrid(x1range, x2range)
    mesh = np.c_[x1x1.ravel(), x2x2.ravel()]
    with torch.no_grad():
        _, logits = model(torch.tensor(mesh.astype(np.float32)), domain)
        logits = torch.nn.functional.softmax(logits, dim=1)
        z = np.array(logits)[:,1]
    z = z.reshape(x1x1.shape)
    # plot
    yA, yB = np.unique(y)
    fig = go.Figure(data=[go.Scatter(x=x[y==yA, 0], y=x[y==yA, 1], mode='markers')])  # Type=Aのプロットを追加
    fig.add_scatter(x=x[y==yB, 0], y=x[y==yB, 1], mode='markers')  # Type=Bのプロットを追加
    fig.update_layout(width=600, height=500, xaxis_title='x1', yaxis_title='x2', margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    fig.add_trace(go.Contour(x=x1range, y=x2range, z=z, showscale=False, colorscale=['blue', 'white', 'red'], opacity=0.3))
    return fig


def add_var_scatter_plot(fig:go.Figure, x:np.ndarray, y:np.ndarray, color:list, name:str, 
                         dash:str='solid', row:int=1, col:int=1, showlegend:bool=True, y_std=None, mode='lines+markers'):
    """
    @memo
    about plotly color
    px.colors.qualitative.swatches()
    px.colors.qualitative.Plotly
    px.colors.hex_to_rgb('#636EFA')
    dash args -> "solid", "dot", "dash", "longdash", "dashdot", "longdashdot"
    """
    rgb = 'rgb' + str(tuple(color))
    rgba = 'rgba' + str(tuple(color+[0.3]))
    if y_std is None:
        mean = np.mean(y, axis=1)
        std = np.std(y, axis=1)
    else:
        mean = y
        std = y_std
    fig.add_scatter(x=x, y=mean, mode=mode, name=name, showlegend=showlegend, line=dict(color=rgb, dash=dash), row=row, col=col)
    fig.add_scatter(x=x, y=mean+std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='none', row=row, col=col)
    fig.add_scatter(x=x, y=mean-std, mode='lines', fill="tonexty", line=dict(width=0), showlegend=False, hoverinfo='none',
                    fillcolor=rgba, row=row, col=col)
    return fig


def save_result(res:list, x:list, exp_type:str, data_type:str, model_type:str):
    """
    @param
    res : contain -> model, accuracy, (sampled_index)
    exp_type : ['sampling', 'domain', 'cost']
    data_type : ['moon', 'mnist', 'gas', 'portraits', 'cover']
    model_type : ['so', 'oracle', 'gdamf+random', 'gdamf+al', 'gst', 'aux', 'gift', 'dsaoda']
    """
    if not result_dir.exists():
        result_dir.mkdir()
    if len(res) == 2:
        obj = dict(x=x, acc=res[1])
    else:
        obj = dict(x=x, acc=res[1], sampled_index=res[2])
    file_path = result_dir / f'{exp_type}_{data_type}_{model_type}.pkl'
    pd.to_pickle(obj, file_path)


def plot_NumSampleExperiment():
    global plotly_rgb
    data = {p.stem: pd.read_pickle(p) for p in result_dir.glob('sampling*')}
    # plot
    fig = go.Figure().set_subplots(1, 2, horizontal_spacing=0.05, subplot_titles=("Rotating Moon", "Gas Sensor"))  # shared_yaxes=True, 
    for key in data.keys():
        col, showlegend = (1, True) if 'moon' in key else (2, False)
        label, color, dash = ('TargetOnly', plotly_rgb[3], 'dot') if 'oracle' in key else ('GDAMF+Random', plotly_rgb[0], 'solid')
        add_var_scatter_plot(fig, data[key]['x'], data[key]['acc'], color, label, dash, 1, col, showlegend)
    # update layout
    fig.update_layout(yaxis_title='Accuracy',
                      #yaxis1_range=[0.4, 1.05], yaxis2_range=[0.4, 1.05],
                      xaxis1_title='Number of Samples from Each Domain',
                      xaxis2_title='Number of Samples from Each Domain',
                      width=900, height=350,
                      margin=dict(t=30, b=10, r=30, l=50),
                      font=dict(family="PTSerif", size=14,),
                      legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.5, xanchor="center", x=0.5))
    # output
    file_path = result_dir / 'NumSampleExperiment.pdf'
    fig.write_image(file_path)
    return fig


def plot_NumInterDomainExperiment():
    global plotly_rgb
    data = {p.stem: pd.read_pickle(p) for p in result_dir.glob('domain*')}
    # plot
    fig = go.Figure().set_subplots(1, 1)
    for key in data.keys():
        label, color = ('GDAMF+Random', plotly_rgb[0]) if 'gdamf' in key else ('GradualSelfTrain', plotly_rgb[5])
        add_var_scatter_plot(fig, data[key]['x'], data[key]['acc'], color, label)
    # update layout
    fig.update_layout(yaxis_title='Accuracy',
                      yaxis1_range=[0.3, 1.0],
                      xaxis1_title='Number of Intermediate Domain',
                      width=500, height=420,
                      margin=dict(t=30, b=10, r=30, l=50),
                      font=dict(family="PTSerif", size=14,),
                      legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.35, xanchor="center", x=0.5))
    # output
    file_path = result_dir / 'NumInterDomainExperiment.pdf'
    fig.write_image(file_path)
    return fig


def plot_NumQuerySample():
    global plotly_rgb, plot_dict, pos_dict

    fig = go.Figure().set_subplots(1, 4, vertical_spacing=0.2, horizontal_spacing=0.01,
                                   subplot_titles=data_label, shared_xaxes=True, shared_yaxes=True)

    showlegend = True
    target = ['gdamf+random', 'gdamf+al']
    for d_type in pos_dict.keys():
        # read data
        d_path = list(result_dir.glob(f'cost_{d_type}*'))
        data = {p.stem.split('_')[-1]: pd.read_pickle(p) for p in d_path if p.stem.split('_')[-1] in target}
        # count sampled number
        for key in target:
            sampled_index = np.asarray(data[key]['sampled_index'], dtype=np.object0)
            len_x, num_loop, num_domain = sampled_index.shape
            for i in range(len_x):
                for j in range(num_loop):
                    sampled_index[i][j] = [k.shape[0] for k in sampled_index[i][j]]
            sampled_index = sampled_index[:,:,1:].astype(int)
            # subset maiximum budget result
            mean = sampled_index.mean(axis=1)[-1]
            std = sampled_index.std(axis=1)[-1]
            x = np.arange(len(mean))+1
            # plot
            row, col = pos_dict[d_type]
            name, color, mode, dash = plot_dict[key]
            rgb = 'rgb' + str(tuple(color))
            fig.add_bar(x=x, y=mean, error_y=dict(type='data', array=std), name=name, marker_color=rgb,
                        showlegend=showlegend, row=row, col=col,)
        showlegend = False
    fig.update_layout(yaxis1_title='Number of Query Sample',
                      xaxis1_title='Query Cost of Each Domain', xaxis2_title='Query Cost of Each Domain',
                      xaxis3_title='Query Cost of Each Domain', xaxis4_title='Query Cost of Each Domain',
                      xaxis3=dict(tickmode='linear', tick0=1, dtick=1), xaxis5=dict(tickmode='linear', tick0=1, dtick=1),
                      font=dict(family="PTSerif", size=14,),
                      legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.45, xanchor="center", x=0.5),
                      width=1200, height=350, margin=dict(t=30, b=50),)
    # output
    file_path = result_dir / 'NumQuerySample.pdf'
    fig.write_image(file_path)
    return fig


def plot_CompareBudget():

    def find_nearest_budget(budget, accuracy, th,):
        from scipy import stats as st
        accuracy = np.asarray(accuracy)
        idx = np.argwhere(accuracy >= th).min()
        _x, _y = budget[idx-1:idx+1], accuracy[idx-1:idx+1]
        slope, intercept, _, _, _ = st.linregress(_x, _y)
        nearest_budget = (th - intercept) / slope
        return nearest_budget

    th_dict = dict(mnist=0.65, portraits=0.84, cover=0.70, gas=0.62)
    target = ['gdamf+random', 'gdamf+al']

    # parse data
    r_budget, a_budget = list(), list()
    for d_type in pos_dict.keys():
        # read data
        d_path = list(result_dir.glob(f'cost_{d_type}*'))
        data = {p.stem.split('_')[-1]: pd.read_pickle(p) for p in d_path if p.stem.split('_')[-1] in target}
        th = th_dict[d_type]
        budget = np.asarray(data['gdamf+random']['x'])
        r_accuracy = np.mean(data['gdamf+random']['acc'], axis=1)
        a_accuracy = np.mean(data['gdamf+al']['acc'], axis=1)
        r_budget.append(find_nearest_budget(budget, r_accuracy, th))
        a_budget.append(find_nearest_budget(budget, a_accuracy, th))
    # plot
    fig = go.Figure()
    fig.add_bar(x=data_label, y=r_budget, name='GDAMF+Random')
    fig.add_bar(x=data_label, y=a_budget, name='GDAMF+AL')
    fig.update_layout(yaxis_title='Budget', width=800, height=500,
                      margin=dict(t=15, b=30, l=100),
                      font=dict(family="PTSerif", size=14,),
                      legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
    # output
    file_path = result_dir / 'CompareBudget.pdf'
    fig.write_image(file_path)
    return fig


def plot_CostExperiment(num_repeat:int):
    global plotly_rgb, plot_dict, pos_dict

    fig = go.Figure().set_subplots(1, 4, vertical_spacing=0.1, horizontal_spacing=0.03, subplot_titles=data_label, shared_xaxes=True)

    all_result_df = {}
    showlegend = True
    for d_type in pos_dict.keys():
        d_path = list(result_dir.glob(f'cost_{d_type}*'))
        data = {p.stem.split('_')[-1]: pd.read_pickle(p) for p in d_path}
        dataframe = []
        for key in data.keys():
            # for budget-free methods
            if key.split('-')[0] in ['so', 'gst', 'gift', 'aux']:
                len_x = data[key]['x'].shape[0]
                data[key]['acc'] = np.tile(np.array(data[key]['acc']), len_x).reshape(len_x, num_repeat)
            else:
                data[key]['acc'] = np.array(data[key]['acc'])
            # as dataframe
            df = pd.DataFrame(index=data[key]['x'], data=data[key]['acc'])
            df = df.melt(ignore_index=False, value_name=key).drop('variable', axis=1).groupby(level=0).agg(['mean', 'std'])
            dataframe.append(df)
        dataframe = pd.concat(dataframe, axis=1)
        all_result_df[d_type] = dataframe
        # Specify the display order
        for key in ['aux', 'gift', 'gst', 'dsaoda', 'gdamf+al', 'gdamf+random', 'oracle', 'so']:
            # select the best hyper parameter
            if key in ['dsaoda', 'gift', 'aux']:
                subset_col = [c for c in dataframe.columns if (key in c[0]) & ('mean' in c[1])]
                key = dataframe[subset_col].max().idxmax()[0]
            # plot mean and std
            row, col = pos_dict[d_type]
            name, color, mode, dash = plot_dict[key] if '-' not in key else plot_dict[key.split('-')[0]]
            rgb = 'rgb' + str(tuple(color))
            rgba = 'rgba' + str(tuple(color+[0.3]))
            x = dataframe.index.to_numpy()
            mean = dataframe[key]['mean'].values
            std = dataframe[key]['std'].values
            add_var_scatter_plot(fig, x, mean, color, name, dash, row, col, showlegend, std, mode)
        # not need legend, because already ploted
        showlegend = False
    # modify layout
    x_tick = dict(tickmode='linear', tick0=0, dtick=500)
    x_range, y_range = [1000, 3100], [0.25, 1.0]
    fig.update_layout(yaxis1_title='Accuracy', 
                      xaxis1_title='Budget', xaxis2_title='Budget',xaxis3_title='Budget',xaxis4_title='Budget',
                      yaxis1_range=y_range, yaxis2_range=y_range, yaxis3_range=y_range, yaxis4_range=y_range,
                      xaxis1_range=x_range, xaxis2_range=x_range, xaxis3_range=x_range, xaxis4_range=x_range, 
                      xaxis1=x_tick, xaxis2=x_tick, xaxis3=x_tick, xaxis4=x_tick, xaxis5=x_tick,
                      margin=dict(t=30, b=30, r=30, l=30),
                      width=1250, height=350,
                      font=dict(family="PTSerif", size=14,),
                      legend=dict(orientation="h", bordercolor="Black", borderwidth=0.3, yanchor="bottom", y=-0.45, xanchor="center", x=0.5))
    # output
    file_path = result_dir / 'CostExperiment.pdf'
    fig.write_image(file_path)
    return fig, all_result_df


# In[ ]:




