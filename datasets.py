#!/usr/bin/env python
# coding: utf-8

import warnings
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import ndimage
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

data_dir = Path('/home/')


def mnist_with_loader(batch_size:int=128) -> dict:
    global data_dir
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kwargs = dict(dataset=MNIST(data_dir, train=True, download=True, transform=ToTensor()), batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(**kwargs)
        kwargs['dataset'].train = False
        test_loader = DataLoader(**kwargs)
    return {'train': train_loader, 'test': test_loader}


def load_RotatedMNIST(num_inter_domain:int, rot_kwargs:dict, seed:int=1234) -> dict:
    """
    @memo
    num_inter_domain : inter domain data will be vsplit by this param
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    dict needs this info -> num of sumple, start_angle, end_angle, gradual
    my settings
    num_inter_domain = 21
    rot_kwargs = {'source': [2000, 0, 5, False],
                  'inter': [42000, 5, 55, True],
                  'target': [2000, 55, 60, False],
                  'eval': [2000, 55, 60, False]}),
    """
    global data_dir
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # rotated mnist does not need test data
        dataset = MNIST(data_dir, train=True, download=True)
    x = np.array(dataset.data).astype(np.float32) / 255
    y = np.array(dataset.targets)
    index = np.arange(x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    num_domain_samples = [i[0] for i in rot_kwargs.values()]  # [5000, 42000, 2000]
    split_index = np.split(index, np.cumsum(num_domain_samples))
    x_all, y_all = list(), list()
    for idx, (key, item) in zip(split_index, rot_kwargs.items()):
        rotated_x = make_rotated_mnist(x[idx], *item[1:])
        if key == 'inter':
            x_all += np.vsplit(rotated_x.reshape(-1, 1, 28, 28), num_inter_domain)
            y_all += np.hsplit(y[idx], num_inter_domain)
        else:
            x_all.append(rotated_x.reshape(-1, 1, 28, 28))
            y_all.append(y[idx])
    return x_all, y_all


def make_rotated_mnist(x:np.ndarray, start:int, end:int, gradual:bool) -> np.ndarray:
    """
    @param
    x : image array, shape -> (height, width, channel)
    start, end : rotate angle
    gradual : True -> Intermediate Domain
    """
    num_points = x.shape[0]
    if not gradual:
        angle = np.random.uniform(low=start, high=end, size=num_points)
    else:
        angle = np.linspace(start, end, num_points, endpoint=False)
    rotated_x = np.array([ndimage.rotate(i, a, reshape=False) for i,a in zip(x, angle)])
    return rotated_x


def load_Portraits(num_inter_domain:int, num_domain_samples:dict):
    """
    @memo
    num_inter_domain : inter domain data will be vsplit by this param
    num_domain_samles : number of samples in each domain.

    my settings
    num_inter_domain=14,
    num_domain_samles = {'source': 1000,
                          'inter': 14000,
                         'target': 1000,
                           'eval': 1000}
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0
    """
    global data_dir
    # prepare portraits image
    f_path = Path(data_dir) / 'portraits/F'
    m_path = Path(data_dir) / 'portraits/M'
    f_list = list(f_path.glob('*.png'))
    m_list = list(m_path.glob('*.png'))
    f = pd.DataFrame({'img_path':f_list})
    m = pd.DataFrame({'img_path':m_list})
    f['sex'] = 1  # female as 1
    m['sex'] = 0  # male as 0
    df = pd.concat([f,m]).reset_index(drop=True)
    df['year'] = df['img_path'].apply(lambda p : p.stem.split('_')[0]).astype(int)
    df = df.sort_values(by='year').reset_index(drop=True)

    def convert_portraits(p:Path):
        # read, gray scale, resize
        img = Image.open(p).convert('L').resize((32,32), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32) / 255
        return img

    # split to each domain
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.loc[idx, 'img_path'].apply(convert_portraits).tolist()
        x = np.array(x).reshape(-1, 1, 32, 32)
        y = df.loc[idx, 'sex'].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def load_GasSensor(num_inter_domain:int, num_domain_samples:dict):
    """
    @memo
    num_inter_domain : inter domain data will be vsplit by this param
    num_domain_samles : number of samples in each domain.

    my setting
    num_inter_domain = 7
    num_domain_samples = {'source': 1000,
                          'inter': 7000,
                          'target': 1000,
                          'eval': 1000}
    @ about data
    1: Ethanol; 2: Ethylene; 3: Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene
    we merge Toluene data to Acetaldehyde, because Toluene is absence in 3 batch
    we drop last 3,600 data, because it is not continuous data.
    http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations#
    """
    # prepare gas sensor data
    global data_dir
    d_path = Path(data_dir) / 'gas_sensor'
    dat_path = list(d_path.glob('*.dat'))

    def read_gas_sensor_data(p:Path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            columns = ['target'] + [f'x{i+1}' for i in range(128)]
            df = pd.read_csv(p, sep='\s+', header=None, names=columns)
            extract_label = lambda x : int(x.split(';')[0])
            extract_feature = lambda x : float(x.split(':')[1])
            df['target'] = df['target'].apply(extract_label)
            df.iloc[:,1:] = df.iloc[:,1:].applymap(extract_feature)
            df['batch_num'] = int(p.stem.replace('batch',''))
        return df.sample(frac=1, random_state=1234)

    df = pd.concat([read_gas_sensor_data(p) for p in dat_path])
    df = df.query('batch_num != 10')
    df = df.sort_values(by='batch_num').drop('batch_num', axis=1).reset_index(drop=True)
    df['target'] = df['target'].apply(lambda t : 4 if t== 6 else t)
    df['target'] -= 1
    df.iloc[:,1:] = df.iloc[:,1:].astype(np.float32)

    # split to each domain
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.iloc[idx, 1:].values
        y = df.loc[idx, 'target'].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def load_CoverType(num_inter_domain:int, num_domain_samples:dict):
    """
    @memo
    num_inter_domain : inter domain data will be vsplit by this param
    num_domain_samles : number of samples in each domain.
    my settings
    num_inter_domain = 30
    num_domain_samples = {'source': 10000, 'inter': 300000, 'target': 10000, 'eval': 10000})
    @ about data
    1:Spruce/Fir, 2:Lodgepole Pine, 3:Ponderosa Pine, 4:Cottonwood/Willow, 5:Aspen, 6:Douglas-fir, 7:Krummholz
    we extract label 1 and 2
    https://archive.ics.uci.edu/ml/datasets/covertype
    """
    global data_dir
    # prepare CoverType data
    data_path = Path(data_dir) / 'cover_type/covtype.data'
    columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
               'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    columns += [f'Wilderness_Area_{i+1}' for i in range(4)]
    columns += [f'Soil_Type_{i+1}' for i in range(40)]
    columns += ['Cover_Type']
    df = pd.read_csv(data_path, header=None, names=columns)
    df = df.query('Cover_Type < 3').reset_index(drop=True)
    df['Cover_Type'] = df['Cover_Type'] - 1  # label convert 0/1
    df['Distance_To_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df = df.query('0 < Distance_To_Hydrology < 700')
    df = df.sort_values(by='Distance_To_Hydrology', ascending=False).reset_index(drop=True)
    df = df.drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Distance_To_Hydrology'], axis=1)
    df.iloc[:,:-1] = StandardScaler().fit_transform(df.drop(['Cover_Type'], axis=1).values)

    # split to each domain
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.iloc[idx, :-1].values.astype(np.float32)
        y = df.loc[idx, 'Cover_Type'].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def load_Bank(num_inter_domain:int, num_domain_samples:dict):
    """
    @memo
    num_inter_domain : inter domain data will be vsplit by this param
    num_domain_samles : number of samples in each domain.
    my settings
    num_inter_domain = 17
    num_domain_samples = {'source': 2000,
                          'inter': 34000,
                          'target': 2000,
                          'eval': 2000}
    https://archive.ics.uci.edu/ml/datasets/bank+marketing
    """
    global data_dir
    d_path = Path(data_dir) / 'BankMarketing/bank-full.csv'
    df = pd.read_csv(d_path, sep=';')
    df = df.drop('duration', axis=1)  # see Attribute Information
    df['y'] = df['y'].apply(lambda y: 1 if y=='yes' else 0)
    df = pd.get_dummies(df, drop_first=True)
    df = df.reset_index(drop=True).astype(np.float32)
    # split to each domain
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.loc[idx, :].drop('y', axis=1).values
        y = df.loc[idx, 'y'].values.astype(int)
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def make_gradual_data(steps:int=20, n_samples:int=2000, start:int=0, end:int=90, mode:str='moon') -> dict:
    """
    @param
    steps : int, how gradual is it
    n_samples : int, how many samples, each domains
    start : int, param of shift
    end : int, param of shift
    mode : str, moon, blob, xor
    @return
    data : dict, source_xy, inter_xy, target_xy
    """
    if mode == 'moon':
        x, y = make_moons(n_samples=n_samples, random_state=8, noise=0.05)
        convert_f = _convert_moon
    elif mode == 'blob':
        x, y = make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.5, random_state=1233345)
        convert_f = _convert_blob
    elif mode == 'xor':
        np.random.seed(1234)
        size = n_samples // 2
        x1 = np.random.normal(loc=5, scale=1, size=(size,2)).astype(np.float32)
        x2 = np.random.normal(loc=-5, scale=1, size=(size,2)).astype(np.float32)
        x = np.vstack([x1, x2])
        y = np.array([0]*x1.shape[0] + [1]*x2.shape[0])
        convert_f = _convert_xor
    else:
        raise

    shifts = np.linspace(start, end, steps)
    x_all, y_all = list(), list()
    for shift in shifts:
        x_all.append(convert_f(x, shift))
        y_all.append(y)
        # for eval data
        if shift == shifts[-1]:
            x_all.append(convert_f(x, shift))
            y_all.append(y)
    return x_all, y_all


def _convert_moon(x:np.ndarray, shift:int) -> np.ndarray:
    x_copy = x.copy()
    rad = np.deg2rad(shift)
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)],
                           [-np.sin(rad), np.cos(rad)]])
    rot_x = x_copy @ rot_matrix
    return rot_x.astype(np.float32)


def _convert_blob(x:np.ndarray, shift:int) -> np.ndarray:
    x_copy = StandardScaler().fit_transform(x)
    rad = np.deg2rad(shift * 10)
    rot_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                           [np.sin(rad), np.cos(rad)]])
    rot_x = x_copy @ rot_matrix
    rot_x[:,0] += shift
    return rot_x.astype(np.float32)


def _convert_xor(x:np.ndarray, shift:int) -> np.ndarray:
    x_copy = x.copy()
    size = x_copy.shape[0] // 2
    x_copy[:size, 1] -= shift
    x_copy[size:, 1] += shift
    return x_copy.astype(np.float32)
