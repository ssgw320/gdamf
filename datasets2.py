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

# +
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import ndimage
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from torchvision.datasets import MNIST

# +
#data_dir = Path('/home/')
data_dir = Path('/home/jupyter-e12813/ISM/data/')


def load_RotatedMNIST2(start=0, end=60, num_inter_domain=20, num_domain_samples=2000):
    """
    @param
    start, end: int, rotate angles
    num_inter_domain: int, how many intermediate domains needed
    num_inter_samples: set the same sample size in all domains (source, inter, target, eval)

    @memo
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    """
    global data_dir
    seed = 1234
    # load MNIST
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # rotated mnist does not need test data
        dataset = MNIST(data_dir, train=True, download=True)
    x = np.array(dataset.data).astype(np.float32) / 255
    y = np.array(dataset.targets)
    # set angles
    angles = np.linspace(start, end, num_inter_domain+2)
    angles = np.append(angles, end)
    # set sample size and index
    index = np.arange(x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    each_domain_samples = np.full(shape=(num_inter_domain+3), fill_value=num_domain_samples)  # source + inter + target +eval
    split_index = np.split(index, np.cumsum(each_domain_samples))
    # rotate
    x_all, y_all = list(), list()
    for idx, angle in zip(split_index, angles):
        #rotated_x = np.array([ndimage.rotate(i, np.random.normal(loc=angle, scale=5), reshape=False) for i in x[idx]])
        rotated_x = np.array([ndimage.rotate(i, angle, reshape=False) for i in x[idx]])
        x_all.append(rotated_x.reshape(-1, 1, 28, 28))
        y_all.append(y[idx])
    return x_all, y_all


def make_split_data(df: pd.DataFrame, target: str, num_inter_domain: int, num_domain_samples: dict):
    """ use for Portraits, Gas Sensor, Cover Type """
    split_index = np.split(np.arange(df.shape[0]), np.cumsum(list(num_domain_samples.values())))
    x_all, y_all = list(), list()
    for idx, key in zip(split_index, num_domain_samples.keys()):
        x = df.drop(target, axis=1).loc[idx].values
        y = df.loc[idx, target].values
        if key == 'inter':
            x_all += np.vsplit(x, num_inter_domain)
            y_all += np.hsplit(y, num_inter_domain)
        else:
            x_all.append(x)
            y_all.append(y)
    return x_all, y_all


def shuffle_target_and_eval(x_all: list, y_all: list):
    """ use for Portraits, Gas Sensor, Cover Type """
    tx, ty = x_all[-2].copy(), y_all[-2].copy()
    ex, ey = x_all[-1].copy(), y_all[-1].copy()
    marge_x = np.vstack([tx, ex])
    marge_y = np.hstack([ty, ey])
    idx = np.arange(marge_x.shape[0])
    np.random.seed(1234)
    np.random.shuffle(idx)
    t_idx, e_idx = idx[:tx.shape[0]], idx[tx.shape[0]:]
    x_all[-2], y_all[-2] = marge_x[t_idx], marge_y[t_idx]
    x_all[-1], y_all[-1] = marge_x[e_idx], marge_y[e_idx]
    return x_all, y_all


def load_Portraits(num_inter_domain=6, num_domain_samples='default', return_df=False, inverse=True):
    """
    @param
    num_inter_domain: inter domain data will be vsplit by this param
    num_domain_samles: number of samples in each domain.

    @memo
    image shape will be change, (N, height, width) -> (N, 1, height, width)
    https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0

    @ Kumar's setting
    In Kumar's setting, his last intermediate domain equal to our target domain
    """
    global data_dir
    if num_domain_samples == 'default':
        num_domain_samples = {'source': 2000, 'inter': 12000, 'target': 2000, 'eval': 2000}

    def read_path(sex: int):
        p = 'portraits/F' if sex == 1 else 'portraits/M'
        p = Path(data_dir) / p
        p_list = list(p.glob("*.png"))
        data_frame = pd.DataFrame({'img_path': p_list})
        data_frame['sex'] = sex
        return data_frame

    def convert_portraits(p: Path):
        # read, gray scale, resize
        img = Image.open(p).convert('L').resize((32,32), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32) / 255
        return img

    # prepare portraits image, female as 1, male as 0
    df = pd.concat([read_path(1), read_path(0)]).reset_index(drop=True)
    df['year'] = df['img_path'].apply(lambda p: p.stem.split('_')[0]).astype(int)
    if return_df:
        df['decade'] = df['year'].apply(lambda y: int(str(y)[:3]+'0'))
        return df
    if inverse:
        df = df.sort_values(by='year', ascending=False).reset_index(drop=True).drop('year', axis=1)
    else:
        df = df.sort_values(by='year').reset_index(drop=True).drop('year', axis=1)

    # split to each domain
    x_all, y_all = make_split_data(df, 'sex', num_inter_domain, num_domain_samples)
    x_all, y_all = shuffle_target_and_eval(x_all, y_all)
    for i, domain in enumerate(x_all):
        domain = np.array([convert_portraits(x) for x in domain.flatten()])
        x_all[i] = domain.reshape(-1, 1, 32, 32)
    return x_all, y_all


def load_GasSensor(num_inter_domain=1, num_domain_samples='default', drop_batch_num=[10], return_df=False):
    """
    @memo
    num_inter_domain: inter domain data will be vsplit by this param
    num_domain_samles: number of samples in each domain
    drop_batch_num: unused batch number
    return_df: return dataframe before vsplit

    @ about data
    we drop last 3,600 data (batch 10) , because it is not continuous data.
    http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations#
    """
    global data_dir
    d_path = Path(data_dir) / 'gas_sensor'
    dat_path = list(d_path.glob('*.dat'))

    if num_domain_samples == 'default':
        num_domain_samples = {'source': 3000, 'inter': 3000, 'target': 3000, 'eval': 1000}

    def read_gas_sensor_data(p: Path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            columns = ['target'] + [f'x{i+1}' for i in range(128)]
            df = pd.read_csv(p, sep='\s+', header=None, names=columns)
            df['target'] = df['target'].apply(lambda y: int(y.split(';')[0]))
            df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: float(x.split(':')[1]))
            df['batch_num'] = int(p.stem.replace('batch', ''))
        return df.sample(frac=1, random_state=1234)

    df = pd.concat([read_gas_sensor_data(p) for p in dat_path])
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float32)
    if return_df:
        return df
    #df['target'] = df['target'].apply(lambda t: 4 if t == 6 else t)
    df['target'] -= 1
    for b in drop_batch_num:
        df = df.query('batch_num != @b')
    df = df.sort_values(by='batch_num').reset_index(drop=True)
    df.iloc[:, 1:-1] = StandardScaler().fit_transform(df.iloc[:, 1:-1].values)
    df = df.drop('batch_num', axis=1)

    # split to each domain
    x_all, y_all = make_split_data(df, 'target', num_inter_domain, num_domain_samples)
    x_all, y_all = shuffle_target_and_eval(x_all, y_all)
    return x_all, y_all


def load_CoverType(num_inter_domain=8, num_domain_samples='default', sampling=None):
    """
    @param
    num_inter_domain: inter domain data will be vsplit by this param
    num_domain_samles: number of samples in each domain
    samplling: int, to reduce sample, random sampling from each domains. If not need, set None.

    @memo
    1:Spruce/Fir, 2:Lodgepole Pine, 3:Ponderosa Pine, 4:Cottonwood/Willow, 5:Aspen, 6:Douglas-fir, 7:Krummholz
    we extract label 1 and 2
    https://archive.ics.uci.edu/ml/datasets/covertype
    """
    global data_dir
    seed = 12345
    if num_domain_samples == 'default':
        num_domain_samples = {'source': 50000, 'inter': 400000, 'target': 30000, 'eval': 15000}
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
    # df = df.query('0 < Distance_To_Hydrology < 700')
    df = df.sort_values(by='Distance_To_Hydrology', ascending=False).reset_index(drop=True)
    df = df.drop(['Distance_To_Hydrology'], axis=1)
    df.iloc[:, :-1] = StandardScaler().fit_transform(df.drop(['Cover_Type'], axis=1).values)

    # split to each domain
    x_all, y_all = make_split_data(df, 'Cover_Type', num_inter_domain, num_domain_samples)
    x_all, y_all = shuffle_target_and_eval(x_all, y_all)
    # down sampling
    if sampling is not None:
        np.random.seed(seed)
        for i, x in enumerate(x_all):
            sampled_idx = np.random.choice(np.arange(x.shape[0]), size=sampling, replace=False)
            x_all[i], y_all[i] = x[sampled_idx], y_all[i][sampled_idx]
    return x_all, y_all


def make_gradual_data(steps=3, n_samples=2000, start=0, end=90):
    """
    @param
    steps: int, how gradual is it
    n_samples: int, how many samples, each domains
    start: int, param of shift
    end: int, param of shift
    """
    x, y = make_moons(n_samples=n_samples, random_state=8, noise=0.05)
    shifts = np.linspace(start, end, steps)
    x_all, y_all = list(), list()
    for shift in shifts:
        x_all.append(_convert_moon(x, shift))
        y_all.append(y)
        # for eval data
        if shift == shifts[-1]:
            x_all.append(_convert_moon(x, shift))
            y_all.append(y)
    return x_all, y_all


def _convert_moon(x: np.ndarray, shift: int) -> np.ndarray:
    x_copy = x.copy()
    rad = np.deg2rad(shift)
    rot_matrix = np.array([[np.cos(rad), np.sin(rad)],
                           [-np.sin(rad), np.cos(rad)]])
    rot_x = x_copy @ rot_matrix
    return rot_x.astype(np.float32)
