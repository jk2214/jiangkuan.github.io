import numpy as np
import pandas as pd
from scipy.io import loadmat
import sklearn
import torch
from joblib import dump, load


def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):

    stride = int(time_steps * (1 - overlap_ratio))
    samples = (len(data) - time_steps) // stride + 1

    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    index_count = 0
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame


def normalize(data):

    s = (data - min(data)) / (max(data) - min(data))
    return s


def make_datasets(data_file_csv, split_rate=[0.7, 0.2, 0.1]):

    origin_data = pd.read_csv(data_file_csv)

    time_steps = 256
    overlap_ratio = 0.5
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    label = 0
    for column_name, column_data in origin_data.items():

        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1
        samples_data = pd.concat([samples_data, split_data])
        samples_data = sklearn.utils.shuffle(samples_data)

    sample_len = len(samples_data)
    train_len = int(sample_len * split_rate[0])
    val_len = int(sample_len * split_rate[1])
    train_set = samples_data.iloc[0:train_len, :]
    val_set = samples_data.iloc[train_len:train_len + val_len, :]
    test_set = samples_data.iloc[train_len + val_len:sample_len, :]
    return train_set, val_set, test_set



def make_data_labels(dataframe):

    x_data = dataframe.iloc[:, 0:-1]
    # 标签值
    y_label = dataframe.iloc[:, -1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64'))
    return x_data, y_label


if __name__ == '__main__':

    file_names = ['0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat', '14_1.mat', '14_2.mat', '14_3.mat', '21_1.mat',
                  '21_2.mat', '21_3.mat']


    data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                    'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
    columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer',
                    'de_21_inner', 'de_21_ball', 'de_21_outer']

    data_12k_10c = pd.DataFrame()
    for index in range(10):

        data = loadmat(f'matfiles\\{file_names[index]}')
        dataList = data[data_columns[index]].reshape(-1)
        data_12k_10c[columns_name[index]] = dataList[:119808]
    print(data_12k_10c.shape)

    data_12k_10c.set_index('de_normal', inplace=True)
    data_12k_10c.to_csv('data_12k_10c.csv')

    train_set, val_set, test_set = make_datasets('data_12k_10c.csv')

    train_xdata, train_ylabel = make_data_labels(train_set)
    val_xdata, val_ylabel = make_data_labels(val_set)
    test_xdata, test_ylabel = make_data_labels(test_set)

    dump(train_xdata, 'trainX_256_10c')
    dump(val_xdata, 'valX_256_10c')
    dump(test_xdata, 'testX_256_10c')
    dump(train_ylabel, 'trainY_256_10c')
    dump(val_ylabel, 'valY_256_10c')
    dump(test_ylabel, 'testY_256_10c')

    print('数据 形状：')
    print(train_xdata.size(), train_ylabel.shape)
    print(val_xdata.size(), val_ylabel.shape)
    print(test_xdata.size(), test_ylabel.shape)
