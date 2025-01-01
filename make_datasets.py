import pandas as pd
import numpy as np
from vmdpy import VMD
import torch
from joblib import dump, load



alpha = 2000
tau = 0
DC = 0
init = 1
tol = 1e-7

def lms_filter(signal, mu=0.01, M=32):

    N = len(signal)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)


    for n in range(M, N):
        x = signal[n-M:n]
        y[n] = np.dot(w, x)
        e[n] = signal[n] - y[n]
        w = w + mu * e[n] * x

    return e

def make_feature_datasets(data, imfs_k):

    samples = data.shape[0]
    signl_len = data.shape[1]
    data = np.array(data)

    features_num = imfs_k + 1

    features = np.zeros((samples, features_num, signl_len))

    for i in range(samples):

        u, u_hat, omega = VMD(data[i], alpha, tau, imfs_k, DC, init, tol)

        lms_result = lms_filter(data[i])

        combined_matrix = np.vstack((u, lms_result))
        features[i] = combined_matrix


    features = torch.tensor(features).float()
    return features


if __name__ == '__main__':


    train_xdata = load('trainX_256_10c')
    val_xdata = load('valX_256_10c')
    test_xdata = load('testX_256_10c')
    train_ylabel = load('trainY_256_10c')
    val_ylabel = load('valY_256_10c')
    test_ylabel = load('testY_256_10c')

    K = 4

    train_features = make_feature_datasets(train_xdata, K)
    val_features = make_feature_datasets(val_xdata, K)
    test_features = make_feature_datasets(test_xdata, K)


    dump(train_features, 'train_features_256_10c')
    dump(val_features, 'val_features_256_10c')
    dump(test_features, 'test_features_256_10c')

    print('数据 形状：')
    print(train_features.shape, train_ylabel.shape)
    print(val_features.shape, val_ylabel.shape)
    print(test_features.shape, test_ylabel.shape)

    import pandas as pd
    import numpy as np
    from joblib import dump, load

    all_features = np.concatenate([train_features, val_features, test_features], axis=0)

    all_features_df = pd.DataFrame(all_features.reshape(all_features.shape[0], -1))  # reshape为一维

    all_features_df.to_csv('all_features.csv', index=False)

    print("所有特征数据已保存为 'all_features.csv'")

import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

signal = np.random.randn(1024)

alpha = 2000
tau = 0
DC = 0
init = 1
tol = 1e-7
K = 4


u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)


def lms_filter(signal, mu=0.01, M=32):
    N = len(signal)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)

    for n in range(M, N):
        x = signal[n-M:n]
        y[n] = np.dot(w, x)
        e[n] = signal[n] - y[n]
        w = w + mu * e[n] * x

    return e

lms_result = lms_filter(signal)

lms_results = [lms_filter(u[i, :]) for i in range(K)]



import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False



plt.figure(figsize=(15, 12))
plt.subplot(K + 1, 1, 1)
plt.plot(signal, 'r')
plt.title("Original Signal")

for num in range(K):
    plt.subplot(K + 1, 1, num + 2)
    plt.plot(u[num, :], 'b')
    plt.title(f"IMF {num + 1} - Original Component")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))
plt.subplot(K + 1, 1, 1)
plt.plot(signal, 'r')
plt.title("Original Signal")

for num in range(K):
    plt.subplot(K + 1, 1, num + 2)
    plt.plot(lms_results[num], 'g')
    plt.title(f"IMF {num + 1} - LMS Filtered Component")

plt.tight_layout()
plt.show()
