import torch
from joblib import dump, load
import torch.nn as nn
import time
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE  # 导入t-SNE
import numpy as np

matplotlib.rc("font", family='Microsoft YaHei')

torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

from model import CNNBiLSTMCrossAttModel


def dataloader(batch_size, workers=2):
    train_xdata = load('train_features_1024_10c')
    train_ylabel = load('trainY_1024_10c')
    val_xdata = load('val_features_1024_10c')
    val_ylabel = load('valY_1024_10c')
    test_xdata = load('test_features_1024_10c')
    test_ylabel = load('testY_1024_10c')

    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                  batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader


import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def model_train(train_loader, test_loader, model, parameter):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_size = parameter['batch_size']
    epochs = parameter['epochs']
    learn_rate = parameter['learn_rate']
    loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
    optimizer = torch.optim.Adam(model.parameters(), learn_rate) 

    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size


    best_accuracy = 0.0
    best_model = model

    train_loss = [] 
    train_acc = []  
    validate_acc = []
    validate_loss = []

    print('*' * 20, '开始训练', '*' * 20)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        loss_epoch = 0.  
        correct_epoch = 0  
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
          
            optimizer.zero_grad()
           
            y_pred = model(seq)  # torch.Size([16, 10])
           
            probabilities = F.softmax(y_pred, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_epoch += (predicted_labels == labels).sum().item()
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        with torch.no_grad():
            loss_validate = 0.
            correct_validate = 0
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
  
                probabilities = F.softmax(pre, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()
            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model 

    last_model = model
    print('*' * 20, '训练结束', '*' * 20)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print("best_accuracy :", best_accuracy)

    plt.plot(range(epochs), train_loss, color='b', label='train_loss')
    plt.plot(range(epochs), train_acc, color='g', label='train_acc')
    plt.plot(range(epochs), validate_loss, color='y', label='validate_loss')
    plt.plot(range(epochs), validate_acc, color='r', label='validate_acc')
    plt.legend()
    plt.savefig('train_result', dpi=100)

    return last_model, best_model

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def plot_tsne(model, train_loader, device):
    model.eval()  
    features, labels = [], []

    with torch.no_grad():
        for seq, label in train_loader:
            seq = seq.to(device)
            feature = model(seq) 
            features.append(feature.cpu().numpy()) 
            labels.append(label.cpu().numpy())  

    features = np.concatenate(features, axis=0)  
    labels = np.concatenate(labels, axis=0)  

    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', alpha=0.6)
    plt.colorbar(scatter)
    num_classes = 10  
    class_labels = ['normal'] + [f'fault{i}' for i in range(1, num_classes)]
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=14,
               frameon=False)

    plt.title('t-SNE Visualization', fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('tsne_result.png', dpi=300, bbox_inches='tight')
    plt.show()


