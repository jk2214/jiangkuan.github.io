
from joblib import dump, load
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import numpy as np

torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import ResNetBiGRU

def model_test(model, test_loader):

    loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss

    correct_test = 0
    test_loss = 0

    class_labels = []
    predicted_labels = []

    with torch.no_grad():
        for test_data, test_label in test_loader:
            model.eval()

            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predictedlabel = torch.argmax(probabilities, dim=1)
            correct_test += (predictedlabel == test_label).sum().item()
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()
            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predictedlabel.tolist())

    test_accuracy = correct_test / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Accuracy: {test_accuracy:4.4f}  Test Loss: {test_loss:10.8f}')

    return class_labels, predicted_labels


