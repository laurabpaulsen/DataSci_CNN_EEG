import numpy as np

# sklearn tools
from sklearn.model_selection import train_test_split

# pytorch tools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# skorch tools
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

def get_cnn_model(**kwargs):
    """
    Returns the CNN model

    Parameters
    ----------
    kwargs : dict
        Dictionary of arguments to pass to the model

    Returns
    -------
    NeuralNetClassifier
        The CNN model
    """
    early_stopping = get_early_stopping()
    return NeuralNetClassifier(
        Net,
        max_epochs=30,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        iterator_train__shuffle=True,
        callbacks=[early_stopping],
        **kwargs
    )

# early stopping callback
def get_early_stopping():
    return EarlyStopping(
        monitor='valid_loss',
        lower_is_better=True,
        patience=5,
        threshold=0.0001,
        threshold_mode='rel'
    )

class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv3d(50, 18, kernel_size=3)
            self.pool1 = nn.MaxPool3d(kernel_size=1)
            self.bn1 = nn.BatchNorm3d(18)
            self.drop1 = nn.Dropout3d(p=0.2)

            self.conv2 = nn.Conv3d(18, 128, kernel_size=(3, 3, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn2 = nn.BatchNorm3d(128)
            self.drop2 = nn.Dropout3d(p=0.2)
            
            self.conv3 = nn.Conv3d(128, 128, kernel_size=(3, 3, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn3 = nn.BatchNorm3d(128)
            self.drop3 = nn.Dropout3d(p=0.2)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 2)

        def forward(self, x):
            # convert tensor to float to avoid error 
            x = x.float()
            x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
            x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv2(x)))))
            x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv3(x)))))
            x = self.avgpool(x)
            x = x.view(-1, 128)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        

def return_param_grid():
    """
    Returns the parameter grid for grid search for the CNN model. Ensures that the same parameters are used across all scripts.
    """

    return {
        "lr": [0.0001, 0.0001, 0.001, 0.01, 0.1],
        "batch_size": [4, 8, 16],
    }
