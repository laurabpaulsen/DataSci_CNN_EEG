import numpy as np
from pathlib import Path
import argparse
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# sklearn tools
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier # allows us to use pytorch with sklearn for grid search
from sklearn.model_selection import GridSearchCV
# local imports 
import sys
sys.path.append(str(Path(__file__).parents[1]))


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

    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN on GAFs')
    parser.add_argument('--sub', type=str, default='sub-02')

    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(__file__)

    gaf_path = path.parents[1] / "data" / "gaf" 
    gafs, labels = np.load(gaf_path / args.sub / f"{args.sub}_gafs_train.npy"), np.load(gaf_path / args.sub / f"{args.sub}_labels_train.npy")

    # change labels to LongTensor to avoid the error: RuntimeError: Expected object of scalar type Long but got scalar type Float
    labels = torch.LongTensor(labels)

    # prep for grid search
    param_grid = {
        "lr": [0.0001, 0.001, 0.01],
        "batch_size": [4, 8, 16],

    }

    net = NeuralNetClassifier(
        Net,
        max_epochs=15,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        iterator_train__shuffle=True
    )


    # run grid search
    gs = GridSearchCV(net, param_grid, cv=10, scoring='accuracy', verbose=2)
    gs.fit(gafs, labels)

    print(gs.best_score_, gs.best_params_)

    # generate the model with the best parameters to save
    best_params = gs.best_params_

    model = NeuralNetClassifier(
        Net,
        max_epochs=15,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        iterator_train__shuffle=True,
        lr=best_params['lr'],
        batch_size=best_params['batch_size']
    )

    model.fit(gafs, labels)

    # create output dir for the subject
    output_dir = path.parents[1] / "models" / args.sub

    torch.save(model, output_dir / f"{args.sub}_model.pt")



    


if __name__ == "__main__":
    main()