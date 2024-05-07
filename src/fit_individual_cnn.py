import numpy as np
from pathlib import Path
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier # allows us to use pytorch with sklearn for grid search
from sklearn.model_selection import GridSearchCV

# local imports 
import sys
sys.path.append(str(Path(__file__).parents[1]))

from utils.cnn import Net


def main():
    path = Path(__file__)

    ### SETTING UP PARAMETERS ###
    max_epochs = 3 ## UP THIS ALOT  
    # prep for grid search
    param_grid = {
        "lr": [0.0001, 0.001, 0.01],
        "batch_size": [4, 8],

    }

    train_subjects = ["sub-01", "sub-02"]

    for subj in train_subjects:

        gaf_path = path.parents[1] / "data" / "gaf" 
        gafs, labels = np.load(gaf_path / subj / f"{subj}_gafs_train.npy"), np.load(gaf_path / subj / f"{subj}_labels_train.npy")

        # change labels to LongTensor to avoid the error: RuntimeError: Expected object of scalar type Long but got scalar type Float
        labels = torch.LongTensor(labels) 

        net = NeuralNetClassifier(
            Net,
            max_epochs = max_epochs,
            criterion = nn.CrossEntropyLoss,
            optimizer = optim.SGD,
            iterator_train__shuffle = True
        )

        # run grid search
        gs = GridSearchCV(net, param_grid, cv=10, scoring='accuracy', verbose=2)
        gs.fit(gafs, labels)

        # generate the model with the best parameters to save
        best_params = gs.best_params_

        model = NeuralNetClassifier(
            Net,
            max_epochs = max_epochs,
            criterion = nn.CrossEntropyLoss,
            optimizer = optim.SGD,
            iterator_train__shuffle = True,
            lr = best_params['lr'],
            batch_size = best_params['batch_size']
        )

        model.fit(gafs, labels)

        # create output dir for the subject
        output_dir = path.parents[1] / "mdl" / subj
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        torch.save(model, output_dir / f"model.pt")



    


if __name__ == "__main__":
    main()