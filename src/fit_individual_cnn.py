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

from utils.cnn import get_cnn_model, return_param_grid


def main():
    path = Path(__file__)

    param_grid = return_param_grid()
    train_subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08"]

    for subj in train_subjects:

        gaf_path = path.parents[1] / "data" / "gaf" 
        gafs, labels = np.load(gaf_path / subj / f"gafs_train.npy"), np.load(gaf_path / subj / f"labels_train.npy")

        # change labels to LongTensor to avoid the error: RuntimeError: Expected object of scalar type Long but got scalar type Float
        labels = torch.LongTensor(labels) 

        model = get_cnn_model()

        # run grid search
        gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
        gs.fit(gafs, labels)

        # generate the model with the best parameters to save
        best_params = gs.best_params_

        model2 = get_cnn_model(
            lr = best_params['lr'],
            batch_size = best_params['batch_size']
        )

        model2.fit(gafs, labels)

        # create output dir for the subject
        output_dir = path.parents[1] / "mdl" / subj
        
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        torch.save(model2, output_dir / f"model.pt")

if __name__ == "__main__":
    main()