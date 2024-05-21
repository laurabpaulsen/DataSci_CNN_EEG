from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from skorch import NeuralNetClassifier 
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.cnn import get_cnn_model, return_param_grid


def prep_data(X, y):
    
    # shuffle the data
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    # change labels to LongTensor to avoid the error: RuntimeError: Expected object of scalar type Long but got scalar type Float    
    y = torch.LongTensor(y) 
    
    return X, y


if __name__ in "__main__":
    path = Path(__file__).parents[1]

    param_grid = return_param_grid()

    subject_data_X = []
    subject_data_y = []

    train_subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]

    for subj in train_subjects:

        gaf_path = path / "data" / "gaf" 
        gafs, labels = np.load(gaf_path / subj / f"gafs_train.npy"), np.load(gaf_path / subj / f"labels_train.npy")
        print(gafs.shape, labels.shape)
        subject_data_X.append(gafs)
        subject_data_y.append(labels)

    X = np.vstack(subject_data_X)
    y = np.hstack(subject_data_y)


    X, y = prep_data(X, y)

    
    model = get_cnn_model()

    # run grid search
    gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
    gs.fit(X, y)

    # generate the model with the best parameters to save
    model = get_cnn_model(lr = gs.best_params_['lr'], batch_size = gs.best_params_['batch_size'])
    model.fit(X, y)

    # create output dir for the subject
    output_dir = path / "mdl" / "joint"
        
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    torch.save(model, output_dir / f"model.pt")