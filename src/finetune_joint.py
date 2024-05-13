from pathlib import Path
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))


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
    train_subjects = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]


    for subj in train_subjects:
        gaf_path = path / "data" / "gaf" 
        gafs, labels = np.load(gaf_path / subj / f"gafs_train.npy"), np.load(gaf_path / subj / f"labels_train.npy")
        # change labels to LongTensor to avoid the error: RuntimeError: Expected object of scalar type Long but got scalar type Float
        labels = torch.LongTensor(labels) 

        # load the joined model
        joined_model = torch.load(path / "mdl" / "joint" / "model.pt")


        joined_model.fit(gafs, labels)

        # create output dir for the subject
        output_dir = path / "mdl" / subj
            
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        torch.save(joined_model, output_dir / f"model_finetuned.pt")