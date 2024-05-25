from pathlib import Path
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))



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

        # freeze all layers except for fc1 and fc2
        for name, param in joined_model.module_.named_parameters():
            if "fc1" in name or "fc2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        joined_model.fit(gafs, labels)

        # create output dir for the subject
        output_dir = path / "mdl" / subj
            
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        torch.save(joined_model, output_dir / f"model_finetuned.pt")