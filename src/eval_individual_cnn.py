from pathlib import Path
import numpy as np
import torch


import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.cnn import Net

if __name__ in "__main__":
    path = Path(__file__).parents[1]

    results_path = path / "results" / "individual_cnn"
    results_path.mkdir(parents=True, exist_ok=True)

    # loop over the subjects

    subjects_gaf_folder = list((path / "data" / "gaf").glob("sub-*"))
    individual_models = list((path / "mdl").glob("sub-*"))

    results = {}

    # loop over the models
    for model_folder in individual_models:
        model_name = model_folder.name
        
        # load the model
        model = torch.load(model_folder / "model.pt")
    
        for folder in subjects_gaf_folder: # loop over gafs for each subject to test
            test_subject = folder.name
            print(f"Testing {model_name} on {test_subject}")
            # reading in the test data
            gafs, labels = np.load(folder / f"{folder.name}_gafs_test.npy"), np.load(folder / f"{folder.name}_labels_test.npy")
            labels = torch.LongTensor(labels)

            # get the accuracy
            accuracy = model.score(gafs, labels)

            # save the results
            tmp_results = {
                "model": model_name,
                "test_subject": test_subject,
                "accuracy": accuracy
            }

            results[f"{model_name}_{test_subject}"] = tmp_results

    # save the results as a text file
    with open(results_path / "individual_models.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key} : {value}\n")
