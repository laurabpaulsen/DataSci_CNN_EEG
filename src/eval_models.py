from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.cnn import Net

def load_model(model_folder: Path):
    """
    Load the model from the folder.
    """
    model = torch.load(model_folder / "model.pt")
    
    return model


class ensemble_model:
    def __init__(self, models):
        self.models = models

    def predict(self, test_data):
        predictions = []
        for mdl in self.models:
            # get the predictions
            predictions.append(mdl.predict(test_data))

        # ensemble the predictions
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)

        # turn the predictions into a integer
        predictions = predictions.astype(int)

        return predictions


if __name__ in "__main__":
    path = Path(__file__).parents[1]

    results_path = path / "results" 
    results_path.mkdir(parents=True, exist_ok=True)

    # loop over the subjects
    subjects_gaf_folder = list((path / "data" / "gaf").glob("sub-*"))
    models = list((path / "mdl").glob("*")) + ["ensemble"]

    results = {}

    # loop over the models
    for model in models:
        
        if model == "ensemble":
            models_for_ensemble = list((path / "mdl").glob("sub-*"))
            mdl = ensemble_model([load_model(model_folder) for model_folder in models_for_ensemble])
            model_name = "ensemble"

        else:
            model_name = model.name
            mdl = torch.load(model / "model.pt")
    
        for folder in subjects_gaf_folder: # loop over gafs for each subject to test
            test_subject = folder.name
            
            # reading in the test data
            gafs, labels = np.load(folder / f"gafs_test.npy"), np.load(folder / f"labels_test.npy")
            labels = torch.LongTensor(labels)

            # get the accuracy
            pred = mdl.predict(gafs)

            # f1 score
            labels = labels.numpy()
            accuracy = (pred == labels).mean()
            f1 = f1_score(labels, pred, average="macro")
            recall = recall_score(labels, pred, average="macro")
            precision = precision_score(labels, pred, average="macro")


            # save the results
            tmp_results = {
                "model": model_name,
                "test_subject": test_subject,
                "accuracy": accuracy,
                "f1": f1,
                "recall": recall,
                "precision": precision
            }

            results[f"{model_name}_{test_subject}"] = tmp_results

    # save the results as a text file
    with open(results_path / "results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key} : {value}\n")
