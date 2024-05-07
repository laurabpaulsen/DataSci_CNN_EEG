"""
    Converts the timeseries data into Gramian Angular Fields (both GAFD and GAFS), as well as Markov transition fields. For each timeseries a 3D array containing these are made. 
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pathlib import Path
import mne

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.gaf import generate_gafs


def train_test_split(X, y, train_size=0.8, test_size = 0.2):
    """
    Splits the data into train, validation and test sets. 

    NOTE: The data is not shuffled, so the data should be shuffled before calling this function.
    NOTE: Maybe implement splitting taking the number of samples in each class into account.
    """
    
    assert train_size +  + test_size == 1, "The sum of train_size, val_size and test_size must be 1"
    n = len(X)
    train_end = int(n * train_size)
    test_end = int(n * (train_size + test_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_test = X[train_end:test_end]
    y_test = y[train_end:test_end]

    return X_train, y_train, X_test, y_test


def main():
    path = Path(__file__).parents[1]
    preprc_path = path / 'data' /'preprocessed'
    outpath = path / 'data' / 'gaf'

    # loop over subjects
    subjects = [x.name for x in preprc_path.iterdir()]

    # check that outpath exists
    if not outpath.exists():
        outpath.mkdir(parents=True)

    for subject in subjects:

        outpath_sub = outpath / subject

        if not outpath_sub.exists():
            outpath_sub.mkdir()

        print(f"Converting timeseries to images for participant {subject}")

        # find the fif file in the subject directory
        fif_file = list(preprc_path.glob(f'{subject}/*.fif'))
        if len(fif_file) == 0:
            print(f"No fif file found for subject {subject}")
        
        elif len(fif_file) > 1:
            print(f"Multiple fif files found for subject {subject}")
        
        else:
            fif_file = fif_file[0]

        epochs = mne.read_epochs(fif_file, preload=True)
        
        # get Image and buttonpress events
        epochs = epochs['Image', 'Correct', 'Incorrect']

        # get the data
        X = epochs.get_data(copy = True)
        y = epochs.events[:, -1]
        
        event_id = {
            'Image/wPos': 21, # positive image (always following pos word) 
            'Word/wNeg': 12, # negative word
            'Image/wNeg': 22, # negative image (always following neg word) 
            'Word/wNeu': 13, # neutral word
            'Image/wNeu/iPos': 41, # positive image (after neu word) 
            'Image/wNeu/iNeg': 42, # negative image (after neu word) 
            'Correct/wPos': 101, # correct response ('b') to pos w + image 
            'Correct/wNeg': 102, # correct response ('y') to neg w + image 
            'Correct/wNeu/iPos': 111, # cor resp ('b') to neu w + pos image 
            'Correct/wNeu/iNeg': 112, # cor resp ('y') to neu w + neg image 
            'Incorrect/wPos': 202, # incor resp ('y') to pos w + image 
            'Incorrect/wNeg': 201, # incor resp ('b') to neg w + image 
            'Incorrect/wNeu/iPos': 212, # incor resp ('y') to neu w + pos i 
            'Incorrect/Neu/iNeg': 211 # incor resp ('b') to neu w + neg i
        }

        # convert y into 0 and 1 (image, button press) using event id as mapping
        # if key starts with Image  = 0
        # if key starts with Incorrect or Correct = 1
        for i in range(len(y)):
            current_y = y[i]
            key = list(event_id.keys())[list(event_id.values()).index(current_y)]
            
            if key.startswith("Image"):
                y[i] = 0
                
            elif key.startswith("Correct") or key.startswith("Incorrect"):
                y[i] = 1
        

        # SCALE THE DATA (-1 to 1) -> Maybe?? 
        # REASONS: 1) GAFs are sensitive to the range of the data, 2) the range of the data is different for each participant
        # problem: if we have some noisy data, then maybe min max normalisation is not the best, as the noise will be around the max and min, and maybe the actual signal will be around 0.
        # https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3
        #X = (2*X - X.min()) / (X.max() - X.min()) 

        # generate the GAFs
        gafs = generate_gafs(X, image_size=50)

        X_train, y_train, X_test, y_test = train_test_split(gafs, y)

        # save the GAFs
        np.save(outpath_sub / f'gafs_train.npy', X_train)
        np.save(outpath_sub / f'gafs_test.npy', X_test)

        # save the labels
        np.save(outpath_sub / f'labels_train.npy', y_train)
        np.save(outpath_sub / f'labels_test.npy', y_test)

if __name__ == '__main__':
    main()