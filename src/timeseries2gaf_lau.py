"""
    Converts the timeseries data into Gramian Angular Fields (both GAFD and GAFS), as well as Markov transition fields. For each timeseries a 3D array containing these are made. 
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.gaf import generate_gafs


def main():
    path = Path(__file__).parents[1]
    preprc_path = path / 'data' /'preprocessed_lau'
    outpath = path / 'data' / 'gaf_lau'

    # loop over subjects
    subjects = [x.name for x in preprc_path.iterdir()]

    # check that outpath exists
    if not outpath.exists():
        outpath.mkdir(parents=True)

    for subject in subjects:

        print(f"Converting timeseries to images for participant {subject}")

        # load the data
        X, y = np.load(preprc_path / subject / 'X.npy'), np.load(preprc_path / subject / 'y.npy')
        

        print(X.shape)
        print(y.shape)

        # SCALE THE DATA (-1 to 1) -> Maybe?? 
        # REASONS: 1) GAFs are sensitive to the range of the data, 2) the range of the data is different for each participant
        # problem: if we have some noisy data, then maybe min max normalisation is not the best, as the noise will be around the max and min, and maybe the actual signal will be around 0.
        # https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3
        #X = (2*X - X.min()) / (X.max() - X.min()) 

        # generate the GAFs
        gafs = generate_gafs(X, image_size=32, n_bins=4)
        print("-------------------")
        print(X.shape)
        print(y.shape)
        print(gafs.shape)

        # save the GAFs
        np.save(outpath / f'{subject}_gafs.npy', gafs)
        np.save(outpath / f'{subject}_labels.npy', y)

if __name__ == '__main__':
    main()