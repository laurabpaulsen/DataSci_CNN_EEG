"""
    Converts the timeseries data into Gramian Angular Fields (both GAFD and GAFS), as well as Markov transition fields. For each timeseries a 3D array containing these are made. 
    The GAFs are saved as numpy arrays in the data/gaf folder.
"""

import numpy as np
from pathlib import Path
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.gaf import generate_gafs


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
        # load the data
        X, y = np.load(preprc_path / subject / 'X.npy'), np.load(preprc_path / subject / 'y.npy')

        # generate the GAFs
        gafs, y = generate_gafs(X, y)

        # save the GAFs
        np.save(outpath / f'{subject}_gafs.npy', gafs)
        np.save(outpath / f'{subject}_labels.npy', y)

if __name__ == '__main__':
    main()