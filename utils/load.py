from pathlib import Path
import numpy as np

def load_gaf(file: Path):
    """
    Load a gaf image from path and return it as a numpy array

    Parameters
    ----------
    file : Path
        Path to gaf
    
    Returns
    -------
    gaf : np.array
        The gaf image
    label : int
        Label indicating the label (animate or inanimate)
    """
    
    gaf = np.load(file)
    label = str(file)[-5]
    
    return gaf, int(label)


def load_gafs(gaf_path: Path, n_jobs: int = 1, all_subjects=False):
    """
    Loads gaf images from path and return them as a numpy array using multiprocessing

    Parameters
    ----------
    gaf_path : Path
        Path to gaf images
    n_jobs : int, optional
        Number of jobs to use for multiprocessing, by default 1
    
    Returns
    -------
    gafs : np.array
        The gaf images
    labels : np.array
        Labels indicating the label (animate or inanimate) for each gaf image
    """
    gafs = []
    labels = []

    files = list(gaf_path.iterdir())

    if all_subjects:
        files = [list(dir.iterdir()) for dir in files]
        files = [item for sublist in files for item in sublist]
        files = random.choices(files, k=10000) # choosing 50000 random trials to avoid memory overload

    
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            for gaf, label in tqdm(pool.imap(load_gaf, files), total=len(files), desc="Loading in data"):
                gafs.append(gaf)
                labels.append(label)
    else:
        for file in tqdm(files, desc="Loading in data"):
            gaf, label = load_gaf(file)
            gafs.append(gaf)
            labels.append(label)
    
    return np.array(gafs), np.array(labels)

