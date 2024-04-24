import numpy as np
from tqdm import tqdm
from pyts.image import GramianAngularField, MarkovTransitionField


def trial_to_gaf(X:np.ndarray, image_size = None, n_bins = 4):
    """
    Transform a set of time series into a single image tensor using Gramian Angular Field (GAF) and 
    Markov Transition Field (MTF) techniques.

    Parameters:
    -----------
    X: np.ndarray
        The input time series

    image_size: int
        The size of the output image.

    Returns:
    --------
    np.ndarray
        The transformed image tensor
    """
    trans_s = GramianAngularField(method = 'summation', image_size=image_size)
    trans_d = GramianAngularField(method = 'difference', image_size=image_size)
    trans_m = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    
    # transform each trial
    X_gaf_s = trans_s.fit_transform(X)
    X_gaf_d = trans_d.fit_transform(X)
    X_mtf = trans_m.fit_transform(X)

    # loop over GAFs and MTF per channel
    im = np.stack([X_gaf_s[0], X_gaf_d[0], X_mtf[0]], axis=-1)[:, :, np.newaxis, :]

    for gaf_s, gaf_d, mtf in zip(X_gaf_s[1:], X_gaf_d[1:], X_mtf[1:]):
        gaf = np.stack([gaf_s, gaf_d, mtf], axis=-1)[:, :, np.newaxis, :]
        im = np.concatenate((im, gaf), axis=2)

    return im

def generate_gafs(X:np.array, image_size = None, n_bins = 4):
    """
    Converts the timeseries data into Gramian Angular Fields (GAFs) and maps them onto a image with 3 channels.

    Parameters
    ----------
    X : np.array
        The timeseries data with trials as the first dimension


    Returns
    -------
    np.array
        The GAFs for the timeseries data
    
    """
    n_trials = X.shape[0]
    n_sensors = X.shape[1]

    if image_size is None:
        image_size = X.shape[-1]

    
    gafs = np.zeros((n_trials, image_size, image_size, n_sensors, 3))

    # loop over trials
    for i, x in tqdm(enumerate(X), total=n_trials):
        gafs[i] = trial_to_gaf(x, image_size=image_size, n_bins=n_bins)

    return gafs