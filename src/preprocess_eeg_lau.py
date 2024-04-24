"""
This script preprocesses the EEG data for each subject. 
    - Set the reference to common average
    - Filters the data between 0.1 and 40 Hz.
    - Splits the data into epochs of 500 ms.
"""

import mne
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_data(eeg_path, event_path):
    """
    Loads in the raw EEG data and the event information.

    Parameters
    ----------
    eeg_path : Path
        Path to the eeg file.
    event_path : Path
        Path to the tsv file with event information.
    
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data.
    events : list
        Array with event information.
    """
    raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
    event_df =  pd.read_csv(event_path, sep = '\t')

    # mapping between the second level label and the assigned event id
    event_id = return_eventids()

    events = []
    for _, row in event_df.iterrows():
        try:
            new_event = [row.Sample, 0, event_id[row.Condition]]
            events.append(new_event)
        except KeyError:
            print(f"Event {row.Condition} not found in the event_id mapping.")
            continue

    return raw, events

def return_eventids():

    return {
        'standard_1': 1, 'standard_2': 1, 'standard_3': 1, 'standard_4': 1, 'standard_5': 1, 
        'omission_4': 2, 'omission_5': 2, 'omission_6': 2,
        'non_stimulation': 3
        }

def preprocess_meg(raw, events):
    """
    Performs the following preprocessing steps on the raw EEG data.
    - Filters the data between 1 and 40 Hz.
    - Splits the data into epochs of 500 ms.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    events : list
        List of events.
    
    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epoched EEG data. 
    """
    
    picks = mne.pick_types(raw.info, meg="mag", eeg=False, eog=False, stim=False)

    # high and low pass filtering of the data (low-pass at 40 hz to avoid 50 hz power line interference, high-pass at 0.1 Hz)
    raw.filter(l_freq = 0.1, h_freq = 40, verbose=False)

    # crop the raw minutes to the first 5 minutes
    raw.crop(tmax=60*5, include_tmax=False)

    # epoch data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, proj=True, picks=picks, baseline=None, preload=True, verbose=False)
    
    return epochs

def preprocess_subject(sub_path:Path):
    """
    Preprocesses the EEG data for a single subject and saves the data.

    Parameters
    ----------
    sub_path : Path
        Path to the subject directory.
    """

    path = Path(__file__)
    out_path = path.parents[1] / 'data' / 'preprocessed_lau' / sub_path.name

    # create output directory if it does not exist
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # get path for the fif file
    fif_path = sub_path / "ses-meg" / "meg" / "oddball_absence-tsss-mc_meg.fif"

    # tsv file with event information
    event_path = sub_path / "ses-meg" / "meg" / f"{sub_path.name}_oddball_absence-tsss-mc_meg.fif_run_01_events.tsv"

    X_path = out_path / f'X.npy'
    y_path = out_path / f'y.npy'
        
    raw, events = get_data(fif_path, event_path)

    epochs = preprocess_meg(raw, events)

    # save epochs as numpy array
    X = epochs.get_data(copy = True) # copy = True to avoid future warning
    y = epochs.events[:, -1]

    np.save(X_path, X)
    np.save(y_path, y)

    print(f"Subject {sub_path.name} preprocessed.")

def main():
    path = Path(__file__)

    bids_path = path.parents[1] / 'data' / 'raw_lau'

    subjects = [x for x in bids_path.iterdir() if x.is_dir()]
    subjects = [subject for subject in subjects if subject.name.startswith("sub-")]

    for subject in tqdm(subjects):
        preprocess_subject(subject)


if __name__ == '__main__':
    main()