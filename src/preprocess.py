"""
Preprocessing pipeline for EEG data

Loops over the participants and saves the epochs.

Note that this is just a quick script made for you to have some data to play with! Refer to the preprocessing notebook if you want to see the steps in more detail.
"""

from pathlib import Path
import mne
import logging

def setup_logger():
    """
    Sets up the logger for the script
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    file_handler = logging.FileHandler("preprocessing.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger



if __name__ == "__main__":
    path = Path(__file__).parent

    data_path = Path("/work/raw/FaceWord")
    session_info_path = path / "session_info.txt"
    output_path = path.parent / "data" / "preprocessed"

    # make sure that the output path exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # load in session information (bad channels, etc.) txt file with dictionary
    with open(session_info_path, "r") as f:
        session_info = eval(f.read())

    logger = setup_logger()


    # defining the events
    event_id = {
        'Word/wPos': 11, # positive word
        'Wait/wPos': 31, # waiting time after positive word 
        'Image/wPos': 21, # positive image (always following pos word) 
        'Word/wNeg': 12, # negative word
        'Wait/wNeg': 32, # waiting time after negative word 
        'Image/wNeg': 22, # negative image (always following neg word) 
        'Word/wNeu': 13, # neutral word
        'Wait/wNeu/iPos': 51, # wait time after neu w (before pos i) 
        'Image/wNeu/iPos': 41, # positive image (after neu word) 
        'Wait/wNeu/iNeg': 52, # wait time after neu w (before neg i) 
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

    participant_paths = list(data_path.glob("*.vhdr"))

    # drop the one for group78
    participant_paths = [participant_path for participant_path in participant_paths if "13" not in participant_path.stem]

    for participant_path in participant_paths:

        participant_filename = participant_path.stem  
        participant = session_info[participant_filename]["nickname"]

        # loading in the raw data
        logging.info(f"Loading in raw data for participant {participant}")
        raw = mne.io.read_raw_brainvision(str(participant_path), preload=True)

        try:
            raw.set_channel_types({"EOG1": "eog", "EOG2": "eog"})
        except ValueError:
            try:
                raw.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
            except ValueError:
                pass

        if "41" in raw.ch_names:
            raw.set_channel_types({"41": "misc"})
            

        logger.info(f"Setting the montage for participant {participant}")
        raw.set_montage("standard_1020", on_missing="warn")

        # setting the reference
        logger.info(f"Setting the reference for participant {participant}")
        raw.set_eeg_reference("average", projection=False)

        # dropping the bad channels
        bad_channels = session_info[participant_filename]["bad_channels"]

        if session_info[participant_filename]["bad_channels"] != []:
            logger.info(f"Dropping the bad channels for participant {participant}. Bad channels: {bad_channels}")
            raw.info["bads"] = bad_channels
            
            # interpolate the bad channels (except for Fp1 and Fp2 which are bad for all participants)
            raw.interpolate_bads(reset_bads=True, mode="accurate", exclude=["Fp1", "Fp2"])

            # dropping the bad channels
            raw.drop_channels(raw.info["bads"])

        else:
            logger.info(f"No bad channels for participant {participant}")

    
        logger.info(f"Filtering the data for participant {participant}")
        raw.filter(0.1, 40)

        # creating the events
        events, _ = mne.events_from_annotations(raw)

        # remove events from event id that are not in the data (to avoid errors when creating the epochs)
        event_id_tmp = {key: value for key, value in event_id.items() if value in events[:, 2]}

        reject = {"eeg": 100e-6}

        picks = mne.pick_types(raw.info, eeg=True)
        # creating the epochs
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id=event_id_tmp,
            tmin=-0.2, 
            tmax=0.5, 
            baseline=(-0.2, 0), 
            preload=True, 
            reject=None,
            picks=picks
        )

        # downsample the epochs
        epochs = epochs.resample(250)

        # saving the epochs
        output_path_sub = output_path / participant 
        output_path_sub.mkdir(parents=True, exist_ok=True)
        epochs.save(output_path_sub / "epoch-epo.fif", overwrite=True)






