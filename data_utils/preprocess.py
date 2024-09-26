import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from data_utils.HHAR.hhar_preprocessor import HHARPreprocessor
from data_utils.Swimming.swimming_preprocessor import SwimmingPreprocessor
from data_utils.UTDMHAD.utdmhad_preprocessor import UTDMHADPreprocessor
from data_utils.PTBXL_CD.ptbxl_cd_preprocessor import PTBXL_CD_Preprocessor
from data_utils.PTBXL_HYP.ptbxl_hyp_preprocessor import PTBXL_HYP_Preprocessor
from data_utils.PTBXL_MI.ptbxl_mi_preprocessor import PTBXL_MI_Preprocessor
from data_utils.PTBXL_STTC.ptbxl_sttc_preprocessor import PTBXL_STTC_Preprocessor
from data_utils.sEMG_HG.sEMG_HG_preprocessor import sEMG_HG_Preprocessor
from data_utils.WESAD.wesad_preprocessor import WESADPreprocessor


def preprocess(dataset: str, out_dir: str = "path_to_save_preprocessed_data") -> None:
    """Preprocess the dataset."""
    if dataset == "WESAD":
        task = "Emotion recognition"
        raw_data_path = "path_to_raw_data"
        sampling_rate = 700
        win_len = 30 * sampling_rate
        hop_len = 30 * sampling_rate
        txt_sampling_rate = 30
        num_channels = 1
        classes = ["baseline", "stress", "amusement"]
        channels = ["Chest respiration"]
        data_description = f"The signal is collected from a chest-worn respiration sensing device.\
The data is collected over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."
        task_description = f"a task for classifying signal measured from respiration sensor into {len(classes)} categories: {', '.join(classes)} each showing the emotion level of the user."

        preprocessor = WESADPreprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "sEMG_HG":
        task = "Hand gesture recognition"
        raw_data_path = "path_to_raw_data"
        sampling_rate = 2000
        win_len = 0.2 * sampling_rate
        hop_len = 0.2 * sampling_rate
        txt_sampling_rate = 10
        num_channels = 4
        classes = [
            "Rest",
            "Extension",
            "Flexion",
            "Ulnar Deviation",
            "Radial Deviation",
            "Grip",
            "Abduction of Fingers",
            "Adduction of Fingers",
            "Supination",
            "Pronation",
        ]
        channels = ["EMG1", "EMG2", "EMG3", "EMG4"]
        data_description = f"The electromyographic sensor data is collected from an sEMG measuring 4-channel armband. \
The data is collected over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."
        task_description = f"a task for classifying a 4-channel sEMG data collected form the forearm into {len(classes)} hand gestures: {', '.join(classes)}."

        preprocessor = sEMG_HG_Preprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "PTB-XL-CD":
        task = "Conduction disturbance detection"
        raw_data_path = "path_to_raw_data"
        win_len = 1000
        hop_len = 0
        sampling_rate = 100
        txt_sampling_rate = 100
        num_channels = 1
        classes = ["conduction disturbance", "normal"]
        channels = ["lead II"]
        data_description = f"The ECG data is collected from a lead II ECG sensor. \
The ECG data is recorded over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."
        task_description = f"a task for classifying ECG data into {len(classes)} categories: {', '.join(classes)}."

        preprocessor = PTBXL_CD_Preprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "PTB-XL-HYP":
        task = "Hypertrophy detection"
        raw_data_path = "path_to_raw_data"
        win_len = 1000
        hop_len = 0
        sampling_rate = 100
        txt_sampling_rate = 100
        num_channels = 1
        classes = ["hypertrophy", "normal"]
        channels = ["lead II"]
        data_description = f"The ECG data is collected from a lead II ECG sensor. \
The ECG data is recorded over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."

        task_description = f"a task for classifying ECG data into {len(classes)} categories: {', '.join(classes)}."

        preprocessor = PTBXL_HYP_Preprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "PTB-XL-MI":
        task = "Myocardial infarction detection"
        # task = "Arrhythmia diagnosis"
        raw_data_path = "path_to_raw_data"
        win_len = 1000
        hop_len = 0
        sampling_rate = 100
        txt_sampling_rate = 100
        num_channels = 1
        classes = ["myocardial infarction", "normal"]
        channels = ["lead II"]
        data_description = f"The ECG data is collected from a lead II ECG sensor. \
The ECG data is recorded over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."
        task_description = f"a task for classifying ECG data into {len(classes)} categories: {', '.join(classes)}."

        preprocessor = PTBXL_MI_Preprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "PTB-XL-STTC":
        task = "ST-T change detection"
        raw_data_path = "path_to_raw_data"
        win_len = 1000
        hop_len = 0
        sampling_rate = 100
        txt_sampling_rate = 100
        num_channels = 1
        classes = ["ST/T segment change", "normal"]
        channels = ["lead II"]
        data_description = f"The ECG data is collected from a lead II ECG sensor. \
The ECG data is recorded over {win_len//sampling_rate} seconds. The data is normalized with the statistics of the user's data."
        task_description = f"a task for classifying ECG data into {len(classes)} categories: {', '.join(classes)}."

        preprocessor = PTBXL_STTC_Preprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "HHAR":
        task = "Human activity recognition"
        raw_data_path = "path_to_raw_data"
        win_len = 500
        hop_len = 250
        sampling_rate = 100
        txt_sampling_rate = 10
        num_channels = 3
        # classes = ["bike", "sit", "stand", "walk"]
        channels = ["X-axis", "Y-axis", "Z-axis"]
        classes = ["bike", "sit", "stand", "walk", "stairsdown", "stairsup"]
        data_description = f"The sensor data is collected from an accelerometer measuring \
acceleration along the x, y, and z axes. The data is normalized with the statistics of the user's data.\
The data is collected over {win_len//sampling_rate} seconds. The data is measured from a smartwatch which was attached to the wrist of a user."
        task_description = f"a task for classifying {len(classes)} human activities\
, {', '.join(classes)}, using three-axis accelerometer data measured from a wrist-worn smartwatch."
        preprocessor = HHARPreprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "Swimming":
        task = "Swimming style recognition"
        raw_data_path = "path_to_raw_data"
        win_len = 180
        hop_len = 90
        sampling_rate = 30
        txt_sampling_rate = 10
        num_channels = 3
        classes = ["backstroke", "breaststroke", "butterfly", "freestyle", "stationary"]
        channels = ["X-axis", "Y-axis", "Z-axis"]
        data_description = f"The sensor data is collected from an accelerometer measuring \
acceleration along the x, y, and z axes. The data is normalized with the statistics of the user's data.\
The data is collected over {win_len//sampling_rate} seconds. \
The data is measured from a smartwatch which was attached to the wrist of a user."
        task_description = f"a task for classifying {len(classes)} swimming styles\
, {', '.join(classes)}, using three-axis accelerometer data measured from a wrist-worn \
smartwatch equipped by swimmers."

        preprocessor = SwimmingPreprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    elif dataset == "UTD-MHAD":
        task = "gesture recognition"
        raw_data_path = "path_to_raw_data"
        win_len = 150
        hop_len = 75
        sampling_rate = 50
        txt_sampling_rate = 10
        num_channels = 3
        classes = [
            "Swipe left",
            "Swipe right",
            "Wave",
            "Clap",
            "Throw",
            "Arm cross",
            "Basketball shoot",
            "Draw X",
            "Draw circle (clockwise)",
            "Draw circle (counter clockwise)",
            "Draw triangle",
            "Bowling",
            "Boxing",
            "Baseball swing",
            "Tennis swing",
            "Arm curl",
            "Tennis serve",
            "Push",
            "Knock",
            "Catch",
            "Pickup and throw",
        ]
        channels = ["X-axis", "Y-axis", "Z-axis"]
        data_description = f"The sensor data is collected from an accelerometer measuring \
acceleration along the x, y, and z axes. The data is normalized with the statistics of the user's data.\
The data is collected over {win_len//sampling_rate} seconds. \
The data is measured from a smartwatch which was attached to the wrist of a user."
        task_description = f"a task for classifying {len(classes)} gestures\
, {', '.join(classes)}, using three-axis accelerometer data measured from a wrist-worn smartwatch."

        preprocessor = UTDMHADPreprocessor(
            raw_data_path=raw_data_path,
            out_dir=os.path.join(out_dir, dataset),
            win_len=win_len,
            hop_len=hop_len,
            sampling_rate=sampling_rate,
        )

    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

    preprocessor.preprocess()
    preprocessor.store_data()
    preprocessor.store_metadata(
        dataset=dataset,
        task=task,
        data_description=data_description,
        task_description=task_description,
        classes=classes,
        channels=channels,
        sampling_rate=sampling_rate,
        txt_sampling_rate=txt_sampling_rate,
        duration=win_len / sampling_rate,
        num_channels=num_channels,
    )


if __name__ == "__main__":
    datasets = [
        "HHAR",
        "Swimming",
        "UTD-MHAD",
        "PTB-XL-CD",
        "PTB-XL-HYP",
        "PTB-XL-MI",
        "PTB-XL-STTC",
        "sEMG_HG",
        "WESAD",
    ]
    for dataset_ in datasets:
        preprocess(dataset_)
