import os
import sys
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy.io import loadmat

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from data_utils.preprocessor import Preprocessor


class UTDMHADPreprocessor(Preprocessor):
    def preprocess(self):
        self.data_dict = {}
        sampling_rate = 50
        win_len = sampling_rate * (self.win_len // self.sampling_rate)

        label_mapping = {
            "1": "Swipe left",
            "2": "Swipe right",
            "3": "Wave",
            "4": "Clap",
            "5": "Throw",
            "6": "Arm cross",
            "7": "Basketball shoot",
            "8": "Draw X",
            "9": "Draw circle (clockwise)",
            "10": "Draw circle (counter clockwise)",
            "11": "Draw triangle",
            "12": "Bowling",
            "13": "Boxing",
            "14": "Baseball swing",
            "15": "Tennis swing",
            "16": "Arm curl",
            "17": "Tennis serve",
            "18": "Push",
            "19": "Knock",
            "20": "Catch",
            "21": "Pickup and throw",
        }

        mat_files = glob(os.path.join(self.raw_data_path, "Inertial/*.mat"))
        for mat_file in tqdm(mat_files, desc="processing mat files", leave=False):
            filename = mat_file.split("/")[-1]
            label = filename.split("_")[0].strip("a")
            if label not in label_mapping.keys():
                continue
            label = label_mapping[label]
            subject = filename.split("_")[1].strip("s")

            # load mat file
            mat = loadmat(mat_file)
            window = mat["d_iner"][:, 0:3] * 9.8
            if window.shape[0] < win_len:
                window = np.pad(
                    window, ((0, win_len - window.shape[0]), (0, 0)), "constant"
                )
            else:
                window = window[:win_len, :]

            domain = f"user_{subject}"
            if domain not in self.data_dict:
                self.data_dict[domain] = []

            if sampling_rate != self.sampling_rate:
                window = self.resample_data(window, sampling_rate, self.sampling_rate)

            self.data_dict[domain].append((window, label))

        self.data_dict = self.normalize_data(self.data_dict)
