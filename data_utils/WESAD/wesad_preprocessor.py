import os
import sys
import numpy as np
import pandas as pd

from glob import glob
from typing import List, Tuple

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from data_utils.preprocessor import Preprocessor


class WESADPreprocessor(Preprocessor):
    def preprocess(self):
        self.data_dict = {}

        users = glob(self.raw_data_path + "/S*")
        for user in users:
            user_id = user.split("/")[-1]
            data_path = os.path.join(user, f"{user_id}.pkl")

            user_data = pd.read_pickle(data_path)
            user_eda_dat = user_data["signal"]["chest"]["Resp"]
            user_label = user_data["label"]

            data_label = self.window_user_data(user_eda_dat, user_label)

            self.data_dict[user_id] = data_label

        self.data_dict = self.normalize_data(self.data_dict)

    def window_user_data(
        self, user_data: np.ndarray, user_label: np.ndarray
    ) -> List[Tuple[np.ndarray, str]]:
        label_map = {1: "baseline", 2: "stress", 3: "amusement"}
        data_label = []
        win_size = int(self.win_len)
        hop_size = int(self.hop_len)

        for idx in range(0, len(user_data) + 1 - win_size, hop_size):
            window = user_data[idx : idx + win_size]
            label_window = user_label[idx : idx + win_size]

            label = self.choose_label(label_window)
            if label == -1 or label not in label_map:
                continue

            label = label_map[label]

            data_label.append((window, label))

        return data_label

    def choose_label(self, label_window: np.ndarray) -> int:
        label_threshold = 0.99
        label = np.bincount(label_window).argmax()
        label_count = np.bincount(label_window)[label]

        if label_count / len(label_window) >= label_threshold:
            return label

        return -1
