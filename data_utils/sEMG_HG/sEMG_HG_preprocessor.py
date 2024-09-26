import os
import sys
import numpy as np
import pandas as pd

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Any

from data_utils.preprocessor import Preprocessor


class sEMG_HG_Preprocessor(Preprocessor):
    def preprocess(self):
        user_files = glob(f"{self.raw_data_path}/*.csv")
        self.label_map = {
            0: "Rest",
            1: "Extension",
            2: "Flexion",
            3: "Ulnar Deviation",
            4: "Radial Deviation",
            5: "Grip",
            6: "Abduction of Fingers",
            7: "Adduction of Fingers",
            8: "Supination",
            9: "Pronation",
        }

        self.data_dict = {}

        for user_idx, user_file in tqdm(enumerate(user_files)):
            user_data = pd.read_csv(user_file)

            total_length = 640 * self.sampling_rate
            if user_data.shape[0] < total_length:

                pad = np.zeros((total_length - user_data.shape[0], user_data.shape[1]))
                user_data = np.concatenate((user_data, pad), axis=0)

            data_label = self.segement_user_data(user_data)

            print(f"number of windows per user {len(data_label)}")
            user = f"user_{user_idx + 1}"

            if user not in self.data_dict:
                self.data_dict[user] = []

            self.data_dict[user].extend(data_label)

        self.data_dict = self.normalize_data(self.data_dict)
        print("the shape of each window", self.data_dict["user_1"][0][0].shape)

    def segement_user_data(self, user_data: np.ndarray) -> List[Tuple[np.ndarray, str]]:

        user_rest = 30 * self.sampling_rate
        user_activity = 104 * self.sampling_rate

        num_activities = 10
        num_cycles = 5

        data_label = []

        for i in range(num_cycles):

            cycle_idx = i * (user_activity + user_rest)
            cycle_data = user_data[cycle_idx : cycle_idx + user_activity]

            cycle_rest_len = 4 * self.sampling_rate
            cycle_activity_len = 6 * self.sampling_rate

            for activity in range(num_activities):

                start_idx = (
                    activity * (cycle_activity_len + cycle_rest_len) + cycle_rest_len
                )
                cycle_activity_data = cycle_data[
                    start_idx : start_idx + cycle_activity_len
                ]

                cycle_activity_label = self.label_map[activity]

                win_size = int(self.win_len)
                step_size = int(self.hop_len)

                oversampled_data = self.oversample_with_step(
                    cycle_activity_data, cycle_activity_label, win_size, step_size
                )

                data_label.extend(oversampled_data)

        return data_label

    def oversample_with_step(
        self, data: np.ndarray, label: str, win_size: int, step_size: int
    ) -> List[Tuple[np.ndarray, str]]:

        data_label = []

        for i in range(0, len(data) + 1 - win_size, step_size):
            data_label.append((data[i : i + win_size], label))

        return data_label
