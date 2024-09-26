import os
import sys

import pandas as pd

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from data_utils.preprocessor import Preprocessor


class HHARPreprocessor(Preprocessor):
    def preprocess(self):
        self.data_dict = {}
        df = pd.read_csv(os.path.join(self.raw_data_path, "Watch_accelerometer.csv"))
        print("Successfully loaded the data.")
        device_sr = {"lgwatch": 200, "gear": 100}
        idx = 0
        while True:
            row = df.iloc[idx]
            model = row["Model"]
            win_duration = int(self.win_len // self.sampling_rate)
            hop_duration = int(self.hop_len // self.sampling_rate)
            win_size = device_sr[model] * win_duration
            hop_size = device_sr[model] * hop_duration

            if idx + win_size >= len(df):
                break

            user = row["User"]
            label = row["gt"]
            window = df.iloc[idx : idx + win_size]

            if (
                not (
                    window["User"].nunique() == 1
                    and window["Device"].nunique() == 1
                    and window["gt"].nunique() == 1
                )
                or df.iloc[idx + win_size - 1]["Creation_Time"]
                - df.iloc[idx]["Creation_Time"]
                > 1e9 * (win_duration) * 1.1
                or pd.isna(label)
            ):
                idx += hop_size
                continue

            domain = f"user_{user}"
            if domain not in self.data_dict:
                self.data_dict[domain] = []

            window = window.iloc[:, 3:6].values
            if device_sr[model] != self.sampling_rate:
                window = self.resample_data(
                    window, device_sr[model], self.sampling_rate
                )
            self.data_dict[domain].append((window, label))
            idx += hop_size

        self.data_dict = self.normalize_data(self.data_dict)
