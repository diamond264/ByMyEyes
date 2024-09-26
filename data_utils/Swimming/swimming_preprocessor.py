import os
import sys
import pandas as pd

from glob import glob
from tqdm import tqdm

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from data_utils.preprocessor import Preprocessor


class SwimmingPreprocessor(Preprocessor):
    def preprocess(self):
        self.data_dict = {}
        sampling_rate = 30
        win_len = sampling_rate * (self.win_len // self.sampling_rate)
        hop_len = sampling_rate * (self.hop_len // self.sampling_rate)
        label_mapping = {
            0: "stationary",
            1: "freestyle",
            2: "breaststroke",
            3: "backstroke",
            4: "butterfly",
            5: "stationary",
        }

        for user_dir in tqdm(
            glob(f"{self.raw_data_path}/*"), desc="processing per user", leave=False
        ):
            if user_dir.endswith("readme.txt"):
                continue
            data_files = glob(f"{user_dir}/*.csv")
            for data_file in data_files:
                df = pd.read_csv(data_file)

                idx = 0
                while True:
                    row = df.iloc[idx]
                    label = row["label"]

                    if idx + win_len >= len(df):
                        break
                    window = df.iloc[idx : int(idx + win_len)]
                    if (
                        not window["label"].nunique() == 1
                        or df.iloc[idx + win_len - 1]["timestamp"]
                        - df.iloc[idx]["timestamp"]
                        > 1e9 * (self.win_len // self.sampling_rate) * 1.5
                        or pd.isna(label)
                        or not label in label_mapping.keys()
                    ):
                        idx += hop_len
                        continue

                    domain = f"user_{user_dir.split('/')[-1]}"
                    if domain not in self.data_dict:
                        self.data_dict[domain] = []

                    window = window.iloc[:, 2:5].values
                    if sampling_rate != self.sampling_rate:
                        window = self.resample_data(
                            window, sampling_rate, self.sampling_rate
                        )

                    label = label_mapping[label]
                    self.data_dict[domain].append((window, label))
                    idx += hop_len

        self.data_dict = self.normalize_data(self.data_dict)
