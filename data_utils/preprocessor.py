import os
import json
import datasets
import numpy as np

from scipy.signal import resample
from typing import Dict, Tuple


class Preprocessor:
    def __init__(self, raw_data_path, out_dir, win_len, hop_len, sampling_rate):
        self.raw_data_path = raw_data_path
        self.out_dir = out_dir
        self.win_len = win_len
        self.hop_len = hop_len
        self.sampling_rate = sampling_rate
        self.data_dict = None

    def preprocess(self):
        """
        preprocess raw data to data_dict
        {domain(str): [(data(np.array), label(str)), ...], ...}
        """
        raise NotImplementedError

    def normalize_data(
        self, data: Dict[str, Tuple[np.array, str]]
    ) -> Dict[str, Tuple[np.array, str]]:
        new_data = {}
        for domain in data.keys():
            user_data = [window for window, _ in data[domain]]
            user_data = np.array(user_data)
            user_data = user_data.reshape(-1, user_data.shape[-1])
            mean = np.mean(user_data, axis=0)
            std = np.std(user_data, axis=0)
            new_data[domain] = [
                ((window - mean) / std, label) for window, label in data[domain]
            ]

        return new_data

    def resample_data(self, data: np.array, sr: int, target_sr: int) -> np.array:
        resampled_size = int(data.shape[0] * target_sr / sr)
        resampled_data = resample(data, resampled_size)

        return resampled_data

    def store_data(self) -> None:
        if self.data_dict is None:
            raise ValueError("data_dict is None. Preprocess data first.")

        domains = list(self.data_dict.keys())
        np.random.seed(0)
        np.random.shuffle(domains)

        splits = {"train": [], "val": [], "test": []}
        for i, domain in enumerate(domains):
            if i < int(len(domains) * 0.6):
                split = "train"
            elif i < int(len(domains) * 0.7):
                split = "val"
            else:
                split = "test"
            for data, label in self.data_dict[domain]:
                splits[split].extend(
                    [
                        dict(
                            data=data,
                            label=label,
                            domain=domain,
                        )
                    ]
                )

        for split, data_list in splits.items():
            save_dir = os.path.join(self.out_dir, "HF", split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            data_hf = datasets.Dataset.from_list(data_list)
            data_hf.save_to_disk(save_dir)

    def store_metadata(
        self,
        dataset: str,
        task: str,
        data_description: str,
        task_description: str,
        classes: list,
        channels: list,
        sampling_rate: int,
        txt_sampling_rate: int,
        duration: int,
        num_channels: int,
    ) -> None:
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        metadata = {
            "dataset": dataset,
            "task": task,
            "data_description": data_description,
            "task_description": task_description,
            "classes": classes,
            "channels": channels,
            "sampling_rate": sampling_rate,
            "txt_sampling_rate": txt_sampling_rate,
            "duration": duration,
            "num_channels": num_channels,
        }

        with open(
            os.path.join(self.out_dir, "meta_data.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f)
