import os
import sys
import ast
import wfdb
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from data_utils.preprocessor import Preprocessor


class PTBXL_MI_Preprocessor(Preprocessor):
    def preprocess(self):
        self.data_dict = {}
        class_mappings = {
            "MI": "myocardial infarction",
            "NORM": "normal",
            "STTC": "ST/T segment change",
            "CD": "conduction disturbance",
            "HYP": "hypertrophy",
        }
        sampling_rate = 100
        data_path = os.path.join(
            self.raw_data_path,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        )
        ecg_data, raw_labels = self.load_dataset(data_path, sampling_rate)
        aggregated_labels = self.compute_label_aggregations(raw_labels, data_path)

        data, labels, classes = self.select_data(
            ecg_data, aggregated_labels, min_samples=0
        )

        data = data[:, :, 1:2]
        for idx, (d, l) in enumerate(zip(data, labels)):
            label = "normal"

            label_idx = l.tolist().index(1)
            cls_name = classes[label_idx]
            if cls_name == "MI" or cls_name == "NORM":
                label = class_mappings[cls_name]
            else:
                continue
                # label = "normal"
            domain = f"sample_{idx}"
            if domain not in self.data_dict:
                self.data_dict[domain] = []
            self.data_dict[domain].append((d, label))
        self.data_dict = self.normalize_data(self.data_dict)

    def load_dataset(self, path, sampling_rate):
        labels_df = pd.read_csv(
            os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id"
        )
        labels_df.scp_codes = labels_df.scp_codes.apply(ast.literal_eval)
        ecg_data = self.load_ecg_data(labels_df, sampling_rate, path)
        return ecg_data, labels_df

    def load_ecg_data(self, df, sampling_rate, path):
        filename_column = "filename_lr" if sampling_rate == 100 else "filename_hr"
        data = [
            wfdb.rdsamp(os.path.join(path, row[filename_column]))
            for _, row in tqdm(list(df.iterrows()))
        ]
        signals = np.array([signal for signal, _ in data])
        return signals

    def compute_label_aggregations(self, df, folder):
        agg_df = pd.read_csv(os.path.join(folder, "scp_statements.csv"), index_col=0)
        diag_agg_df = agg_df[agg_df.diagnostic == 1.0]

        def aggregate_diagnostic(y_dict):
            output = []
            for key in y_dict:
                if y_dict[key] == 0:
                    continue
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key, "diagnostic_class"]
                    if pd.notna(c):
                        output.append(c)
            return list(set(output))

        df["superdiagnostic"] = df.scp_codes.apply(aggregate_diagnostic)
        df["superdiagnostic_len"] = df.superdiagnostic.apply(len)

        return df

    def select_data(self, data, labels, min_samples):
        mlb = (
            MultiLabelBinarizer()
        )  # creates a binary label table for multilabel classification

        counts = pd.Series(
            np.concatenate(labels.superdiagnostic.values)
        ).value_counts()  # a dictionary of each superclass and the number of times it appears
        valid_labels = counts[counts > min_samples].index

        labels.superdiagnostic = labels.superdiagnostic.apply(
            lambda x: list(set(x).intersection(valid_labels))
        )  # remove superclasses that appear less than min_samples times
        labels["superdiagnostic_len"] = labels.superdiagnostic.apply(len)

        filtered_data = data[labels.superdiagnostic_len > 0]
        filtered_labels = labels[labels.superdiagnostic_len > 0]
        mlb.fit(
            filtered_labels.superdiagnostic.values
        )  # a column of the superclass for each sample and 1 or 0 is the value
        transformed_labels = mlb.transform(filtered_labels.superdiagnostic.values)
        classes = mlb.classes_

        return filtered_data, transformed_labels, classes
