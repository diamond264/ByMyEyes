import os
import sys
import numpy as np
import neurokit2 as nk
from scipy.signal import resample

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, ".."))

import core.vis.ecg as ecg


class TextGenerator:
    def __init__(self, channels, sr, style, rp, txt_sr):
        self.channels = channels
        self.sr = sr
        self.style = style
        self.rp = rp
        self.txt_sr = txt_sr

    def resample(self, data, sr_from=None, sr_to=None):
        if sr_from and sr_to:
            return resample(data, int(len(data) * sr_to / sr_from))
        return resample(data, int(len(data) * self.txt_sr / self.sr))

    def raw_waveform(self, data):
        txt = f"Given sensor data (list of {self.channels}): "
        if self.sr != self.txt_sr:
            data = self.resample(data)

        if len(self.channels) == 1:
            data = data[:, 0]
            txt += f"[{', '.join([str(np.round(i, self.rp)) for i in data])}]"
            return txt

        samples = []
        for i in range(len(data)):
            samples.append(
                f"[{', '.join([str(np.round(j, self.rp)) for j in data[i]])}]"
            )
        txt += f"[{', '.join(samples)}]"
        return txt

    def psd(self, data):
        txt = "Given sensor data PSD (list of (frequency, density) by channels): "
        if self.sr != self.txt_sr:
            data = self.resample(data)

        psd_samples = []
        for i, channel in enumerate(self.channels):
            psd = nk.signal_psd(data[:, i], sampling_rate=self.sr, method="fft")
            psd_sample = []
            for freq, power in zip(psd["Frequency"], psd["Power"]):
                psd_sample.append(
                    f"({np.round(freq, self.rp)}, {np.round(power, self.rp)})"
                )
            psd_samples.append(f"{channel}: [{', '.join(psd_sample)}]")

        txt += f"[{', '.join(psd_samples)}]"
        return txt

    def ecg_signal(self, data):
        signal, _ = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        data = signal["ECG_Clean"].values
        txt = "Cleaned ECG signal: "
        if self.sr != self.txt_sr:
            data = self.resample(data)

        txt += f"[{', '.join([str(np.round(i, self.rp)) for i in data])}]"
        return txt

    def ecg_hr(self, data):
        signal, info = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        rate = signal["ECG_Rate"].values
        peaks = info["ECG_R_Peaks"]

        txt = f"Heart rate in the ECG signal (mean value {np.round(np.mean(rate), self.rp)}): "
        txt += f"[{', '.join([str(np.round(i, self.rp)) for i in rate])}]\n"
        txt += "R-peaks in the ECG signal (list of the index): "
        txt += f"[{', '.join([str(i) for i in peaks])}]"
        return txt

    def ecg_ind(self, data):
        signal, info = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        heartbeats, waves = ecg.ecg_hb_features(signal, info)
        heartbeats = heartbeats["ECG_Clean"].values
        txt = f"Average heartbeat in the ECG signal (list of {self.channels}): "
        txt += f"[{', '.join([str(np.round(i, self.rp)) for i in heartbeats])}]\n"
        for k, vals in waves.items():
            txt += f"{k} in the ECG signal (list of (index, value)): "
            peaks = []
            for idx, v in vals:
                peaks.append(f"({idx}, {np.round(v, self.rp)})")
            txt += f"[{', '.join(peaks)}]\n"
        return txt

    def emg_signal(self, data):
        data_list = []
        for i in range(len(data[0])):
            signal, _ = nk.emg_process(data[:, i], sampling_rate=self.sr)
            signal = signal["EMG_Clean"]
            if self.sr != self.txt_sr:
                signal = self.resample(signal)
            data_list.append(signal)
        txt = f"Cleaned EMG signal (list of {self.channels}): "
        data_ = []
        for i in range(len(data_list[0])):
            aggregated_channel = []
            for d in data_list:
                aggregated_channel.append(d[i])
            data_.append(
                "["
                + ", ".join([str(np.round(i, self.rp)) for i in aggregated_channel])
                + "]"
            )
        txt += f"[{', '.join(data_)}]"
        return txt

    def emg_ma(self, data):
        data_list = []
        for i in range(len(data[0])):
            signal, _ = nk.emg_process(data[:, i], sampling_rate=self.sr)
            signal = signal["EMG_Amplitude"]
            if self.sr != self.txt_sr:
                signal = self.resample(signal)
            data_list.append(signal)
        txt = f"EMG muscle activation (list of {self.channels}): "
        data_ = []
        for i in range(len(data_list[0])):
            aggregated_channel = []
            for d in data_list:
                aggregated_channel.append(d[i])
            data_.append(
                "["
                + ", ".join([str(np.round(i, self.rp)) for i in aggregated_channel])
                + "]"
            )
        txt += f"[{', '.join(data_)}]"
        return txt

    def gen_txt(self, data, label=None):
        txt = ""
        if not label is None:
            txt += f"*Example of {label}*:\n"
        if self.style == "raw waveform":
            txt += f"{self.raw_waveform(data)}"
        elif self.style == "spectrogram":
            # spectrograms cannot be expressed in text
            txt += f"{self.raw_waveform(data)}"
        elif self.style == "signal power spectrum density":
            txt += f"{self.psd(data)}"
        elif self.style == "ECG signal and peaks":
            txt += f"{self.ecg_signal(data)}"
        elif self.style == "ECG heart rate":
            txt += f"{self.ecg_hr(data)}"
        elif self.style == "ECG individual heart beats":
            txt += f"{self.ecg_ind(data)}"
        elif self.style == "EMG signal":
            txt += f"{self.emg_signal(data)}"
        elif self.style == "EMG muscle activation":
            txt += f"{self.emg_ma(data)}"

        # The following visualizations were not selected from our visualization tool filtering
        # The results can be reproduced without the functions

        # elif self.style == "EDA signal":
        #     txt += f"{self.eda_signal(data)}"
        # elif self.style == "EDA skin conductance response (SCR)":
        #     txt += f"{self.eda_scr(data)}"
        # elif self.style == "EDA skin conductance level (SCL)":
        #     txt += f"{self.eda_scl(data)}"
        # elif self.style == "RSP signal":
        #     txt += f"{self.rsp_signal(data)}"
        # elif self.style == "RSP breathing rate":
        #     txt += f"{self.rsp_br(data)}"
        # elif self.style == "RSP breathing amplitude":
        #     txt += f"{self.rsp_ba(data)}"
        # elif self.style == "RSP respiratory volume per time":
        #     txt += f"{self.rsp_vpt(data)}"
        # elif self.style == "RSP cycle symmetry":
        #     txt += f"{self.rsp_cs(data)}"

        else:
            txt += f"{self.raw_waveform(data)}"
        return txt
