import os
import sys
import base64
import matplotlib

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, ".."))

import numpy as np
from io import BytesIO
import neurokit2 as nk
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

import core.vis.ecg as ecg
import core.vis.emg as emg
import core.vis.rsp as rsp

matplotlib.use("Agg")


class Visualizer:
    def __init__(self, channels, sampling_rate, plot, args):
        self.channels = channels
        self.sr = sampling_rate
        self.plot = plot
        self.args = args

        if plot == "spectrogram":
            fig, canvas = plt.subplots(
                len(channels), 1, figsize=(5, 1 + 2 * len(channels))
            )
        else:
            fig, canvas = plt.subplots(1, 1, figsize=(5, 4))
        self.fig = fig
        self.canvas = canvas
        self.resize(512)

    def close(self):
        plt.close(self.fig)

    def resize(self, max_size):
        # Get the original size of the figure
        orig_width, orig_height = self.fig.get_size_inches()

        # Calculate the aspect ratio
        aspect_ratio = orig_width / orig_height

        # Set the largest dimension to 512 pixels while maintaining the aspect ratio
        if orig_width > orig_height:
            new_width = max_size / self.fig.dpi
            new_height = new_width / aspect_ratio
        else:
            new_height = max_size / self.fig.dpi
            new_width = new_height * aspect_ratio

        # Set the new figure size
        self.fig.set_size_inches(new_width, new_height)

    def plot_waveform(self, data, **kwargs):
        self.canvas.cla()
        self.canvas.set_title(self.plot)
        self.canvas.set_xlabel("Time [sec]")
        self.canvas.set_ylabel("Normalized value")
        for i, channel in enumerate(self.channels):
            self.canvas.plot(np.arange(len(data)) / self.sr, data[:, i], label=channel)
        self.canvas.legend()
        self.canvas.set_ylim(kwargs["ylim"])

    def plot_spectrogram(self, data, **kwargs):
        for i, c in enumerate(self.channels):
            if len(self.channels) == 1:
                canvas = self.canvas
            else:
                canvas = self.canvas[i]
            canvas.cla()
            canvas.set_title(self.plot + " of " + c)
            canvas.set_xlabel("Time [sec]")
            canvas.set_ylabel("Frequency [Hz]")
            frequencies, times, Sxx = spectrogram(
                data[:, i],
                fs=self.sr,
                noverlap=kwargs["noverlap"],
                nfft=kwargs["nfft"],
                nperseg=kwargs["nperseg"],
                mode=kwargs["mode"],
            )
            canvas.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading="gouraud")

    def plot_psd(self, data):
        self.canvas.cla()
        self.canvas.set_title(self.plot)
        self.canvas.set_xlabel("Frequency [Hz]")
        self.canvas.set_ylabel("Power [dB]")
        for i, channel in enumerate(self.channels):
            psd = nk.signal_psd(data[:, i], sampling_rate=self.sr, method="fft")
            self.canvas.loglog(psd["Frequency"], psd["Power"], label=channel)
        self.canvas.legend()

    def plot_ecg(self, data):
        self.canvas.cla()
        signals, info = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        ecg.ecg_plot_signal(signals, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_ecg_hr(self, data):
        self.canvas.cla()
        signals, info = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        ecg.ecg_plot_hr(signals, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_ecg_ind(self, data):
        self.canvas.cla()
        signals, info = nk.ecg_process(data[:, 0], sampling_rate=self.sr)
        ecg.ecg_plot_ind(signals, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_emg(self, data):
        self.canvas.cla()
        for i, channel in enumerate(self.channels):
            signals, info = nk.emg_process(data[:, i], sampling_rate=self.sr)
            emg.emg_plot_signal(signals, info=info, ax=self.canvas, label=channel)
        self.canvas.legend()
        self.canvas.set_title(self.plot)

    def plot_emg_act(self, data):
        self.canvas.cla()
        for i, channel in enumerate(self.channels):
            signals, info = nk.emg_process(data[:, i], sampling_rate=self.sr)
            emg.emg_plot_act(signals, info=info, ax=self.canvas, label=channel)
        self.canvas.legend()
        self.canvas.set_title(self.plot)

    def plot_rsp(self, data):
        self.canvas.cla()
        signal, info = nk.rsp_process(data[:, 0], sampling_rate=self.sr)
        rsp.rsp_plot_signal(signal, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_rsp_br(self, data):
        self.canvas.cla()
        signal, info = nk.rsp_process(data[:, 0], sampling_rate=self.sr)
        rsp.rsp_plot_br(signal, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_rsp_ba(self, data):
        self.canvas.cla()
        signal, info = nk.rsp_process(data[:, 0], sampling_rate=self.sr)
        rsp.rsp_plot_ba(signal, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_rsp_vpt(self, data):
        self.canvas.cla()
        signal, info = nk.rsp_process(data[:, 0], sampling_rate=self.sr)
        rsp.rsp_plot_vbt(signal, info, self.canvas)
        self.canvas.set_title(self.plot)

    def plot_rsp_cs(self, data):
        self.canvas.cla()
        signal, info = nk.rsp_process(data[:, 0], sampling_rate=self.sr)
        rsp.rsp_plot_cs(signal, info, self.canvas)
        self.canvas.set_title(self.plot)

    # The following visualizations were not selected from our visualization tool filtering
    # FYI, the results in the paper can be reproduced without the functions
    def plot_eda(self, data):
        pass

    def plot_eda_scr(self, data):
        pass

    def plot_eda_scl(self, data):
        pass

    def gen_b64_img(self, data, label=None):
        data = np.array(data)
        if label is None:
            label = "target data"
        plt.suptitle(label, fontsize=20)

        if self.plot == "raw waveform":
            self.plot_waveform(data, **self.args)
        elif self.plot == "spectrogram":
            self.plot_spectrogram(data, **self.args)
        elif self.plot == "signal power spectrum density":
            self.plot_psd(data)
        elif self.plot == "ECG signal and peaks":
            self.plot_ecg(data)
        elif self.plot == "ECG heart rate":
            self.plot_ecg_hr(data)
        elif self.plot == "ECG individual heart beats":
            self.plot_ecg_ind(data)
        elif self.plot == "EMG signal":
            self.plot_emg(data, **self.args)
        elif self.plot == "EMG muscle activation":
            self.plot_emg_act(data, **self.args)
        elif self.plot == "EDA signal":
            self.plot_eda(data)
        elif self.plot == "EDA skin conductance response (SCR)":
            self.plot_eda_scr(data)
        elif self.plot == "EDA skin conductance level (SCL)":
            self.plot_eda_scl(data)
        elif self.plot == "RSP signal":
            self.plot_rsp(data)
        elif self.plot == "RSP breathing rate":
            self.plot_rsp_br(data)
        elif self.plot == "RSP breathing amplitude":
            self.plot_rsp_ba(data)
        elif self.plot == "RSP respiratory volume per time":
            self.plot_rsp_vpt(data)
        elif self.plot == "RSP cycle symmetry":
            self.plot_rsp_cs(data)

        # The following visualizations were not selected from our visualization tool filtering
        # The results can be reproduced without the functions

        # elif self.plot == "PPG signal and peaks":
        #     self.plot_waveform(self.canvas, data, **args)
        # elif self.plot == "PPG heart rate":
        #     self.plot_waveform(self.canvas, data, **args)
        # elif self.plot == "PPG individual heart beats":
        #     self.plot_waveform(self.canvas, data, **args)
        # elif self.plot == "EOG signal":
        #     self.plot_waveform(self.canvas, data, **args)
        # elif self.plot == "EOG blink rate":
        #     self.plot_waveform(self.canvas, data, **args)
        # elif self.plot == "EOG individual blinks":
        #     self.plot_waveform(self.canvas, data, **args)

        else:
            raise ValueError("Plot not supported")

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        return b64_img
