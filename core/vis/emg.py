import numpy as np
import pandas as pd


def emg_plot_signal(emg_signals, info=None, ax=None, label=None):
    if not isinstance(emg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `emg_signals` argument must"
            " be the DataFrame returned by `emg_process()`."
        )

    sampling_rate = info["sampling_rate"]
    x_axis = np.linspace(
        0, emg_signals.shape[0] / info["sampling_rate"], emg_signals.shape[0]
    )

    if sampling_rate is not None:
        ax.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax.set_xlabel("Samples")

    # Plot cleaned and raw EMG.
    ax.set_title("Raw and Cleaned Signal")
    ax.plot(
        x_axis,
        emg_signals["EMG_Clean"],
        label=f"Cleaned {label}",
        zorder=1,
        linewidth=1.5,
    )


def emg_plot_act(emg_signals, info=None, ax=None, label=None):
    # Sanity-check input.
    if not isinstance(emg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: The `emg_signals` argument must"
            " be the DataFrame returned by `emg_process()`."
        )

    # Determine what to display on the x-axis, mark activity.
    x_axis = np.linspace(
        0, emg_signals.shape[0] / info["sampling_rate"], emg_signals.shape[0]
    )

    sampling_rate = info["sampling_rate"]
    if sampling_rate is not None:
        ax.set_xlabel("Time (seconds)")
    elif sampling_rate is None:
        ax.set_xlabel("Samples")

    # Plot Amplitude.
    ax.set_title("Muscle Activation")
    ax.plot(
        x_axis,
        emg_signals["EMG_Amplitude"],
        # color="#FF9800",
        # label="Amplitude",
        label=label,
        linewidth=1.5,
    )


def _emg_plot_activity(emg_signals, onsets, offsets):
    activity_signal = pd.Series(np.full(len(emg_signals), np.nan))
    activity_signal[onsets] = emg_signals["EMG_Amplitude"][onsets].values
    activity_signal[offsets] = emg_signals["EMG_Amplitude"][offsets].values
    activity_signal = activity_signal.bfill()

    if np.any(activity_signal.isna()):
        index = np.min(np.where(activity_signal.isna())) - 1
        value_to_fill = activity_signal[index]
        activity_signal = activity_signal.fillna(value_to_fill)

    return activity_signal
