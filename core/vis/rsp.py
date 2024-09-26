import pandas as pd
import numpy as np


def rsp_plot_signal(rsp_signals, info, ax):
    # Mark peaks, troughs and phases.
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
    inhale = np.where(rsp_signals["RSP_Phase"] == 1)[0]
    exhale = np.where(rsp_signals["RSP_Phase"] == 0)[0]

    nrow = 2

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Get signals marking inspiration and expiration.
    exhale_signal, inhale_signal = _rsp_plot_phase(rsp_signals, troughs, peaks)

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))
    ax.set_xlabel(x_label)

    ax.plot(x_axis, rsp_signals["RSP_Raw"], color="#B0BEC5", label="Raw", zorder=1)
    ax.plot(
        x_axis,
        rsp_signals["RSP_Clean"],
        color="#2196F3",
        label="Cleaned",
        zorder=2,
        linewidth=1.5,
    )

    ax.scatter(
        x_axis[peaks],
        rsp_signals["RSP_Clean"][peaks],
        color="red",
        label="Exhalation Onsets",
        zorder=3,
    )
    ax.scatter(
        x_axis[troughs],
        rsp_signals["RSP_Clean"][troughs],
        color="orange",
        label="Inhalation Onsets",
        zorder=4,
    )

    # Shade region to mark inspiration and expiration.
    ax.fill_between(
        x_axis[exhale],
        exhale_signal[exhale],
        rsp_signals["RSP_Clean"][exhale],
        where=rsp_signals["RSP_Clean"][exhale] > exhale_signal[exhale],
        color="#CFD8DC",
        linestyle="None",
        label="exhalation",
    )
    ax.fill_between(
        x_axis[inhale],
        inhale_signal[inhale],
        rsp_signals["RSP_Clean"][inhale],
        where=rsp_signals["RSP_Clean"][inhale] > inhale_signal[inhale],
        color="#ECEFF1",
        linestyle="None",
        label="inhalation",
    )

    ax.legend()


def rsp_plot_br(rsp_signals, info, ax):
    nrow = 2

    # Determine mean rate.
    rate_mean = np.mean(rsp_signals["RSP_Rate"])

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))
    ax.set_xlabel(x_label)

    # Plot rate and optionally amplitude.
    ax.set_title("Breathing Rate")
    ax.plot(
        x_axis,
        rsp_signals["RSP_Rate"],
        color="#4CAF50",
        label="Rate",
        linewidth=1.5,
    )
    ax.axhline(y=rate_mean, label="Mean", linestyle="--", color="#4CAF50")
    ax.legend()


def rsp_plot_ba(rsp_signals, info, ax):
    nrow = 2

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
        # Determine mean amplitude.
        amplitude_mean = np.mean(rsp_signals["RSP_Amplitude"])
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))
    ax.set_xlabel(x_label)

    if "RSP_Amplitude" in list(rsp_signals.columns):
        ax.set_title("Breathing Amplitude")

        ax.plot(
            x_axis,
            rsp_signals["RSP_Amplitude"],
            color="#009688",
            label="Amplitude",
            linewidth=1.5,
        )
        ax.axhline(y=amplitude_mean, label="Mean", linestyle="--", color="#009688")
        ax.legend()


def rsp_plot_vbt(rsp_signals, info, ax):
    nrow = 2

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
        rvt_mean = np.mean(rsp_signals["RSP_RVT"])
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))
    ax.set_xlabel(x_label)

    if "RSP_RVT" in list(rsp_signals.columns):
        ax.set_title("Respiratory Volume per Time")

        ax.plot(
            x_axis,
            rsp_signals["RSP_RVT"],
            color="#00BCD4",
            label="RVT",
            linewidth=1.5,
        )
        ax.axhline(y=rvt_mean, label="Mean", linestyle="--", color="#009688")
        ax.legend()


def rsp_plot_cs(rsp_signals, info, ax):
    nrow = 2

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(rsp_signals) / info["sampling_rate"], len(rsp_signals))
    ax.set_xlabel(x_label)

    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        ax.set_title("Cycle Symmetry")

        ax.plot(
            x_axis,
            rsp_signals["RSP_Symmetry_PeakTrough"],
            color="green",
            label="Peak-Trough Symmetry",
            linewidth=1.5,
        )
        ax.plot(
            x_axis,
            rsp_signals["RSP_Symmetry_RiseDecay"],
            color="purple",
            label="Rise-Decay Symmetry",
            linewidth=1.5,
        )
        ax.legend()


# =============================================================================
# Internals
# =============================================================================
def _rsp_plot_phase(rsp_signals, troughs, peaks):
    exhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    exhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    exhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    exhale_signal = exhale_signal.bfill()

    inhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    inhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    inhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    inhale_signal = inhale_signal.ffill()

    return exhale_signal, inhale_signal
