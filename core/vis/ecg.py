import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk


def ecg_hb_features(signals, info):
    hbs = nk.ecg_segment(signals, info["ECG_R_Peaks"], info["sampling_rate"])
    df = nk.epochs_to_df(hbs)
    # Get main signal column name
    col = [c for c in ["Signal", "ECG_Raw", "ECG_Clean"] if c in df.columns][-1]
    # Average heartbeat
    mean_heartbeat = df.groupby("Time")[[col]].mean()

    waves = {}
    # Plot individual waves
    for wave in ["P", "Q", "S", "T"]:
        wave_col = f"ECG_{wave[0]}_Peaks"
        waves[wave_col] = []
        if wave_col in df.columns:
            series = df[col][df[wave_col] == 1]
            for t, val in series.items():
                if not np.isnan(val):
                    waves[wave_col].append((t, val))

    return mean_heartbeat, waves


def ecg_plot_signal(signals, info, ax=None):
    phase = None
    if "ECG_Phase_Ventricular" in signals.columns:
        phase = signals["ECG_Phase_Ventricular"].values
    ecg_peaks_plot(
        signals["ECG_Clean"].values,
        info=info,
        sampling_rate=info["sampling_rate"],
        raw=signals["ECG_Raw"].values,
        quality=signals["ECG_Quality"].values,
        phase=phase,
        ax=ax,
    )


def ecg_plot_hr(signals, info, ax=None):
    signal_rate_plot(
        signals["ECG_Rate"].values,
        info["ECG_R_Peaks"],
        sampling_rate=info["sampling_rate"],
        title="Heart Rate",
        ytitle="Beats per minute (bpm)",
        color="#FF5722",
        color_mean="#FF9800",
        color_points="#FFC107",
        ax=ax,
    )


def ecg_plot_ind(signals, info, ax=None):
    nk.ecg_segment(
        signals,
        info["ECG_R_Peaks"],
        info["sampling_rate"],
        show="return",
        ax=ax,
    )


def ecg_segment_plot(hbs, ax):
    df = nk.epochs_to_df(hbs)
    col = [c for c in ["Signal", "ECG_Raw", "ECG_Clean"] if c in df.columns][-1]
    mean_heartbeat = df.groupby("Time")[[col]].mean()

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("ECG")

    # Add Vertical line at 0
    ax.axvline(x=0, color="grey", linestyle="--")

    # Plot average heartbeat
    ax.plot(
        mean_heartbeat.index,
        mean_heartbeat,
        color="#F44336",
        linewidth=5,
        label="Average beat shape",
        zorder=1,
    )

    # Plot individual waves
    for wave in [
        ("P", "#3949AB"),
        ("Q", "#1E88E5"),
        ("S", "#039BE5"),
        ("T", "#00ACC1"),
    ]:
        wave_col = f"ECG_{wave[0]}_Peaks"
        if wave_col in df.columns:
            ax.scatter(
                df["Time"][df[wave_col] == 1],
                df[col][df[wave_col] == 1],
                color=wave[1],
                marker="+",
                label=f"{wave[0]}-waves",
                zorder=3,
            )

    # Legend
    ax.legend(loc="upper right")


def ecg_peaks_plot(
    ecg_cleaned,
    info=None,
    sampling_rate=1000,
    raw=None,
    quality=None,
    phase=None,
    ax=None,
):
    x_axis = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned))

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel("Time (seconds)")
    ax.set_title("ECG signal and peaks")

    # Quality Area -------------------------------------------------------------
    if quality is not None:
        quality = nk.rescale(
            quality,
            to=[
                np.min([np.min(raw), np.min(ecg_cleaned)]),
                np.max([np.max(raw), np.max(ecg_cleaned)]),
            ],
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax.fill_between(
            x_axis,
            minimum_line,
            quality,
            alpha=0.12,
            zorder=0,
            interpolate=True,
            facecolor="#4CAF50",
            label="Signal quality",
        )

    # Raw Signal ---------------------------------------------------------------
    if raw is not None:
        ax.plot(x_axis, raw, color="#B0BEC5", label="Raw signal", zorder=1)
        label_clean = "Cleaned signal"
    else:
        label_clean = "Signal"

    # Peaks -------------------------------------------------------------------
    ax.scatter(
        x_axis[info["ECG_R_Peaks"]],
        ecg_cleaned[info["ECG_R_Peaks"]],
        color="#FFC107",
        label="R-peaks",
        zorder=2,
    )

    # Artifacts ---------------------------------------------------------------
    _ecg_peaks_plot_artefacts(
        x_axis,
        ecg_cleaned,
        info,
        peaks=info["ECG_R_Peaks"],
        ax=ax,
    )

    # Clean Signal ------------------------------------------------------------
    if phase is not None:
        mask = (phase == 0) | (np.isnan(phase))
        diastole = ecg_cleaned.copy()
        diastole[~mask] = np.nan

        # Create overlap to avoid interuptions in signal
        mask[np.where(np.diff(mask))[0] + 1] = True
        systole = ecg_cleaned.copy()
        systole[mask] = np.nan

        ax.plot(
            x_axis,
            diastole,
            color="#B71C1C",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )
        ax.plot(
            x_axis,
            systole,
            color="#F44336",
            zorder=3,
            linewidth=1,
        )
    else:
        ax.plot(
            x_axis,
            ecg_cleaned,
            color="#F44336",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )

    # Optimize legend
    if raw is not None:
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
        )
    else:
        ax.legend(loc="upper right")

    return ax


def _ecg_peaks_plot_artefacts(
    x_axis,
    signal,
    info,
    peaks,
    ax,
):
    raw = [s for s in info.keys() if str(s).endswith("Peaks_Uncorrected")]
    if len(raw) == 0:
        return "No correction"
    raw = info[raw[0]]
    if len(raw) == 0:
        return "No bad peaks"
    if any([i < len(signal) for i in raw]):
        return (
            "Peak indices longer than signal. Signals might have been cropped. "
            + "Better skip plotting."
        )

    extra = [i for i in raw if i not in peaks]
    if len(extra) > 0:
        ax.scatter(
            x_axis[extra],
            signal[extra],
            color="#4CAF50",
            label="Peaks removed after correction",
            marker="x",
            zorder=2,
        )

    added = [i for i in peaks if i not in raw]
    if len(added) > 0:
        ax.scatter(
            x_axis[added],
            signal[added],
            color="#FF9800",
            label="Peaks added after correction",
            marker="x",
            zorder=2,
        )
    return ax


def signal_rate_plot(
    rate,
    peaks,
    sampling_rate=None,
    interpolation_method=None,
    title="Rate",
    ytitle="Cycle per minute",
    color="black",
    color_mean="orange",
    color_points="red",
    ax=None,
):
    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    if sampling_rate is None:
        x_axis = np.arange(0, len(rate))
        ax.set_xlabel("Time (samples)")
    else:
        x_axis = np.linspace(0, len(rate) / sampling_rate, len(rate))
        ax.set_xlabel("Time (seconds)")

    if interpolation_method is not None:
        title += " (interpolation method: " + str(interpolation_method) + ")"
    ax.set_title(title)
    ax.set_ylabel(ytitle)

    # Plot continuous rate
    ax.plot(
        x_axis,
        rate,
        color=color,
        label="Rate",
        linewidth=1.5,
    )

    # Plot points
    if peaks is not None:
        ax.scatter(
            x_axis[peaks],
            rate[peaks],
            color=color_points,
        )

    # Show average rate
    rate_mean = rate.mean()
    ax.axhline(y=rate_mean, label="Mean", linestyle="--", color=color_mean)

    ax.legend(loc="upper right")

    return ax
