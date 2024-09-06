#!/usr/bin/python3

# Utility packages
import glob
import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# Plot packages
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Data processing tools
from robot_arm_acoustic.measurements.PlotTools import (
    plot_absolute_error,
    plot_relative_error,
    plot_relative_separated_error,
    compute_l2_errors,
    compute_max_min_errors,
    save_fig,
    cmap,
    markers,
    figsize,
    fmin,
    fmax,
    fminValidity,
    fmaxValidity,
    octBand,
)

from robot_arm_acoustic.measurements.DataProcessingTools import get_transfert_function

INTERACTIVE = False

# Reference signal INDEX
INDEX = 2

if __name__ == "__main__":

    # Get processing method
    processingMethod = "welch"
    try:
        processingMethod = os.sys.argv[1].lower()
        if processingMethod not in ["welch", "farina"]:
            raise ValueError("Invalid processing method")
    except IndexError:
        print(
            "Invalid processing method, defaulting to " + processingMethod + " method"
        )

    # Get interest output signal type
    outputSignalType = "log_sweep"
    try:
        outputSignalType = sys.argv[2].lower()
    except IndexError:
        print(
            "Invalid output signal type, defaulting to "
            + str(outputSignalType)
            + " output signal"
        )

    # Get transfer function input and output signals names
    inputSignal = 1  # Voltage
    outputSignal = 0  # Pressure
    try:
        inputSignal = int(sys.argv[3])
        outputSignal = int(sys.argv[4])
    except IndexError:
        print(
            "Invalid input/output signals, defaulting to input : "
            + str(inputSignal)
            + " and output : "
            + str(outputSignal)
        )

    print(
        "Processing input "
        + str(inputSignal)
        + " and output "
        + str(outputSignal)
        + " with "
        + processingMethod
        + " method"
    )

    ControlPointsFolders = sorted(
        glob.glob("Measurements_[0-9]/"), key=lambda folder: int(folder.split("_")[-1])
    )

    WWith = []
    WWithout = []

    for i, folder in enumerate(ControlPointsFolders):

        fileWith = sorted(
            glob.glob(folder + outputSignalType + "_measurement_" + str(INDEX)),
            key=lambda file: int(file.split("_")[-1]),
        )[0]
        fileWithout = sorted(
            glob.glob(folder + outputSignalType + "_measurement_3"),
            key=lambda file: int(file.split("_")[-1]),
        )[0]

        tfeWith, _, _ = get_transfert_function(
            fileWith,
            inputSignal,
            outputSignal,
            processing_method=processingMethod,
            sync_out_chan=2,
            sync_in_chan=1,
            sync_added_time=0.5,
        )
        WWith.append(tfeWith.nth_oct_smooth_to_weight_complex(octBand, fmin, fmax))

        tfeWithout, _, _ = get_transfert_function(
            fileWithout,
            inputSignal,
            outputSignal,
            processing_method=processingMethod,
            sync_out_chan=2,
            sync_in_chan=1,
            sync_added_time=0.5,
        )
        WWithout.append(tfeWithout.nth_oct_smooth_to_weight_complex(octBand, fmin, fmax))

        if i == 0:
            Freqs = tfeWith.freqs[(tfeWith.freqs > fmin) & (tfeWith.freqs < fmax)]
            unit = tfeWith.unit

    if INTERACTIVE:
        axAllAbs = make_subplots(rows=2, cols=1)
        axAllRel = make_subplots(rows=1, cols=1)
        axAllRelSep = make_subplots(rows=2, cols=1)
    else:
        figAllAbs, axAllAbs = plt.subplots(2, figsize=figsize)
        figAllRel, axAllRel = plt.subplots(1, figsize=figsize)
        figAllRelSep, axAllRelSep = plt.subplots(2, figsize=figsize)

    for i, (wWith, wWithout) in enumerate(zip(WWith, WWithout)):

        plot_absolute_error(
            wWith,
            wWithout,
            Freqs,
            ax=axAllAbs,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            ylim_phase=[-np.pi / 4, np.pi / 4],
            ylim_modulus=[-4.5, 4.5],
            marker=markers[i],
            color=cmap(i),
            label=str(i + 1),
        )

        plot_relative_error(
            wWith,
            wWithout,
            Freqs,
            ax=axAllRel,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            ylim_modulus=[-0.05, 1],
            marker=markers[i],
            color=cmap(i),
            label=str(i + 1),
        )

        plot_relative_separated_error(
            wWith,
            wWithout,
            Freqs,
            ax=axAllRelSep,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[i],
            color=cmap(i),
            label=str(i + 1),
        )

        errorAbs, errorRel = compute_l2_errors(
            wWith, wWithout, frequencyRange=[fminValidity, fmaxValidity]
        )
        print(
            "Absolute L2 error with and without robot (index "
            + str(i + 1)
            + ") : "
            + str(errorAbs)
            + " Pa/V"
        )
        print(
            "Relative L2 error with and without robot (index "
            + str(i + 1)
            + ") : "
            + str(100 * errorRel)
            + " %"
        )

        errorAbsMin, errorAbsMax, errorRelMin, errorRelMax = compute_max_min_errors(
            wWith, wWithout, frequencyRange=[fminValidity, fmaxValidity]
        )
        print(
            "Absolute min/max error with and without robot (index "
            + str(i + 1)
            + ") : "
            + str(errorAbsMin)
            + " / "
            + str(errorAbsMax)
            + " Pa/V"
        )
        print(
            "Relative min/max error with and without robot (index "
            + str(i + 1)
            + ") : "
            + str(100 * errorRelMin)
            + " / "
            + str(100 * errorRelMax)
            + " %"
        )

    # set_title(axAllAbs, "Pressure/Input signal TFE absolute error - 1/" + str(octBand) + " octave smoothing")
    # set_title(axAllRel, "Pressure/Input signal TFE relative error - 1/" + str(octBand) + " octave smoothing")
    # set_title(axAllRelSep, "Pressure/Input signal TFE modulus and phase relative error - 1/" + str(octBand) + " octave smoothing")

    if INTERACTIVE:
        axAllAbs.write_html(
            "./" + str(INDEX) + "_" + processingMethod + "_AbsoluteError.html"
        )
        axAllRel.write_html(
            "./" + str(INDEX) + "_" + processingMethod + "_RelativeError.html"
        )
        axAllRelSep.write_html(
            "./" + str(INDEX) + "_" + processingMethod + "_RelativeErrorSeparate.html"
        )
    else:
        save_fig(
            figAllAbs, "./" + str(INDEX) + "_" + processingMethod + "_AbsoluteError.pdf"
        )
        save_fig(
            figAllRel, "./" + str(INDEX) + "_" + processingMethod + "_RelativeError.pdf"
        )
        save_fig(
            figAllRelSep,
            "./" + str(INDEX) + "_" + processingMethod + "_RelativeErrorSeparate.pdf",
        )
        plt.close("all")

    # plt.show()
