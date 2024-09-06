#!/usr/bin/python3

# Acoustics package
from robot_arm_acoustic.measurements.PlotTools import (
    plot_weighting,
    log_formatter,
    plot_absolute_error,
    plot_relative_error,
    plot_relative_separated_error,
    compute_l2_errors,
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

from plotly.subplots import make_subplots
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# Utility packages
import glob
import sys
import numpy as np
import os

np.set_printoptions(threshold=sys.maxsize)

# Plot packages

# Data processing tools

INTERACTIVE = False

# Reference signal index
INDEX = 0

if __name__ == "__main__":

    # Get processing method
    processingMethod = "welch"
    try:
        processingMethod = sys.argv[1].lower()
        if processingMethod not in ["welch", "farina"]:
            raise ValueError("Invalid processing method")
    except IndexError:
        print(
            "Invalid processing method, defaulting to " + processingMethod + " method"
        )

    # Get interest output signal type
    outputSignalType = "sweep"
    try:
        outputSignalType = sys.argv[2].lower()
    except IndexError:
        print(
            "Invalid output signal type, defaulting to "
            + str(outputSignalType)
            + " output signal"
        )

    # Get transfer function input and output signals names
    inputSignal = "Out1"  # Voltage
    outputSignal = "In1"  # Pressure
    try:
        inputSignal = sys.argv[3]
        outputSignal = sys.argv[4]
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

    FilesWith = sorted(
        glob.glob("WithRobot/measurement_[0-9].csv"),
        key=lambda file: int(file.split(".")[0].split("_")[-1]),
    )
    FilesWithout = sorted(
        glob.glob("WithoutRobot/measurement_[0-9].csv"),
        key=lambda file: int(file.split(".")[0].split("_")[-1]),
    )

    WWith = []
    WWithout = []
    PWith = []
    VWith = []

    Freqs = []
    unit = None

    for i, file in enumerate(FilesWith):

        tfeWith, vWith, pWith = get_transfert_function(
            file,
            inputSignal,
            outputSignal,
            processing_method=processingMethod,
            sync_out_chan=2,
            sync_in_chan=1,
            sync_added_time=0.5,
        )
        WWith.append(tfeWith.nth_oct_smooth_to_weight_complex(octBand, fmin, fmax))
        PWith.append(pWith)
        VWith.append(vWith)

        if i == 0:
            Freqs = tfeWith.freqs[(tfeWith.freqs > fmin) & (tfeWith.freqs < fmax)]
            unit = tfeWith.unit

    for i, file in enumerate(FilesWithout):

        tfeWithout, _, _ = get_transfert_function(
            file,
            inputSignal,
            outputSignal,
            processing_method=processingMethod,
            sync_out_chan=2,
            sync_in_chan=1,
            sync_added_time=0.5,
        )
        WWithout.append(tfeWithout.nth_oct_smooth_to_weight_complex(octBand, fmin, fmax))

    if(len(WWith) > 1):

        if INTERACTIVE:
            axAllWith = make_subplots(rows=2, cols=1)
            axAllWithAbs = make_subplots(rows=2, cols=1)
            axAllWithRel = make_subplots(rows=1, cols=1)
            axAllWithRelSep = make_subplots(rows=2, cols=1)
        else:
            figAllWith, axAllWith = plt.subplots(3, figsize=(figsize[0],1.45*figsize[1]))
            figAllWithAbs, axAllWithAbs = plt.subplots(2, figsize=figsize)
            figAllWithRel, axAllWithRel = plt.subplots(1, figsize=figsize)
            figAllWithRelSep, axAllWithRelSep = plt.subplots(2, figsize=figsize)
    
        for i, w in enumerate(WWith):

            out = plot_weighting(
                w,
                Freqs,
                unit=unit,
                ax=axAllWith,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label=str(i + 1),
            )

            plot_absolute_error(
                w,
                WWith[INDEX],
                Freqs,
                ax=axAllWithAbs,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                ylim_modulus=[-0.3, 0.3],
                ylim_phase=[-0.05, 0.05],
                marker=markers[i],
                color=cmap(i),
                label="Absolute error " + str(i + 1),
            )

            plot_absolute_error(
                w,
                WWith[INDEX],
                Freqs,
                plot_phase=False,
                ax=axAllWith[2],
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                ylim_modulus=[-0.25, 1.8],
                marker=markers[i],
                color=cmap(i),
                label=None,
            )
            axAllWith[1].set_xlabel('')

            plot_relative_error(
                w,
                WWith[INDEX],
                Freqs,
                ax=axAllWithRel,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label="Relative error " + str(i + 1),
            )

            plot_relative_separated_error(
                w,
                WWith[INDEX],
                Freqs,
                ax=axAllWithRelSep,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label="Relative error " + str(i + 1),
            )

            errorAllWithAbs, errorAllWithRel = compute_l2_errors(
                w, WWith[INDEX], frequencyRange=[fminValidity, fmaxValidity]
            )
            print(
                "Absolute L2 repetability error with robot "
                + str(i + 1)
                + " : "
                + str(errorAllWithAbs)
                + " Pa/V"
            )
            print(
                "Relative L2 repetability error with robot "
                + str(i + 1)
                + " : "
                + str(100 * errorAllWithRel)
                + " %"
            )

        # set_title(axAllWith,"Pressure/Input signal TFE with robot\n1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithAbs,"Pressure/Input signal TFE repetability absolute error with robot\n1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithRel,"Pressure/Input signal TFE repetability relative error with robot\n1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithRelSep,"Pressure/Input Signal TFE repetability modulus and phase\nrelative errors with robot - 1/" + str(octBand) + " octave smoothing")

        if INTERACTIVE:
            axAllWith.write_html("./" + processingMethod + "_AllPressuresWith.html")
            axAllWithAbs.write_html(
                "./" + processingMethod + "_AbsoluteErrorAllPressuresWith.html"
            )
            axAllWithRel.write_html(
                "./" + processingMethod + "_RelativeErrorAllPressuresWith.html"
            )
            axAllWithRelSep.write_html(
                "./" + processingMethod + "_RelativeErrorSeparateAllPressuresWith.html"
            )
        else:
            save_fig(figAllWith, "./" + processingMethod + "_AllPressuresWith.pdf")
            save_fig(
                figAllWithAbs,
                "./" + processingMethod + "_AbsoluteErrorAllPressuresWith.pdf",
            )
            save_fig(
                figAllWithRel,
                "./" + processingMethod + "_RelativeErrorAllPressuresWith.pdf",
            )
            save_fig(
                figAllWithRelSep,
                "./" + processingMethod + "_RelativeErrorSeparateAllPressuresWith.pdf",
            )
            plt.close("all")

    if(len(WWithout) > 1):

        if INTERACTIVE:
            axAllWithout = make_subplots(rows=2, cols=1)
            axAllWithoutAbs = make_subplots(rows=2, cols=1)
            axAllWithoutRel = make_subplots(rows=1, cols=1)
            axAllWithoutRelSep = make_subplots(rows=2, cols=1)
        else:
            figAllWithout, axAllWithout = plt.subplots(2, figsize=figsize)
            figAllWithoutAbs, axAllWithoutAbs = plt.subplots(2, figsize=figsize)
            figAllWithoutRel, axAllWithoutRel = plt.subplots(1, figsize=figsize)
            figAllWithoutRelSep, axAllWithoutRelSep = plt.subplots(2, figsize=figsize)

        for i, w in enumerate(WWithout):

            plot_weighting(
                w,
                Freqs,
                unit=unit,
                ax=axAllWithout,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label=str(i + 1),
            )

            plot_absolute_error(
                w,
                WWithout[INDEX],
                Freqs,
                ax=axAllWithoutAbs,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label="Absolute error " + str(i + 1),
            )

            plot_relative_error(
                w,
                WWithout[INDEX],
                Freqs,
                ax=axAllWithoutRel,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label="Relative error " + str(i + 1),
            )

            plot_relative_separated_error(
                w,
                WWithout[INDEX],
                Freqs,
                ax=axAllWithoutRelSep,
                validity_range=[fminValidity, fmaxValidity],
                interactive=INTERACTIVE,
                marker=markers[i],
                color=cmap(i),
                label="Relative error " + str(i + 1),
            )

            errorAllWithAbs, errorAllWithRel = compute_l2_errors(
                w, WWithout[INDEX], frequencyRange=[fminValidity, fmaxValidity]
            )
            print(
                "Absolute L2 repetability error without robot "
                + str(i + 1)
                + " : "
                + str(errorAllWithAbs)
                + " Pa/V"
            )
            print(
                "Relative L2 repetability error without robot "
                + str(i + 1)
                + " : "
                + str(100 * errorAllWithRel)
                + " %"
            )

        # set_title(axAllWithout,"Pressure/Input signal TFE without robot - 1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithoutAbs,"Pressure/Input signal TFE repetability absolute error without robot\n1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithoutRel,"Pressure/Input signal TFE repetability relative error without robot\n1/" + str(octBand) + " octave smoothing")
        # set_title(axAllWithoutRelSep,"Pressure/Input Signal TFE repetability modulus and phase\nrelative errors without robot - 1/" + str(octBand) + " octave smoothing")

        if INTERACTIVE:
            axAllWithout.write_html("./" + processingMethod + "_AllPressuresWithout.html")
            axAllWithoutAbs.write_html(
                "./" + processingMethod + "_AbsoluteErrorAllPressuresWithout.html"
            )
            axAllWithoutRel.write_html(
                "./" + processingMethod + "_RelativeErrorAllPressuresWithout.html"
            )
            axAllWithoutRelSep.write_html(
                "./" + processingMethod + "_RelativeErrorSeparateAllPressuresWithout.html"
            )
        else:
            save_fig(figAllWithout, "./" + processingMethod + "_AllPressuresWithout.pdf")
            save_fig(
                figAllWithoutAbs,
                "./" + processingMethod + "_AbsoluteErrorAllPressuresWithout.pdf",
            )
            save_fig(
                figAllWithoutRel,
                "./" + processingMethod + "_RelativeErrorAllPressuresWithout.pdf",
            )
            save_fig(
                figAllWithoutRelSep,
                "./" + processingMethod + "_RelativeErrorSeparateAllPressuresWithout.pdf",
            )
            plt.close("all")

    if(len(WWithout) == 1 and len(WWith) == 1):

        if INTERACTIVE:
            axBoth = make_subplots(rows=2, cols=1)
            axAbs = make_subplots(rows=2, cols=1)
            axRel = make_subplots(rows=1, cols=1)
            axRelSep = make_subplots(rows=2, cols=1)
        else:
            figBoth, axBoth = plt.subplots(2, figsize=figsize)
            figAbs, axAbs = plt.subplots(2, figsize=figsize)
            figRel, axRel = plt.subplots(1, figsize=figsize)
            figRelSep, axRelSep = plt.subplots(2, figsize=figsize)

        plot_weighting(
            WWithout[0],
            Freqs,
            unit=unit,
            ax=axBoth,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[0],
            color=cmap(0),
            label="Without robot",
        )
        plot_weighting(
            WWith[0],
            Freqs,
            unit=unit,
            ax=axBoth,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[1],
            color=cmap(1),
            label="With robot",
        )

        plot_absolute_error(
            WWith[0],
            WWithout[0],
            Freqs,
            ax=axAbs,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[0],
            color=cmap(0),
            label="Absolute error",
        )

        plot_relative_error(
            WWith[0],
            WWithout[0],
            Freqs,
            ax=axRel,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[0],
            color=cmap(0),
            label="Relative error",
        )

        plot_relative_separated_error(
            WWith[0],
            WWithout[0],
            Freqs,
            ax=axRelSep,
            validity_range=[fminValidity, fmaxValidity],
            interactive=INTERACTIVE,
            marker=markers[0],
            color=cmap(0),
            label="Relative error",
        )

        errorAbs, errorRel = compute_l2_errors(
            WWith[0], WWithout[0], frequencyRange=[fminValidity, fmaxValidity]
        )
        print(
            "Absolute L2 error with and without robot (index "
            + str(0)
            + ") : "
            + str(errorAbs)
            + " Pa/V"
        )
        print(
            "Relative L2 error with and without robot (index "
            + str(0)
            + ") : "
            + str(100 * errorRel)
            + " %"
        )

        # set_title(axBoth[0],"Pressure/Input signal TFE - 1/" + str(octBand) + " octave smoothing")
        # set_title(axAbs[0],"Pressure/Input signal TFE absolute error - 1/" + str(octBand) + " octave smoothing")
        # set_title(axRel,"Pressure/Input signal TFE relative error - 1/" + str(octBand) + " octave smoothing")
        # set_title(axRelSep[0],"Pressure/Input Signal TFE modulus and phase relative errors\n1/" + str(octBand) + " octave smoothing")

        if INTERACTIVE:
            axBoth.write_html("./" + processingMethod + "_Both.html")
            axAbs.write_html("./" + processingMethod + "_AbsoluteError.html")
            axRel.write_html("./" + processingMethod + "_RelativeError.html")
            axRelSep.write_html("./" + processingMethod + "_RelativeErrorSeparate.html")
        else:
            save_fig(figBoth, "./" + processingMethod + "_Pressure.pdf")
            save_fig(figAbs, "./" + processingMethod + "_AbsoluteError.pdf")
            save_fig(figRel, "./" + processingMethod + "_RelativeError.pdf")
            save_fig(figRelSep, "./" + processingMethod + "_RelativeErrorSeparate.pdf")
            plt.close("all")

        figC, axC = plt.subplots(1, figsize=figsize)

        PWith[0].coh(
            VWith[0], nperseg=2 ** (np.ceil(np.log2(VWith[0].fs)))
        ).filterout([fmin, fmax]).plot(axC, label="Coherence", dby=False, plot_phase=False)

        axC.set_title("Pressure/Input Signal coherence", pad=30)
        axC.axvspan(
            fminValidity,
            fmaxValidity,
            color="gray",
            alpha=0.175,
            label="Valid frequency range",
        )
        axC.grid(which="major")
        axC.grid(linestyle="--", which="minor")
        axC.xaxis.set_minor_formatter(FuncFormatter(log_formatter))
        axC.legend(
            bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=5, borderaxespad=0.25
        )
        figC.tight_layout()
        figC.savefig(
            "./" + processingMethod + "_Coherence.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close("all")

        # plt.show()
