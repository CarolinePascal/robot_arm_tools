#!/usr/bin/python3

#System package
import os

#Acoustics package
import measpy as ms

## Function computing the transfer function between two signals from a given measurement
# @param source_path Path to the measurement directory
# @param input_signal Name of the input signal
# @param output_signal Name of the output signal
# @param processing_method Method used to compute the transfer function (default: "welch")
# @param sync_out_chan Name of the output channel used for synchronization
# @param sync_in_chan Name of the input channel used for synchronization
# @param sync_added_time Time added to the synchronization
def get_transfert_function(source_path, input_signal, output_signal, processing_method = "welch", sync_out_chan = None,sync_in_chan = None, sync_added_time = 0.0):

    print("Data processing : " + source_path)
    
    measurement = ms.Measurement.from_csvwav(source_path.split(".")[0])

    #if sync_out_chan is not None and sync_in_chan is not None:
        #measurement.sync_render(sync_out_chan, sync_in_chan, sync_added_time)

    output = measurement.data[output_signal]
    input = measurement.data[input_signal]

    # Check processing method compatibility
    if processing_method == "farina" and not "sweep" in input.desc:
        raise ValueError("Farina method cannot be used with non log sweep signals")

    tfe = None
    if processing_method == "farina":
        unfiltered_tfe, _, _, _ = output.harmonic_disto(
            nh=2,
            win_max_length=2**18,
            freq_min=input.freq_min,
            freq_max=input.freq_max,
            delay=0.0,
        )
        tfe = unfiltered_tfe[0]
        tfe.unit = output.unit / input.unit
    else:
        tfe = output.tfe_welch(input)  # Also possible for dB values : (P*V.rms).tfe_welch(V)
    
    return tfe, input, output