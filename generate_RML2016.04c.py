#!/usr/bin/env python3
from transmitters import transmitters
from source_alphabet import source_alphabet
import timeseries_slicer
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, h5py

'''
Generate dataset with dynamic channel model across range of SNRs
'''

apply_channel = True
output = {}
min_length = 9e9
snr_vals = list(range(-20, 20, 2))
for snr in snr_vals:
    for alphabet_type in transmitters.keys():
        print(alphabet_type)
        for i, mod_type in enumerate(transmitters[alphabet_type]):
            print("running test", i, mod_type)

            tx_len = int(10e3)
            if mod_type.modname == "QAM64":
                tx_len = int(30e3)
            if mod_type.modname == "QAM16":
                tx_len = int(20e3)
            src = source_alphabet(alphabet_type, tx_len, True)
            mod = mod_type()
            fD = 1
            delays = [0.0, 0.9, 1.7]
            mags = [1, 0.8, 0.3]
            ntaps = 8
            noise_amp = 10 ** (-snr / 10.0)
            print(noise_amp)
            chan = channels.dynamic_channel_model(200e3, 0.01, 1e2, 0.01, 1e3, 8, fD, True, 4, delays, mags, ntaps, noise_amp, 0x1337)

            snk = blocks.vector_sink_c()

            tb = gr.top_block()

            # connect blocks
            if apply_channel:
                tb.connect(src, mod, chan, snk)
            else:
                tb.connect(src, mod, snk)
            tb.run()

            modulated_vector = np.array(snk.data(), dtype=np.complex64)
            if len(snk.data()) < min_length:
                min_length = len(snk.data())
                min_length_mod = mod_type
            output[(mod_type.modname, snr)] = modulated_vector

print("min length mod is %s with %i samples" % (min_length_mod.modname, min_length))
# trim the beginning and ends, and make all mods have equal number of samples
start_indx = 100
fin_indx = min_length - 100
for mod, snr in output:
    output[(mod, snr)] = output[(mod, snr)][start_indx:fin_indx]
X = timeseries_slicer.slice_timeseries_dict(output, 128, 64, 1000)

# Save the sliced dataset in HDF5 format
with h5py.File("RML2016.04c_sliced.h5", "w") as h5f:
    for key, value in X.items():
        mod_type, snr = key
        group = h5f.create_group(f"{mod_type}/{snr}")
        group.create_dataset('data', data=value)

# Save the entire dataset in HDF5 format as a stacked array
X_stacked = np.vstack(list(X.values()))
with h5py.File("RML2016.04c_stacked.h5", "w") as h5f:
    h5f.create_dataset('data', data=X_stacked)

print("Data saved in HDF5 format")