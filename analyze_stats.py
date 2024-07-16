import numpy as np
import pickle
import matplotlib.pyplot as plt

def calc_vec_energy(vec):
    isquared = np.power(vec[0], 2.0)
    qsquared = np.power(vec[1], 2.0)
    inst_energy = np.sqrt(isquared + qsquared)
    return sum(inst_energy)

def calc_mod_energies(ds):
    for modulation, snr in ds:
        avg_energy = 0
        nvectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_energy += calc_vec_energy(vec)
        avg_energy /= nvectors
        print(f"{modulation} at {snr} has {nvectors} vectors avg energy of {avg_energy:.1f}")

def calc_mod_bias(ds):
    for modulation, snr in ds:
        avg_bias_re = 0
        avg_bias_im = 0
        nvectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_bias_re += np.mean(vec[0])
            avg_bias_im += np.mean(vec[1])
        print(f"{modulation} at {snr} has {nvectors} vectors avg bias of {avg_bias_re:.1f} + {avg_bias_im:.1f}j")

def calc_mod_stddev(ds):
    for modulation, snr in ds:
        avg_stddev = 0
        nvectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_stddev += np.abs(np.std(vec[0] + 1j * vec[1]))
        print(f"{modulation} at {snr} has {nvectors} vectors avg stddev of {avg_stddev:.1f}")

def open_ds(location="X_4_dict.dat"):
    with open(location, "rb") as f:
        ds = pickle.load(f)
    return ds

def main():
    ds = open_ds()
    # Uncomment the following lines to visualize or calculate energies and stddev
    # plt.plot(ds[('BPSK', 12)][25][0][:])
    # plt.plot(ds[('BPSK', 12)][25][1][:])
    # plt.show()
    # calc_mod_energies(ds)
    # calc_mod_stddev(ds)
    calc_mod_bias(ds)

if __name__ == "__main__":
    main()
