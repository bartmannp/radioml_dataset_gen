#!/usr/bin/env python3
import time, math
from scipy.signal import get_window
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes

sps = 8
ebw = 0.35

class transmitter_mapper(gr.hier_block2):
    def __init__(self, mod_block, txname, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = mod_block
        # pulse shaping filter
        nfilts = 32
        ntaps = nfilts * 11 * int(samples_per_symbol)    # make nfilts filters of ntaps each
        rrc_taps = firdes.root_raised_cosine(
            nfilts,          # gain
            nfilts,          # sampling rate based on 32 filters in resampler
            1.0,             # symbol rate
            excess_bw,       # excess bandwidth (roll-off factor)
            ntaps)
        self.rrc_filter = filter.pfb_arb_resampler_ccf(samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)

class transmitter_bpsk(transmitter_mapper):
    modname = "BPSK"
    def __init__(self):
        constellation = digital.constellation_bpsk()
        mod_block = digital.generic_mod(
            constellation,
            False,  # Differential encoding
            sps,
            True,   # Gray coding
            ebw,
            False,  # verbose
            False,  # log
        )
        super().__init__(mod_block, "transmitter_bpsk", sps, ebw)

class transmitter_qpsk(transmitter_mapper):
    modname = "QPSK"
    def __init__(self):
        constellation = digital.constellation_qpsk()
        mod_block = digital.generic_mod(
            constellation,
            False,  # Differential encoding
            sps,
            True,   # Gray coding
            ebw,
            False,  # verbose
            False,  # log
        )
        super().__init__(mod_block, "transmitter_qpsk", sps, ebw)

class transmitter_8psk(transmitter_mapper):
    modname = "8PSK"
    def __init__(self):
        constellation = digital.constellation_8psk()
        mod_block = digital.generic_mod(
            constellation,
            False,  # Differential encoding
            sps,
            True,   # Gray coding
            ebw,
            False,  # verbose
            False,  # log
        )
        super().__init__(mod_block, "transmitter_8psk", sps, ebw)

class transmitter_pam4(transmitter_mapper):
    modname = "PAM4"
    def __init__(self):
        points = [complex(-3,0), complex(-1,0), complex(1,0), complex(3,0)]
        pre_diff_code = [0, 1, 3, 2]
        constellation = digital.constellation_rect(points, pre_diff_code, 4, 1, 1, 2.0, 2.0)
        mod_block = digital.chunks_to_symbols_bc(constellation.points())
        super().__init__(mod_block, "transmitter_pam4", sps, ebw)

class transmitter_qam16(transmitter_mapper):
    modname = "QAM16"
    def __init__(self):
        constellation = digital.constellation_16qam()
        mod_block = digital.generic_mod(
            constellation,
            False,  # Differential encoding
            sps,
            True,   # Gray coding
            ebw,
            False,  # verbose
            False,  # log
        )
        super().__init__(mod_block, "transmitter_qam16", sps, ebw)

class transmitter_qam64(transmitter_mapper):
    modname = "QAM64"
    def __init__(self):
        points = [complex(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in [-7, -5, -3, -1, 1, 3, 5, 7]]
        pre_diff_code = list(range(64))
        constellation = digital.constellation_rect(points, pre_diff_code, 64, 1, 1, 2.0, 2.0)
        mod_block = digital.generic_mod(
            constellation,
            False,  # Differential encoding
            sps,
            True,   # Gray coding
            ebw,
            False,  # verbose
            False,  # log
        )
        super().__init__(mod_block, "transmitter_qam64", sps, ebw)

class transmitter_gfsk(gr.hier_block2):
    modname = "GFSK"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_gfsk",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.repack = blocks.repack_bits_bb(8, 1, "", False, gr.GR_LSB_FIRST)
        self.mod = digital.gfsk_mod(sps, sensitivity=0.1, bt=ebw)
        self.connect(self, self.repack, self.mod, self)

class transmitter_cpfsk(gr.hier_block2):
    modname = "CPFSK"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_cpfsk",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = analog.cpfsk_bc(0.5, 1.0, sps)
        self.connect(self, self.mod, self)

class transmitter_fm(gr.hier_block2):
    modname = "WBFM"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_fm",
        gr.io_signature(1, 1, gr.sizeof_float),
        gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = analog.wfm_tx(audio_rate=44100.0, quad_rate=220.5e3)
        self.connect(self, self.mod, self)
        self.rate = 200e3 / 44.1e3

class transmitter_am(gr.hier_block2):
    modname = "AM-DSB"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_am",
        gr.io_signature(1, 1, gr.sizeof_float),
        gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = 200e3 / 44.1e3
        self.resampler = filter.rational_resampler_fff(
            interpolation=int(self.rate * 10), 
            decimation=10
        )
        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_cc(1.0)
        self.add = blocks.add_const_cc(1.0)
        self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_cc()
        self.connect(self, self.resampler, self.cnv, self.mul, self.add, self.mod, self)
        self.connect(self.src, (self.mod, 1))

class transmitter_amssb(gr.hier_block2):
    modname = "AM-SSB"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_amssb",
        gr.io_signature(1, 1, gr.sizeof_float),
        gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = 200e3 / 44.1e3
        self.resampler = filter.rational_resampler_fff(
            interpolation=int(self.rate * 10), 
            decimation=10
        )
        self.mul = blocks.multiply_const_ff(1.0)
        self.add = blocks.add_const_ff(1.0)
        self.src = analog.sig_source_f(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_ff()
        self.filt = filter.hilbert_fc(401)
        self.connect(self, self.resampler, self.mul, self.add, self.mod, self.filt, self)
        self.connect(self.src, (self.mod, 1))

transmitters = {
    "discrete": [transmitter_bpsk, transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam16, transmitter_qam64, transmitter_gfsk, transmitter_cpfsk],
    "continuous": [transmitter_fm, transmitter_am, transmitter_amssb]
}
