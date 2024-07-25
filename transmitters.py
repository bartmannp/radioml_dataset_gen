#!/usr/bin/env python3
import time
import math
from scipy.signal import get_window
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes

sps = 8
ebw = 0.35

class transmitter_mapper(gr.hier_block2):
    def __init__(self, constellation, txname, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        
        # Use digital.generic_mod instead of custom mapper
        self.mod = digital.generic_mod(
            constellation=constellation,
            differential=False,
            samples_per_symbol=samples_per_symbol,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False
        )
        
        # Pulse shaping filter
        nfilts = 32
        ntaps = nfilts * 11 * int(samples_per_symbol)
        rrc_taps = firdes.root_raised_cosine(
            nfilts, nfilts, 1.0, excess_bw, ntaps)
        self.rrc_filter = filter.pfb_arb_resampler_ccf(samples_per_symbol, rrc_taps)
        self.connect(self, self.mod, self.rrc_filter, self)

class transmitter_bpsk(transmitter_mapper):
    modname = "BPSK"
    def __init__(self):
        super().__init__(digital.constellation_bpsk().base(), "transmitter_bpsk", sps, ebw)

class transmitter_qpsk(transmitter_mapper):
    modname = "QPSK"
    def __init__(self):
        super().__init__(digital.constellation_qpsk().base(), "transmitter_qpsk", sps, ebw)

class transmitter_8psk(transmitter_mapper):
    modname = "8PSK"
    def __init__(self):
        super().__init__(digital.constellation_8psk().base(), "transmitter_8psk", sps, ebw)

class transmitter_pam4(transmitter_mapper):
    modname = "PAM4"
    def __init__(self):
        # Create a custom PAM4 constellation
        constellation_points = [complex(i - 1.5, 0) for i in range(4)]
        symbol_map = [0, 1, 3, 2]  # Gray coding
        constellation = digital.constellation_rect(constellation_points, symbol_map, 4, 2, 2, 1, 1)
        super().__init__(constellation, "transmitter_pam4", sps, ebw)

class transmitter_qam16(transmitter_mapper):
    modname = "QAM16"
    def __init__(self):
        super().__init__(digital.constellation_16qam().base(), "transmitter_qam16", sps, ebw)

class transmitter_qam64(transmitter_mapper):
    modname = "QAM64"
    def __init__(self):
        # Create a custom 64-QAM constellation
        constellation_points = [
            complex(x, y) for x in [-7, -5, -3, -1, 1, 3, 5, 7]
            for y in [-7, -5, -3, -1, 1, 3, 5, 7]
        ]
        symbol_map = list(range(64))  # Simple mapping, you might want to implement Gray coding
        constellation = digital.constellation_rect(constellation_points, symbol_map, 64, 1, 1, 1, 1)
        super().__init__(constellation, "transmitter_qam64", sps, ebw)

class transmitter_gfsk(gr.hier_block2):
    modname = "GFSK"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_gfsk",
            gr.io_signature(1, 1, gr.sizeof_char),
            gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.repack = blocks.repack_bits_bb(1, 8, "", False, gr.GR_MSB_FIRST)
        self.mod = digital.gfsk_mod(samples_per_symbol=sps, sensitivity=0.1, bt=ebw)
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
        self.rate = 44.1e3 / 200e3
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_cc(1.0)
        self.add = blocks.add_const_cc(1.0)
        self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_cc()
        self.connect(self, self.interp, self.cnv, self.mul, self.add, self.mod, self)
        self.connect(self.src, (self.mod, 1))

class transmitter_amssb(gr.hier_block2):
    modname = "AM-SSB"
    def __init__(self):
        gr.hier_block2.__init__(self, "transmitter_amssb",
        gr.io_signature(1, 1, gr.sizeof_float),
        gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = 44.1e3 / 200e3
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
        self.mul = blocks.multiply_const_ff(1.0)
        self.add = blocks.add_const_ff(1.0)
        self.src = analog.sig_source_f(200e3, analog.GR_SIN_WAVE, 0e3, 1.0)
        self.mod = blocks.multiply_ff()
        self.filt = filter.hilbert_fc(401)
        self.connect(self, self.interp, self.mul, self.add, self.mod, self.filt, self)
        self.connect(self.src, (self.mod, 1))

transmitters = {
    "discrete": [transmitter_bpsk, transmitter_qpsk, transmitter_8psk, transmitter_pam4, transmitter_qam16, transmitter_qam64, transmitter_gfsk, transmitter_cpfsk],
    "continuous": [transmitter_fm, transmitter_am, transmitter_amssb]
}
