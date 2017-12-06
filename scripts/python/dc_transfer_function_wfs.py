"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""
from targetpipe.io.camera import Config
Config('checm')

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
    FuncFormatter, AutoMinorLocator
from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.calib.camera.tf import TFApplier
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.plots.official import ThesisPlotter
from tqdm import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from os.path import join, dirname
from IPython import embed
import pandas as pd
from scipy.stats import norm
from targetpipe.utils.dactov import checm_dac_to_volts


class WaveformPlotter(ThesisPlotter):
    name = 'WaveformPlotter'

    def __init__(self, config, tool, **kwargs):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, tool=tool, **kwargs)

    def add(self, waveform):
        self.ax.plot(waveform)

    def save(self, output_path=None):
        self.ax.set_title("Waveforms for one channel, incrementing VPED")
        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Amplitude (ADC Pedestal-Subtracted)")
        self.ax.xaxis.set_major_locator(MultipleLocator(16))
        super().save(output_path)


class TFInvestigator(Tool):
    name = "TFInvestigator"
    description = "Produce plots associated with the " \
                  "transfer function calibration"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.df_file = None
        self.tf = None
        self.r1 = None

        self.n_pixels = None
        self.n_samples = None

        self.p_vi = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        ped_path = "/Volumes/gct-jason/data/170314/pedestal/Run00053_ped.tcal"
        self.r1 = TargetioR1Calibrator(pedestal_path=ped_path,
                                       **kwargs,
                                       )

        dfl = []
        bp = '/Volumes/gct-jason/data/170314/tf/Run{:05}_r0.tio'
        it = range(50)
        for i in it:
            vped = 800 + i * 40
            dfl.append(dict(path=bp.format(i+1), vped=vped))

        for d in dfl:
            d['reader'] = TargetioFileReader(input_path=d['path'], **kwargs)
        self.df_file = pd.DataFrame(dfl)

        first_event = dfl[0]['reader'].get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        r1 = first_event.r1.tel[telid].pe_samples[0]
        self.n_pixels, self.n_samples = r1.shape

        p_kwargs = kwargs
        p_kwargs['script'] = "dc_transfer_function_wfs"
        p_kwargs['figure_name'] = "vped_increments"
        self.p_vi = WaveformPlotter(**kwargs)

    def start(self):

        desc1 = 'Looping through files'
        n_rows = len(self.df_file.index)
        t = tqdm(self.df_file.iterrows(), total=n_rows, desc=desc1)
        for index, row in t:
            path = row['path']
            reader = row['reader']
            vped = row['vped']

            source = reader.read()
            n_events = reader.num_events

            event = reader.get_event(0)
            self.r1.calibrate(event)
            wf = event.r1.tel[0].pe_samples[0, 0]

            self.p_vi.add(wf)

    def finish(self):
        self.p_vi.save()


exe = TFInvestigator()
exe.run()
