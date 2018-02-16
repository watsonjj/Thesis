from targetpipe.io.camera import Config
Config('checs')

import numpy as np
from matplotlib.ticker import MultipleLocator
from traitlets import Dict, List
from ctapipe.core import Tool
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.r1 import TargetioR1Calibrator
from targetpipe.plots.official import ThesisPlotter
from tqdm import tqdm, trange
from IPython import embed


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

    def create(self, waveform, title, units):
        self.ax.plot(waveform, color='black')
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (ns)")
        self.ax.set_ylabel("Amplitude ({})".format(units))
        self.ax.xaxis.set_major_locator(MultipleLocator(16))


class HistPlotter(ThesisPlotter):
    name = 'HistPlotter'

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

    def add(self, data, label):
        data = data.ravel()

        mean = np.mean(data)
        stddev = np.std(data)
        N = data.size

        l = "{} (Mean = {:.2f}, Stddev = {:.2f}, N = {:.2e})".format(label, mean, stddev, N)
        self.ax.hist(data, bins=100, range=[-10, 10], label=l, alpha=0.7)

        self.ax.set_xlabel("Pedestal-Subtracted Residuals")
        self.ax.set_ylabel("N")
        # self.ax.xaxis.set_major_locator(MultipleLocator(16))

    def save(self, output_path=None):
        self.ax.legend(loc="upper left", fontsize=5)
        super().save(output_path)


class CalibStages(Tool):
    name = "CalibStages"
    description = "Plot the different stages in the GCT calibration"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None

        self.p_wf_raw = None
        self.p_wf_pedsub = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        filepath = '/Volumes/gct-jason/data_checs/tf/ac_tf_tmSN0074/Pedestals.tio'
        self.reader = TargetioFileReader(input_path=filepath, **kwargs)

        self.r1 = TargetioR1Calibrator(pedestal_path='/Volumes/gct-jason/data_checs/tf/ac_tf_tmSN0074/Pedestals.tcal',
                                       **kwargs,
                                       )

        p_kwargs = kwargs
        p_kwargs['script'] = "pedestal_residuals"
        p_kwargs['figure_name'] = "wf_raw"
        self.p_wf_raw = WaveformPlotter(**p_kwargs)
        p_kwargs['figure_name'] = "wf_pedsub"
        self.p_wf_pedsub = WaveformPlotter(**p_kwargs)
        p_kwargs['figure_name'] = "residuals"
        self.p_residuals = HistPlotter(**p_kwargs)
        p_kwargs['figure_name'] = "residuals_grouped"
        self.p_residuals_grouped = HistPlotter(**p_kwargs)

    def start(self):
        event_index = 227103
        pix = 3

        event = self.reader.get_event(event_index)
        telid = list(event.r0.tels_with_data)[0]

        self.r1.calibrate(event)

        r0 = event.r0.tel[telid].adc_samples[0]
        r1 = event.r1.tel[telid].pe_samples[0]

        r0_pix = r0[pix]
        r1_pix = r1[pix]

        t = "Raw, pix={}".format(pix)
        self.p_wf_raw.create(r0_pix, t, 'ADC')
        t = "R1 Pedestal Subtracted, pix={}".format(pix)
        self.p_wf_pedsub.create(r1_pix, t, 'ADC Pedestal-Subtracted')
        self.p_wf_pedsub.ax.axes.set_ylim(-20, 20)

        n_events = self.reader.num_events
        n_pixels, n_samples = r0.shape
        source = self.reader.read()
        desc = "Looping through events"

        r1_container = np.zeros((n_events, n_pixels, n_samples))

        for event in tqdm(source, desc=desc, total=n_events):
            ev = event.count
            self.r1.calibrate(event)
            r1_container[ev] = event.r1.tel[telid].pe_samples[0]

        self.log.info("Creating Residual plot")
        self.p_residuals.add(r1_container, "")

        size = 40000
        desc = "Looping through grouped residual plots"
        for i in trange(n_events//size, desc=desc):
            lower = i*size
            upper = (i+1)*size
            sl = np.s_[lower:upper]
            if upper > n_events:
                sl = np.s_[lower:]
            self.p_residuals_grouped.add(r1_container[sl], str(i))

    def finish(self):
        self.p_wf_raw.save()
        self.p_wf_pedsub.save()
        self.p_residuals.save()
        self.p_residuals_grouped.save()


exe = CalibStages()
exe.run()
