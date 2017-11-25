from os.path import exists, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
from math import ceil

from traitlets import Dict, List
from ctapipe.core import Tool

from targetpipe.fitting.spe_sipm import sipm_spe_fit
from targetpipe.fitting.chec import CHECSSPEFitter, CHECSSPEMultiFitter
from targetpipe.plots.official import ThesisPlotter

from IPython import embed


def get_params(lambda_=1):
    params = dict(
        norm=1000,
        eped=-0.6,
        eped_sigma=0.4,
        spe=1.4,
        spe_sigma=0.2,
        lambda_=lambda_,
        opct=0.6,
        pap=0.3,
        dap=0.4
    )
    return params.copy()


def get_params_multi(params1, params2, params3):
    params_multi = dict(
        norm1=params1['norm'],
        norm2=params2['norm'],
        norm3=params3['norm'],
        eped=params1['eped'],
        eped_sigma=params1['eped_sigma'],
        spe=params1['spe'],
        spe_sigma=params1['spe_sigma'],
        lambda_1=params1['lambda_'],
        lambda_2=params2['lambda_'],
        lambda_3=params3['lambda_'],
        opct=params1['opct'],
        pap=params1['pap'],
        dap=params1['dap']
    )
    return params_multi.copy()


def get_initial(lambda_=1):
    params = dict(
        norm=None,
        eped=-0.5,
        eped_sigma=0.5,
        spe=1,
        spe_sigma=0.1,
        lambda_=lambda_,
        opct=0.5,
        pap=0.5,
        dap=0.5
    )
    return params.copy()


def get_initial_multi(initial1, initial2, initial3):
    params_multi = dict(
        norm1=initial1['norm'],
        norm2=initial2['norm'],
        norm3=initial3['norm'],
        eped=initial1['eped'],
        eped_sigma=initial1['eped_sigma'],
        spe=initial1['spe'],
        spe_sigma=initial1['spe_sigma'],
        lambda_1=initial1['lambda_'],
        lambda_2=initial2['lambda_'],
        lambda_3=initial3['lambda_'],
        opct=initial1['opct'],
        pap=initial1['pap'],
        dap=initial1['dap']
    )
    return params_multi.copy()


def sample_distribution(x, params, n=30000):
    y = sipm_spe_fit(x, **params)
    samples = np.random.choice(x, 30000, p=y / y.sum())
    return samples, y


class FitPlotter(ThesisPlotter):
    def __init__(self, config, tool, **kwargs):
        super().__init__(config, tool, **kwargs)
        self.figures = dict()

    def plot(self):
        # Create the fitters
        range_ = [-3, 10]
        nbins = 100
        fitter1 = CHECSSPEFitter(self.config, self.parent)
        fitter1.range = range_
        fitter1.nbins = nbins
        fitter1.initial = get_initial(1)
        fitter2 = CHECSSPEFitter(self.config, self.parent)
        fitter2.range = range_
        fitter2.nbins = nbins
        fitter2.initial = get_initial(2)
        fitter3 = CHECSSPEFitter(self.config, self.parent)
        fitter3.range = range_
        fitter3.nbins = nbins
        fitter3.initial = get_initial(3)
        fitter_multi = CHECSSPEMultiFitter(self.config, self.parent)
        fitter_multi.range = range_
        fitter_multi.nbins = nbins
        fitter_multi.initial = get_initial_multi(fitter1.initial, fitter2.initial, fitter3.initial)

        # Generate the functions
        found_good = False
        found_bad = False
        i = 0
        while not found_good or not found_bad:
            self.log.info("FitPlotter: Attempt {}".format(i))
            i += 1
            params1 = get_params(1.2)
            params2 = get_params(1.7)
            params3 = get_params(3.1)
            x = np.linspace(-3, 10, 1000)
            samples1, y1 = sample_distribution(x, params1)
            samples2, y2 = sample_distribution(x, params2)
            samples3, y3 = sample_distribution(x, params3)
            params_multi = get_params_multi(params1, params2, params3)

            fitter1.apply(samples1)
            p1 = fitter1.p_value
            fitter2.apply(samples2)
            p2 = fitter2.p_value
            fitter3.apply(samples3)
            p3 = fitter3.p_value
            fitter_multi.apply_multi(samples1, samples2, samples3)
            pm = fitter_multi.p_value

            print(pm, p1, p2, p3)
            if (pm > p1) & (pm > p2) & (pm > p3) & (p1 < 0.0001):
                if found_good:
                    continue
                self.log.info("FitPlotter: Found good")
                found_good = True
                desc = "good"
            elif (pm < 0.001) & (p3 > 0.001):
                if found_bad:
                    continue
                self.log.info("FitPlotter: Found bad")
                found_bad = True
                desc = "bad"
            else:
                continue

            fig_individual = plt.figure(figsize=(13, 6))
            fig_individual.suptitle("Individual Fit")
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax1_t = plt.subplot2grid((3, 2), (0, 1))
            ax2 = plt.subplot2grid((3, 2), (1, 0))
            ax2_t = plt.subplot2grid((3, 2), (1, 1))
            ax3 = plt.subplot2grid((3, 2), (2, 0))
            ax3_t = plt.subplot2grid((3, 2), (2, 1))

            self.individual_plot(x, y1, params1, samples1, fitter1, ax1, ax1_t, True)
            self.individual_plot(x, y2, params2, samples2, fitter2, ax2, ax2_t)
            self.individual_plot(x, y3, params3, samples3, fitter3, ax3, ax3_t)
            name = "fit_" + desc + "_individual"
            self.figures[name] = fig_individual

            fig_multi = plt.figure(figsize=(13, 6))
            fig_multi.suptitle("Multi Fit")
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (1, 0))
            ax3 = plt.subplot2grid((3, 2), (2, 0))
            ax_mt = plt.subplot2grid((3, 2), (0, 1), rowspan=3)

            self.multi_plot(x, [y1, y2, y3], params_multi, [samples1, samples2, samples3], fitter_multi, [ax1, ax2, ax3], ax_mt)
            name = "fit_" + desc + "_multi"
            self.figures[name] = fig_multi

    def save(self, output_path=None):
        for name, fig in self.figures.items():
            self.fig = fig
            self.figure_name = name
            super().save(output_path)

    @staticmethod
    def individual_plot(x, y, params, samples, fitter, ax_p, ax_t, legend=False):
        hist = fitter.hist
        edges = fitter.edges
        between = fitter.between
        coeff = fitter.coeff.copy()
        coeffl = fitter.coeff_list.copy()
        initial = fitter.initial.copy()
        fit = fitter.fit_function(x, **coeff)
        rc2 = fitter.reduced_chi2
        pval = fitter.p_value

        ax_p.plot(x, y, label="Base")
        ax_p.hist(between, bins=edges, weights=hist, histtype='step', label="Hist")
        ax_p.plot(x, fit, label="Fit")
        td = [['%.3f' % params[i], initial[i], '%.3f' % coeff[i]] for i in coeffl]
        td.append(["", "", '%.3g' % rc2])
        td.append(["", "", '%.3g' % pval])
        tr = coeffl
        tr.append("Reduced Chi^2")
        tr.append("P-Value")
        tc = ['Base', 'Initial', 'Fit']
        ax_t.axis('off')
        table = ax_t.table(cellText=td, rowLabels=tr, colLabels=tc, loc='center')
        table.set_fontsize(6)
        table.scale(0.7, 0.7)
        if legend:
            ax_p.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)

    @staticmethod
    def multi_plot(x, y_list, params, samples_list, fitter, ax_list, ax_t):
        y1, y2, y3 = y_list
        samples1, samples2, samples3 = samples_list
        ax1, ax2, ax3 = ax_list

        hist1, hist2, hist3 = fitter.hist
        edges = fitter.edges
        between = fitter.between
        coeff = fitter.coeff.copy()
        coeffl = fitter.coeff_list.copy()
        initial = fitter.initial.copy()
        fit1, fit2, fit3 = fitter.fit_function(x, **coeff)
        rc2 = fitter.reduced_chi2
        pval = fitter.p_value

        ax1.plot(x, y1, label="Base")
        ax1.hist(between, bins=edges, weights=hist1, histtype='step', label="Hist")
        ax1.plot(x, fit1, label="Fit")
        ax1.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)

        ax2.plot(x, y2, label="Base")
        ax2.hist(between, bins=edges, weights=hist2, histtype='step', label="Hist")
        ax2.plot(x, fit2, label="Fit")

        ax3.plot(x, y3, label="Base")
        ax3.hist(between, bins=edges, weights=hist3, histtype='step', label="Hist")
        ax3.plot(x, fit3, label="Fit")

        ax_t.axis('off')
        td = [['%.3f' % params[i], initial[i], '%.3f' % coeff[i]] for i in coeffl]
        td.append(["", "", '%.3g' % rc2])
        td.append(["", "", '%.3g' % pval])
        tr = coeffl
        tr.append("Reduced Chi^2")
        tr.append("P-Value")
        tc = ['Base', 'Initial', 'Fit']
        table = ax_t.table(cellText=td, rowLabels=tr, colLabels=tc, loc='center')
        table.set_fontsize(6)


class NoInitialPlotter(ThesisPlotter):
    def __init__(self, config, tool, **kwargs):
        super().__init__(config, tool, **kwargs)
        self.figures = dict()
        self.dataset_path = self.output_path + "_data.h5"
        self.initial1 = 1
        self.initial2 = 1
        self.initial3 = 1

        self.figures = {}

    def plot(self):
        df = self.load_dataset()
        df = df[df > 0.01].groupby('x').count().reset_index()
        x = df['x']
        y1 = df['p1']
        y2 = df['p2']
        y3 = df['p3']
        ym = df['pm']

        x = ['%.3f\n%.3f\n%.3f\n' % (i[0], i[1], i[2]) for i in x]

        self.fig, self.ax = self.create_figure()
        self.add_points(x, y1, "Individual1")
        self.add_points(x, y2, "Individual2")
        self.add_points(x, y3, "Individual3")
        self.add_points(x, ym, "Multi")
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Number of signficant p-values")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_p"] = self.fig

    def add_points(self, x, y, label, p='-'):
        x_i = np.arange(len(x))
        self.ax.plot(x_i, y, p, label=label)

        self.ax.set_xticks(x_i)
        self.ax.set_xticklabels(x)

    def add_points_err(self, x, y, y_err, label):
        x_i = np.arange(len(x))
        (_, caps, _) = self.ax.errorbar(x_i, y, xerr=None, yerr=y_err, fmt='o',
                                        mew=0.5, label=label,
                                        markersize=3, capsize=3)
        for cap in caps:
            cap.set_markeredgewidth(1)

        self.ax.set_xticks(x_i)
        self.ax.set_xticklabels(x)

    def save(self, output_path=None):
        for name, fig in self.figures.items():
            self.figure_name = name
            self.fig = fig
            super().save(output_path)

    def load_dataset(self):
        if exists(self.dataset_path):
            store = pd.HDFStore(self.dataset_path)
            df = store['df']
        else:
            df = self.create_dataset()
            store = pd.HDFStore(self.dataset_path)
            store['df'] = df
        return df

    def create_dataset(self):
        df_list = []

        # Create the fitters
        range_ = [-3, 10]
        nbins = 100
        fitter1 = CHECSSPEFitter(self.config, self.parent)
        fitter1.range = range_
        fitter1.nbins = nbins
        fitter1.initial = get_initial(1)
        fitter2 = CHECSSPEFitter(self.config, self.parent)
        fitter2.range = range_
        fitter2.nbins = nbins
        fitter2.initial = get_initial(1)
        fitter3 = CHECSSPEFitter(self.config, self.parent)
        fitter3.range = range_
        fitter3.nbins = nbins
        fitter3.initial = get_initial(1)
        fitter_multi = CHECSSPEMultiFitter(self.config, self.parent)
        fitter_multi.range = range_
        fitter_multi.nbins = nbins
        fitter_multi.initial = get_initial_multi(fitter1.initial, fitter2.initial, fitter3.initial)

        lambda_1 = np.linspace(0.3, 1.5, 10)
        lambda_2 = np.linspace(0.5, 3, 10)
        lambda_3 = np.linspace(0.7, 4.5, 10)
        for i in trange(10):
            params1 = get_params(lambda_1[i])
            params2 = get_params(lambda_2[i])
            params3 = get_params(lambda_3[i])
            params_multi = get_params_multi(params1, params2, params3)
            x = np.linspace(-3, 10, 1000)
            for j in trange(100):
                samples1, y1 = sample_distribution(x, params1)
                samples2, y2 = sample_distribution(x, params2)
                samples3, y3 = sample_distribution(x, params3)

                fitter1.apply(samples1)
                p1 = fitter1.p_value
                fitter2.apply(samples2)
                p2 = fitter2.p_value
                fitter3.apply(samples3)
                p3 = fitter3.p_value
                fitter_multi.apply_multi(samples1, samples2, samples3)
                pm = fitter_multi.p_value

                df_list.append(dict(x=(lambda_1[i], lambda_2[i], lambda_3[i]),
                                    p1=p1, p2=p2, p3=p3, pm=pm))

        df = pd.DataFrame(df_list)
        return df


class WithInitialPlotter(NoInitialPlotter):
    def create_dataset(self):
        df_list = []

        # Create the fitters
        range_ = [-3, 10]
        nbins = 100
        fitter1 = CHECSSPEFitter(self.config, self.parent)
        fitter1.range = range_
        fitter1.nbins = nbins
        fitter2 = CHECSSPEFitter(self.config, self.parent)
        fitter2.range = range_
        fitter2.nbins = nbins
        fitter3 = CHECSSPEFitter(self.config, self.parent)
        fitter3.range = range_
        fitter3.nbins = nbins
        fitter_multi = CHECSSPEMultiFitter(self.config, self.parent)
        fitter_multi.range = range_
        fitter_multi.nbins = nbins

        lambda_1 = np.linspace(0.3, 1.5, 10)
        lambda_2 = np.linspace(0.5, 3, 10)
        lambda_3 = np.linspace(0.7, 4.5, 10)
        for i in trange(10):
            params1 = get_params(lambda_1[i])
            params2 = get_params(lambda_2[i])
            params3 = get_params(lambda_3[i])
            fitter1.initial = get_initial(round(lambda_1[i]))
            fitter2.initial = get_initial(round(lambda_2[i]))
            fitter3.initial = get_initial(round(lambda_3[i]))
            fitter_multi.initial = get_initial_multi(fitter1.initial,
                                                     fitter2.initial,
                                                     fitter3.initial)
            params_multi = get_params_multi(params1, params2, params3)
            x = np.linspace(-3, 10, 1000)
            for j in trange(100):
                samples1, y1 = sample_distribution(x, params1)
                samples2, y2 = sample_distribution(x, params2)
                samples3, y3 = sample_distribution(x, params3)

                fitter1.apply(samples1)
                p1 = fitter1.p_value
                fitter2.apply(samples2)
                p2 = fitter2.p_value
                fitter3.apply(samples3)
                p3 = fitter3.p_value
                fitter_multi.apply_multi(samples1, samples2, samples3)
                pm = fitter_multi.p_value

                df_list.append(dict(x=(lambda_1[i], lambda_2[i], lambda_3[i]),
                                    p1=p1, p2=p2, p3=p3, pm=pm))

        df = pd.DataFrame(df_list)
        return df


class CeilInitialPlotter(NoInitialPlotter):
    def plot(self):
        super().plot()

        df = self.load_dataset()
        u_i, u = pd.factorize(df['x'])
        df['x_i'] = u_i

        def rmse(true, fit):
            fit = fit[~np.isnan(fit)]
            n = fit.count()
            return np.sqrt(np.sum(np.power(true - fit, 2)) / n)

        def rmse_df(row):
            lambda_1 = rmse(row['x'].iloc[0][0], row['rlambda_1'])
            lambda_2 = rmse(row['x'].iloc[0][1], row['rlambda_2'])
            lambda_3 = rmse(row['x'].iloc[0][2], row['rlambda_3'])
            lambda_m1 = rmse(row['x'].iloc[0][0], row['rlambda_m1'])
            lambda_m2 = rmse(row['x'].iloc[0][1], row['rlambda_m2'])
            lambda_m3 = rmse(row['x'].iloc[0][2], row['rlambda_m3'])
            opct_1 = rmse(row['opct'].iloc[0], row['ropct_1'])
            opct_2 = rmse(row['opct'].iloc[0], row['ropct_2'])
            opct_3 = rmse(row['opct'].iloc[0], row['ropct_3'])
            opct_m = rmse(row['opct'].iloc[0], row['ropct_m'])
            pap_1 = rmse(row['pap'].iloc[0], row['rpap_1'])
            pap_2 = rmse(row['pap'].iloc[0], row['rpap_2'])
            pap_3 = rmse(row['pap'].iloc[0], row['rpap_3'])
            pap_m = rmse(row['pap'].iloc[0], row['rpap_m'])

            return dict(
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                lambda_m1=lambda_m1,
                lambda_m2=lambda_m2,
                lambda_m3=lambda_m3,
                opct_1=opct_1,
                opct_2=opct_2,
                opct_3=opct_3,
                opct_m=opct_m,
                pap_1=pap_1,
                pap_2=pap_2,
                pap_3=pap_3,
                pap_m=pap_m
            )

        data = df.groupby('x').apply(rmse_df)
        df_list = []
        for index, d in zip(data.index, data):
            d['x'] = index
            df_list.append(d)
        df_rmse = pd.DataFrame(df_list)
        x = df_rmse['x']
        x = ['%.3f\n%.3f\n%.3f\n' % (i[0], i[1], i[2]) for i in x]

        self.fig, self.ax = self.create_figure()
        self.add_points(x, df_rmse['lambda_1'], "Individual1")
        self.add_points(x, df_rmse['lambda_2'], "Individual2")
        self.add_points(x, df_rmse['lambda_3'], "Individual3")
        self.add_points(x, df_rmse['lambda_m1'], "Multi1")
        self.add_points(x, df_rmse['lambda_m2'], "Multi2")
        self.add_points(x, df_rmse['lambda_m3'], "Multi3")
        print("Lambda Multi1:", df_rmse['lambda_m1'][3])
        print("Lambda Multi2:", df_rmse['lambda_m2'][3])
        print("Lambda Multi3:", df_rmse['lambda_m3'][3])
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Root-Mean-Square Error")
        self.ax.set_title("Lambda")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_lambda"] = self.fig

        self.fig, self.ax = self.create_figure()
        self.add_points(x, df_rmse['opct_1'], "Individual1")
        self.add_points(x, df_rmse['opct_2'], "Individual2")
        self.add_points(x, df_rmse['opct_3'], "Individual3")
        self.add_points(x, df_rmse['opct_m'], "Multi")
        print("opct Multi:", df_rmse['opct_m'][3])
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Root-Mean-Square Error")
        self.ax.set_title("Optical Crosstalk Probability")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_opct"] = self.fig

        self.fig, self.ax = self.create_figure()
        self.add_points(x, df_rmse['pap_1'], "Individual1")
        self.add_points(x, df_rmse['pap_2'], "Individual2")
        self.add_points(x, df_rmse['pap_3'], "Individual3")
        self.add_points(x, df_rmse['pap_m'], "Multi")
        print("pap Multi:", df_rmse['pap_m'][3])
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Root-Mean-Square Error")
        self.ax.set_title("After-pulse Probability")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_pap"] = self.fig

        self.fig, self.ax = self.create_figure()
        x0 = df['x'].values
        x = [i[0] for i in x0]
        y = df['rlambda_1'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Individual1")
        x = [i[1] for i in x0]
        y = df['rlambda_2'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Individual2")
        x = [i[2] for i in x0]
        y = df['rlambda_3'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Individual3")
        x = [i[0] for i in x0]
        y = df['rlambda_m1'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Multi1")
        x = [i[1] for i in x0]
        y = df['rlambda_m2'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Multi2")
        x = [i[2] for i in x0]
        y = df['rlambda_m3'].values
        self.ax.plot(x, y, 'x', mew=0.5, label="Multi3")
        x = np.linspace(0, 5, 100)
        y = np.linspace(0, 5, 100)
        self.ax.plot(x, y, ls='--', c='black', lw=0.5)
        self.ax.set_xlabel("Input Value")
        self.ax.set_ylabel("Reconstructed Value")
        self.ax.set_title("Lambda")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_lambda_rich"] = self.fig

        self.fig, self.ax = self.create_figure()
        x = df['x_i'].values
        u_s = ['%.3f\n%.3f\n%.3f\n' % (i[0], i[1], i[2]) for i in u]
        y = df['rlambda_1'].values
        self.ax.plot(x-0.2, y, 'x', mew=0.5, label="Individual1")
        y = df['rlambda_2'].values
        self.ax.plot(x-0.15, y, 'x', mew=0.5, label="Individual2")
        y = df['rlambda_3'].values
        self.ax.plot(x-0.05, y, 'x', mew=0.5, label="Individual3")
        y = df['rlambda_m1'].values
        self.ax.plot(x+0.05, y, 'x', mew=0.5, label="Multi1")
        y = df['rlambda_m2'].values
        self.ax.plot(x+0.15, y, 'x', mew=0.5, label="Multi2")
        y = df['rlambda_m3'].values
        self.ax.plot(x+0.2, y, 'x', mew=0.5, label="Multi3")
        self.ax.set_xticks(np.arange(u.size))
        self.ax.set_xticklabels(u_s)
        r1 = [i[0] for i in u]
        r2 = [i[1] for i in u]
        r3 = [i[2] for i in u]
        x = np.arange(u.size)
        self.ax.plot(x, r1, c='b', lw=1)
        self.ax.plot(x, r2, c='g', lw=1)
        self.ax.plot(x, r3, c='r', lw=1)
        self.ax.plot(x, r1, ls=':', c='purple', lw=1)
        self.ax.plot(x, r2, ls=':', c='yellow', lw=1)
        self.ax.plot(x, r3, ls=':', c='cyan', lw=1)
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Reconstructed Value")
        self.ax.set_title("Lambda")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_lambda_io"] = self.fig

        self.fig, self.ax = self.create_figure()
        x = df['x_i'].values
        u_s = ['%.3f\n%.3f\n%.3f\n' % (i[0], i[1], i[2]) for i in u]
        y = df['ropct_1'].values
        self.ax.plot(x-0.2, y, 'x', mew=0.5, label="Individual1")
        y = df['ropct_2'].values
        self.ax.plot(x-0.1, y, 'x', mew=0.5, label="Individual2")
        y = df['ropct_3'].values
        self.ax.plot(x+0.1, y, 'x', mew=0.5, label="Individual3")
        y = df['ropct_m'].values
        self.ax.plot(x+0.2, y, 'x', mew=0.5, label="Multi")
        self.ax.set_xticks(np.arange(u.size))
        self.ax.set_xticklabels(u_s)
        self.ax.axhline(df['opct'].iloc[0], c='black', lw=0.5)
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Reconstructed Value")
        self.ax.set_title("Optical Crosstalk Probability")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_opct_io"] = self.fig

        self.fig, self.ax = self.create_figure()
        x = df['x_i'].values
        u_s = ['%.3f\n%.3f\n%.3f\n' % (i[0], i[1], i[2]) for i in u]
        y = df['rpap_1'].values
        self.ax.plot(x-0.2, y, 'x', mew=0.5, label="Individual1")
        y = df['rpap_2'].values
        self.ax.plot(x-0.1, y, 'x', mew=0.5, label="Individual2")
        y = df['rpap_3'].values
        self.ax.plot(x+0.1, y, 'x', mew=0.5, label="Individual3")
        y = df['rpap_m'].values
        self.ax.plot(x+0.2, y, 'x', mew=0.5, label="Multi")
        self.ax.set_xticks(np.arange(u.size))
        self.ax.set_xticklabels(u_s)
        self.ax.axhline(df['pap'].iloc[0], c='black', lw=0.5)
        self.ax.set_xlabel("lambda")
        self.ax.set_ylabel("Reconstructed Value")
        self.ax.set_title("After-pulse Probability")
        self.ax.legend(loc=1, frameon=True, fancybox=True, framealpha=0.7)
        self.figures[self.figure_name + "_pap_io"] = self.fig

    def create_dataset(self):
        df_list = []

        # Create the fitters
        range_ = [-3, 10]
        nbins = 100
        fitter1 = CHECSSPEFitter(self.config, self.parent)
        fitter1.range = range_
        fitter1.nbins = nbins
        fitter2 = CHECSSPEFitter(self.config, self.parent)
        fitter2.range = range_
        fitter2.nbins = nbins
        fitter3 = CHECSSPEFitter(self.config, self.parent)
        fitter3.range = range_
        fitter3.nbins = nbins
        fitter_multi = CHECSSPEMultiFitter(self.config, self.parent)
        fitter_multi.range = range_
        fitter_multi.nbins = nbins

        lambda_1 = np.linspace(0.3, 1.5, 10)
        lambda_2 = np.linspace(0.5, 3, 10)
        lambda_3 = np.linspace(0.7, 4.5, 10)
        for i in trange(10):
            params1 = get_params(lambda_1[i])
            params2 = get_params(lambda_2[i])
            params3 = get_params(lambda_3[i])
            fitter1.initial = get_initial(ceil(lambda_1[i]))
            fitter2.initial = get_initial(ceil(lambda_2[i]))
            fitter3.initial = get_initial(ceil(lambda_3[i]))
            fitter_multi.initial = get_initial_multi(fitter1.initial,
                                                     fitter2.initial,
                                                     fitter3.initial)
            params_multi = get_params_multi(params1, params2, params3)
            x = np.linspace(-3, 10, 1000)
            for j in trange(100):
                samples1, y1 = sample_distribution(x, params1)
                samples2, y2 = sample_distribution(x, params2)
                samples3, y3 = sample_distribution(x, params3)

                fitter1.apply(samples1)
                p1 = fitter1.p_value
                rlambda_1 = fitter1.coeff['lambda_']
                ropct_1 = fitter1.coeff['opct']
                rpap_1 = fitter1.coeff['pap']
                fitter2.apply(samples2)
                p2 = fitter2.p_value
                rlambda_2 = fitter2.coeff['lambda_']
                ropct_2 = fitter2.coeff['opct']
                rpap_2 = fitter2.coeff['pap']
                fitter3.apply(samples3)
                p3 = fitter3.p_value
                rlambda_3 = fitter3.coeff['lambda_']
                ropct_3 = fitter3.coeff['opct']
                rpap_3 = fitter3.coeff['pap']
                fitter_multi.apply_multi(samples1, samples2, samples3)
                pm = fitter_multi.p_value
                rlambda_m1 = fitter_multi.coeff['lambda_1']
                rlambda_m2 = fitter_multi.coeff['lambda_2']
                rlambda_m3 = fitter_multi.coeff['lambda_3']
                ropct_m = fitter_multi.coeff['opct']
                rpap_m = fitter_multi.coeff['pap']

                df_list.append(dict(
                    x=(lambda_1[i], lambda_2[i], lambda_3[i]),
                    p1=p1, p2=p2, p3=p3, pm=pm,
                    rlambda_1=rlambda_1, rlambda_2=rlambda_2, rlambda_3=rlambda_3,
                    rlambda_m1=rlambda_m1, rlambda_m2=rlambda_m2, rlambda_m3=rlambda_m3,
                    pap=params1['pap'], opct=params1['opct'],
                    rpap_1=rpap_1, rpap_2=rpap_2, rpap_3=rpap_3, rpap_m=rpap_m,
                    ropct_1=ropct_1, ropct_2=ropct_2, ropct_3=ropct_3, ropct_m=ropct_m
                                    ))

        df = pd.DataFrame(df_list)
        return df



class SiPMFitTester(Tool):
    name = "SiPMFitTester"
    description = "Test the SiPM fit"

    aliases = Dict(dict())
    classes = List([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_fit = None
        self.p_noinitial = None
        self.p_withinitial = None
        self.p_ceilinitial = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        script = "test_sipm_fit"
        # self.p_fit = FitPlotter(**kwargs, script=script)
        self.p_noinitial = NoInitialPlotter(**kwargs, script=script, figure_name="noinital")
        self.p_withinitial = WithInitialPlotter(**kwargs, script=script, figure_name="withinital")
        self.p_ceilinitial = CeilInitialPlotter(**kwargs, script=script, figure_name="ceilinital")

    def start(self):
        # self.p_fit.plot()
        # self.p_fit.save()
        self.p_noinitial.plot()
        self.p_noinitial.save()
        self.p_withinitial.plot()
        self.p_withinitial.save()
        self.p_ceilinitial.plot()
        self.p_ceilinitial.save()

    def finish(self):
        pass


if __name__ == '__main__':
    exe = SiPMFitTester()
    exe.run()
