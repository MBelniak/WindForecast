import itertools
import os
from typing import Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import wandb
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.config import process_config
from datetime import datetime

from wind_forecast.util.logging import log

marker = itertools.cycle(('+', '.', 'o', '*', 'x', 'v', 'D'))

TICK_FONTSIZE = 18
LABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 20


def run_analysis(config: Config):
    analysis_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'config', 'analysis',
                                 config.analysis.input_file)
    analysis_config = process_config(analysis_file)
    entity = os.getenv('WANDB_ENTITY', '')
    project = os.getenv('WANDB_PROJECT', '')
    run_summaries = []
    run_configs = []
    for run in analysis_config.runs:
        run_id = run['id']
        api = wandb.Api()
        wandb_run = api.run(f"{entity}/{project}/{run_id}")
        run_summaries.append(wandb_run.summary)
        run_configs.append(wandb_run.config)

    log.info('Plotting series analyses for runs:')
    [log.info('\t- ' + run['id']) for run in analysis_config.runs]
    runs_analysis_plotter = RunsAnalysisPlotter(analysis_config.runs, run_summaries, config)
    runs_analysis_plotter.plot_series_comparison()
    runs_analysis_plotter.plot_rmse_by_step_comparison()
    runs_analysis_plotter.plot_mase_by_step_comparison()
    plot_gfs_corr_comparison()


class RunsAnalysisPlotter:
    def __init__(self, analysis_config_runs: List, run_summaries: List[Any], experiment_config: Config):
        self.analysis_config_runs = analysis_config_runs
        self.run_summaries = run_summaries
        self.experiment_config = experiment_config
        self.truth_series = run_summaries[0]['plot_truth']
        self.all_dates = run_summaries[0]['plot_all_dates']
        self.prediction_dates = run_summaries[0]['plot_prediction_dates']
        self.target_mean = run_summaries[0]['synop_mean'][experiment_config.experiment.target_parameter]
        self.target_std = run_summaries[0]['synop_std'][experiment_config.experiment.target_parameter]

    @staticmethod
    def plot_line(ax, x, y, run):
        ax.plot(x, y, linewidth=4 if run['axis_label'] == 'GFS' else 1, label=run['axis_label'], marker=next(marker))

    def plot_series_comparison(self):
        for series_index in range(len(self.truth_series)):
            fig, ax = plt.subplots(figsize=(30, 15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            truth = (np.array(self.truth_series[series_index]) * self.target_std + self.target_mean).tolist()
            ax.plot([datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in self.all_dates[series_index]],
                    truth, label='Wartość rzeczywista', linewidth=4)

            for index, run in enumerate(self.run_summaries):
                prediction_series = run['plot_prediction'][series_index]
                prediction_series = (np.array(prediction_series) * self.target_std + self.target_mean).tolist()
                self.plot_line(ax, [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in self.prediction_dates[series_index]],
                               prediction_series, self.analysis_config_runs[index])

            middle_date = datetime.strptime(self.prediction_dates[series_index][-24], '%Y-%m-%dT%H:%M:%S')
            plt.plot([middle_date, middle_date], [ax.get_ylim()[0], ax.get_ylim()[1]], linewidth=2, color='red',
                     linestyle='dashed')
            ax.annotate('t=T+1', xy=(.5, .85), xycoords='figure fraction', fontsize=22)

            ax.set_ylabel(self.experiment_config.analysis.target_parameter, fontsize=LABEL_FONTSIZE)
            ax.set_xlabel('Data', fontsize=LABEL_FONTSIZE)
            ax.legend(loc='best', prop={'size': LEGEND_FONTSIZE})
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            plt.gcf().autofmt_xdate()
            os.makedirs('analysis', exist_ok=True)
            plt.savefig(f'analysis/series_comparison_{series_index}.png')
            plt.close()

        log.info(f'Series comparisons logged to {os.environ["RUN_DIR"]}/analysis directory')

    def plot_rmse_by_step_comparison(self):
        fig, ax = plt.subplots(figsize=(30, 15))

        for index, run in enumerate(self.run_summaries):
            rmse_by_step = run['rmse_by_step']
            self.plot_line(ax, np.arange(1, len(rmse_by_step) + 1),
                           rmse_by_step, self.analysis_config_runs[index])

        ax.set_ylabel('RMSE', fontsize=LABEL_FONTSIZE)
        ax.set_xlabel('Krok', fontsize=LABEL_FONTSIZE)
        ax.legend(loc='best', prop={'size': LEGEND_FONTSIZE})
        plt.xticks([1, 5, 10, 15, 20, 24])
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        plt.tight_layout()

        os.makedirs('analysis', exist_ok=True)
        plt.savefig(f'analysis/rmse_by_step_comparison.png')
        plt.close()
        log.info(f'RMSE comparison logged to {os.environ["RUN_DIR"]}/analysis directory')

    def plot_mase_by_step_comparison(self):
        fig, ax = plt.subplots(figsize=(30, 15))

        for index, run in enumerate(self.run_summaries):
            mase_by_step = run['mase_by_step']

            self.plot_line(ax, np.arange(1, len(mase_by_step) + 1),
                           mase_by_step, self.analysis_config_runs[index])

        ax.set_ylabel('MASE', fontsize=LABEL_FONTSIZE)
        ax.set_xlabel('Krok', fontsize=LABEL_FONTSIZE)
        ax.legend(loc='best', prop={'size': LEGEND_FONTSIZE})
        plt.xticks([1, 5, 10, 15, 20, 24])
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, labelright=True)
        plt.tight_layout()
        os.makedirs('analysis', exist_ok=True)
        plt.savefig(f'analysis/mase_by_step_comparison.png')
        log.info(f'MASE comparison logged to {os.environ["RUN_DIR"]}/analysis directory')
        plt.close()


def plot_gfs_corr_comparison():
    # for now hardcoded
    labels = ['LSTM', 'BiLSTM', "TCN", "TCN-Attention", "Transformer",
              "Spacetimeformer", 'N-BEATSx', "Regresja liniowa", "ARIMAX"]

    temp_corrs = [0.8438, 0.8215, 0.8426, 0.8551, 0.8088, 0.9629, 0.8067, 0.9591, 0.9273]
    wind_corrs = [0.5364, 0.5118, 0.5452, 0.5804, 0.516, 0.7844, 0.5094, 0.822, 0.6554]
    pres_corrs = [0.8576, 0.8618, 0.8627, 0.8612, 0.8664, 0.9346, 0.852, 0.8595, 0.9628]
    x = np.arange(len(labels))
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(25, 11))
    rects1 = ax.bar(x - width, temp_corrs, width, label='Temperatura')
    rects2 = ax.bar(x, wind_corrs, width, label='Prędkość wiatru')
    rects3 = ax.bar(x + width, pres_corrs, width, label='Ciśnienie')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Korelacja', fontsize=LABEL_FONTSIZE)
    plt.tick_params(labelsize=TICK_FONTSIZE)
    plt.xticks(x, labels)

    ax.legend(fontsize=16)

    plt.tight_layout()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    os.makedirs('analysis', exist_ok=True)
    plt.savefig(f'analysis/gfs_corr.png')
    log.info(f'GFS correlation comparison logged to {os.environ["RUN_DIR"]}/analysis directory')
    plt.close()
