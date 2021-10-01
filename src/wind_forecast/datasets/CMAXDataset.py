import math

import torch
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.cmax_util import get_mean_and_std_cmax, get_min_max_cmax, \
    get_cmax_values_for_sequence, get_cmax_filename_from_offset
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config


class CMAXDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, train_IDs, normalize=True):
        self.config = config
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.dim = config.experiment.cmax_sample_size
        self.normalization_type = config.experiment.cmax_normalization_type
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.use_future_cmax = config.experiment.use_future_cmax

        self.list_IDs = train_IDs

        self.mean, self.std, self.min, self.max = [], [], 0, 0
        self.cmax_values = {}

        if self.normalization_type == NormalizationType.STANDARD:
            if self.use_future_cmax:
                self.cmax_values, self.mean, self.std = get_mean_and_std_cmax(self.list_IDs, self.dim,
                                                                              self.sequence_length,
                                                                              self.future_sequence_length,
                                                                              self.prediction_offset)
            else:
                self.cmax_values, self.mean, self.std = get_mean_and_std_cmax(self.list_IDs, self.dim,
                                                                              self.sequence_length)

        else:
            if self.use_future_cmax:
                self.cmax_values, self.min, self.max = get_min_max_cmax(self.list_IDs, self.sequence_length,
                                                                        self.future_sequence_length,
                                                                        self.prediction_offset)
            else:
                self.cmax_values, self.min, self.max = get_min_max_cmax(self.list_IDs, self.sequence_length)

        self.data = self.list_IDs
        self.normalize = normalize

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data[index]

        x = self.__data_generation(ID)

        return x

    def __data_generation(self, ID):
        # Initialization
        if self.use_future_cmax:
            x = np.empty((self.sequence_length, math.ceil(self.dim[0] / self.config.experiment.cmax_scaling_factor),
                          math.ceil(self.dim[1] / self.config.experiment.cmax_scaling_factor)))

            y = np.empty(
                (self.future_sequence_length, math.ceil(self.dim[0] / self.config.experiment.cmax_scaling_factor),
                 math.ceil(self.dim[1] / self.config.experiment.cmax_scaling_factor)))

            # Generate data
            x[:, ] = get_cmax_values_for_sequence(ID, self.cmax_values, self.sequence_length)
            first_future_id = get_cmax_filename_from_offset(ID, self.sequence_length + self.prediction_offset)
            y[:, ] = get_cmax_values_for_sequence(first_future_id, self.cmax_values, self.future_sequence_length)

            if self.normalize:
                if self.normalization_type == NormalizationType.STANDARD:
                    x[:, ] = (x[:, ] - self.mean) / self.std
                    y[:, ] = (y[:, ] - self.mean) / self.std
                else:
                    x[:, ] = (x[:, ] - self.min) / (self.max - self.min)
                    y[:, ] = (y[:, ] - self.min) / (self.max - self.min)

            return x, y
        else:
            x = np.empty((self.sequence_length, math.ceil(self.dim[0] / self.config.experiment.cmax_scaling_factor),
                          math.ceil(self.dim[1] / self.config.experiment.cmax_scaling_factor)))

            # Generate data
            x[:, ] = get_cmax_values_for_sequence(ID, self.cmax_values, self.sequence_length)
            if self.normalize:
                if self.normalization_type == NormalizationType.STANDARD:
                    x[:, ] = (x[:, ] - self.mean) / self.std
                else:
                    x[:, ] = (x[:, ] - self.min) / (self.max - self.min)
            return x
