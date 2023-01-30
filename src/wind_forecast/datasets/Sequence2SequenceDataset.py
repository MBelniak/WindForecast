from typing import List

from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset


class Sequence2SequenceDataset(BaseDataset):
    SYNOP_PAST_Y_INDEX = 0
    SYNOP_PAST_X_INDEX = 1
    SYNOP_FUTURE_Y_INDEX = 2
    SYNOP_FUTURE_X_INDEX = 3
    DATES_PAST_INDEX = 4
    DATES_FUTURE_INDEX = 5

    def __init__(self, config: Config, synop_data, synop_data_indices, synop_feature_names: List[str]):
        super().__init__()
        'Initialization'
        self.train_params = synop_feature_names
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data
        self.data = synop_data_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        synop_index = self.data[index]
        synop_past_x = self.synop_data.loc[synop_index:synop_index + self.sequence_length - 1][
            self.train_params].to_numpy()
        synop_future_x = self.synop_data.loc[
                         synop_index + self.sequence_length + self.prediction_offset
                         :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1][
            self.train_params].to_numpy()
        synop_y = self.synop_data.loc[
                  synop_index
                  :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1][
            self.target_param].to_numpy()
        synop_past_y = synop_y[:self.sequence_length + self.prediction_offset]
        synop_future_y = synop_y[self.sequence_length + self.prediction_offset
                                 :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]
        past_dates = self.synop_data.loc[synop_index:synop_index + self.sequence_length - 1]['date']
        future_dates = self.synop_data.loc[synop_index + self.sequence_length + self.prediction_offset
                                           :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1]['date']

        return synop_past_y, synop_past_x, synop_future_y, synop_future_x, past_dates, future_dates
