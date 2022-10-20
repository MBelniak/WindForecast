import math
from itertools import chain
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.Coords import Coords
from synop.consts import SYNOP_PERIODIC_FEATURES
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data_for_training, \
    decompose_synop_data, get_feature_names_after_periodic_reduction
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, target_param_to_gfs_name_level, normalize_gfs_data, \
    GFSUtil, extend_wind_components
from wind_forecast.util.logging import log
from wind_forecast.util.synop_util import get_correct_dates_for_sequence


class Sequence2SequenceDataModule(SplittableDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.config = config
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle

        self.synop_train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(self.synop_train_params, self.target_param)
        self.synop_feature_names = list(list(zip(*all_params))[1])

        self.synop_file = config.experiment.synop_file
        self.synop_from_year = config.experiment.synop_from_year
        self.synop_to_year = config.experiment.synop_to_year
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.normalization_type = config.experiment.normalization_type
        self.prediction_offset = config.experiment.prediction_offset
        coords = config.experiment.target_coords
        self.target_coords = Coords(coords[0], coords[0], coords[1], coords[1])

        self.gfs_train_params = process_config(config.experiment.train_parameters_config_file).params
        self.gfs_target_params = self.gfs_train_params

        self.gfs_target_param_indices = [[{'name': param['name'], 'level': param['level']}
                                          for param in self.gfs_train_params].index(param)
                                         for param in target_param_to_gfs_name_level(self.target_param)]

        self.gfs_wind_components_indices = [[{'name': param['name'], 'level': param['level']}
                                             for param in self.gfs_train_params].index(param)
                                            for param in target_param_to_gfs_name_level('wind_direction')]
        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, self.future_sequence_length,
                                self.prediction_offset, self.gfs_train_params, self.gfs_target_params)

        self.periodic_features = config.experiment.synop_periodic_features
        self.uses_future_sequences = True

        self.synop_data = ...
        self.synop_data_indices = ...
        self.synop_mean = ...
        self.synop_std = ...
        self.synop_dates = ...

    def prepare_data(self, *args, **kwargs):
        self.load_from_disk(self.config)

        if self.initialized:
            return

        self.synop_data = prepare_synop_dataset(self.synop_file,
                                                list(list(zip(*self.synop_train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        if self.config.debug_mode:
            self.synop_data = self.synop_data.head(self.sequence_length * 20)

        self.after_synop_loaded()

        self.synop_feature_names = get_feature_names_after_periodic_reduction(self.synop_feature_names)

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(self.synop_dates)].index

        if self.config.experiment.stl_decompose:
            self.synop_decompose()
            features_to_normalize = self.synop_feature_names
        else:
            # do not normalize periodic features
            features_to_normalize = [name for name in self.synop_feature_names if name not in
                                     get_feature_names_after_periodic_reduction(SYNOP_PERIODIC_FEATURES)]

        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, mean, std = normalize_synop_data_for_training(
            self.synop_data, self.synop_data_indices, features_to_normalize,
            self.sequence_length + self.prediction_offset + self.future_sequence_length,
            self.normalization_type)

        self.synop_mean = mean[self.target_param]
        self.synop_std = std[self.target_param]
        log.info(f"Synop mean: {self.synop_mean}")
        log.info(f"Synop std: {self.synop_std}")

    def after_synop_loaded(self):
        self.synop_dates = get_correct_dates_for_sequence(self.synop_data, self.sequence_length, self.future_sequence_length,
                                               self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            return
        if self.get_from_cache(stage):
            return

        if self.config.experiment.load_gfs_data:
            synop_inputs, gfs_past_y, gfs_past_x, gfs_future_y, gfs_future_x = self.prepare_dataset_for_gfs()

            dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices,
                                                      self.synop_feature_names, gfs_future_y, gfs_past_y,
                                                      gfs_future_x, gfs_past_x)

        else:
            dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices,
                                               self.synop_feature_names)

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        self.split_dataset(self.config, dataset, self.sequence_length)

    def prepare_dataset_for_gfs(self):
        log.info("Preparing the dataset")
        # match GFS and synop sequences
        self.synop_data_indices, gfs_past_x, gfs_future_x = self.gfs_util.match_gfs_with_synop_sequence2sequence(
            self.synop_data,
            self.synop_data_indices)

        # save target data
        gfs_future_y = gfs_future_x[:, :, self.gfs_target_param_indices]
        gfs_past_y = gfs_past_x[:, :, self.gfs_target_param_indices]

        # normalize GFS parameters data
        param_names = [x['name'] for x in self.gfs_train_params]
        if "V GRD" in param_names and "U GRD" in param_names:
            gfs_past_x = self.prepare_gfs_data_with_wind_components(gfs_past_x)
            gfs_future_x = self.prepare_gfs_data_with_wind_components(gfs_future_x)
        else:
            gfs_past_x = normalize_gfs_data(gfs_past_x, self.normalization_type, (0, 1))
            gfs_future_x = normalize_gfs_data(gfs_future_x, self.normalization_type, (0, 1))

        if self.target_param == "wind_direction":
            # set sin and cos components as targets, do not normalize them
            gfs_future_y = np.apply_along_axis(
                lambda velocity: [-velocity[1] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)),
                                  -velocity[0] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2))], -1,
                gfs_future_y)
            gfs_past_y = np.apply_along_axis(
                lambda velocity: [-velocity[1] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)),
                                  -velocity[0] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2))], -1,
                gfs_past_y)
        else:
            if self.target_param == "wind_velocity":
                # handle target wind_velocity forecast by GFS
                # velocity[0] is V GRD (northward), velocity[1] is U GRD (eastward)
                gfs_future_y = np.apply_along_axis(lambda velocity: [math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)],
                                                   -1,
                                                   gfs_future_y)
                gfs_past_y = np.apply_along_axis(lambda velocity: [math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)], -1,
                                                 gfs_past_y)

            elif self.target_param == "temperature":
                K_TO_C = 273.15
                gfs_future_y -= K_TO_C
                gfs_past_y -= K_TO_C
            elif self.target_param == "pressure":
                gfs_future_y /= 100
                gfs_past_y /= 100

            gfs_future_y = (gfs_future_y - self.synop_mean) / self.synop_std
            gfs_past_y = (gfs_past_y - self.synop_mean) / self.synop_std

        assert len(self.synop_data_indices) == len(
            gfs_future_x), f"len(all_gfs_target_data) should be {len(self.synop_data_indices)} but was {len(gfs_future_x)}"

        assert len(self.synop_data_indices) == len(
            gfs_past_x), f"len(all_gfs_input_data) should be {len(self.synop_data_indices)} but was {len(gfs_past_x)}"
        return self.synop_data_indices, gfs_past_y, gfs_past_x, gfs_future_y, gfs_future_x

    def resolve_all_synop_data(self):
        synop_inputs = []
        all_synop_targets = []
        synop_data_dates = self.synop_data['date']
        train_params = list(list(zip(*self.synop_train_params))[1])
        # all_targets and dates - dates are needed for matching the labels against GFS dates
        all_targets_and_labels = pd.concat([synop_data_dates, self.synop_data[train_params]], axis=1)

        for index in tqdm(self.synop_data_indices):
            synop_inputs.append(
                self.synop_data.iloc[index:index + self.sequence_length][[*train_params, 'date']])
            all_synop_targets.append(all_targets_and_labels.iloc[
                                     index + self.sequence_length + self.prediction_offset:index + self.sequence_length + self.prediction_offset + self.future_sequence_length])

        return synop_inputs, all_synop_targets

    def prepare_gfs_data_with_wind_components(self, gfs_data: np.ndarray):
        gfs_data = np.delete(gfs_data, self.gfs_wind_components_indices, -1)
        velocity, sin, cos = extend_wind_components(gfs_data[:, :, self.gfs_wind_components_indices])
        gfs_data = normalize_gfs_data(np.concatenate([gfs_data, np.expand_dims(velocity, -1)], -1),
                                      self.normalization_type, (0, 1))
        gfs_data = np.concatenate([gfs_data, np.expand_dims(sin, -1), np.expand_dims(cos, -1)], -1)
        return gfs_data

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle,
                          collate_fn=self.collate_fn, num_workers=self.config.experiment.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.config.experiment.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.config.experiment.num_workers)

    def collate_fn(self, x: List[Tuple]):
        variables, dates = [item[:-2] for item in x], [item[-2:] for item in x]
        all_data = [*default_collate(variables), *list(zip(*dates))]
        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.SYNOP_FUTURE_Y.value: all_data[2],
            BatchKeys.SYNOP_FUTURE_X.value: all_data[3]
        }

        if self.config.experiment.load_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_PAST_Y.value] = all_data[5]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[7]
            dict_data[BatchKeys.DATES_PAST.value] = all_data[8]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[9]
            if self.config.experiment.differential_forecast:
                gfs_past_y = dict_data[BatchKeys.GFS_PAST_Y.value] * self.dataset_train.std + self.dataset_train.mean
                gfs_future_y = dict_data[
                                   BatchKeys.GFS_FUTURE_Y.value] * self.dataset_train.std + self.dataset_train.mean
                synop_past_y = dict_data[BatchKeys.SYNOP_PAST_Y.value].unsqueeze(
                    -1) * self.dataset_train.std + self.dataset_train.mean
                synop_future_y = dict_data[BatchKeys.SYNOP_FUTURE_Y.value].unsqueeze(
                    -1) * self.dataset_train.std + self.dataset_train.mean
                diff_past = gfs_past_y - synop_past_y
                diff_future = gfs_future_y - synop_future_y
                dict_data[BatchKeys.GFS_SYNOP_PAST_DIFF.value] = diff_past / self.dataset_train.std
                dict_data[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value] = diff_future / self.dataset_train.std

        else:
            dict_data[BatchKeys.DATES_PAST.value] = all_data[4]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[5]
        return dict_data

    def synop_decompose(self):
        target_param_series = self.synop_data[self.target_param]
        self.synop_data = decompose_synop_data(self.synop_data, self.synop_feature_names)
        self.synop_data[self.target_param] = target_param_series
        self.synop_feature_names = list(
            chain.from_iterable((f"{feature}_T", f"{feature}_S", f"{feature}_R") for feature in self.synop_feature_names))
        self.synop_feature_names.append(self.target_param)
