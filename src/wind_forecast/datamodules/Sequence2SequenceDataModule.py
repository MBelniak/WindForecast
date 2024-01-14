import os
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from exploration.exploration import explore_data_for_each_gfs_param, explore_data_bias
from synop.consts import SYNOP_PERIODIC_FEATURES, LOWER_CLOUDS, CLOUD_COVER, TEMPERATURE, VELOCITY_COLUMN, PRESSURE
from util.coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceGFSDataset import Sequence2SequenceGFSDataset
from wind_forecast.datasets.Sequence2SequenceSynopDataset import Sequence2SequenceSynopDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, \
    get_feature_names_after_periodic_reduction
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config
from wind_forecast.util.df_util import normalize_data_for_training, decompose_data, resolve_indices, \
    add_angle_from_sin_cos_to_df
from wind_forecast.util.gfs_util import add_param_to_train_params, \
    GFSUtil, extend_wind_components, decompose_gfs_data, get_gfs_target_param
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

        self.gfs_features_params = process_config(config.experiment.train_parameters_config_file).params
        self.gfs_features_names = [f"{f['name']}_{f['level']}" for f in self.gfs_features_params]
        self.gfs_wind_parameters = ["V GRD_HTGL_10", "U GRD_HTGL_10"]

        self.gfs_target_param = get_gfs_target_param(self.target_param)

        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, self.future_sequence_length,
                                self.prediction_offset, self.gfs_features_params)

        self.periodic_features = config.experiment.synop_periodic_features
        self.uses_future_sequences = True

        self.synop_data = ...
        self.gfs_data = ...
        self.data_indices = ...
        self.synop_mean = ...
        self.synop_std = ...
        self.synop_min = ...
        self.synop_max = ...
        self.gfs_mean = ...
        self.gfs_std = ...
        self.gfs_min = ...
        self.gfs_max = ...
        self.synop_dates = ...

    def prepare_data(self, *args, **kwargs):
        self.load_from_disk(self.config)

        if self.initialized:
            if self.config.experiment._tags_[0] == 'GFS':
                self.eliminate_gfs_bias()
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
        self.data_indices = self.synop_data[self.synop_data["date"].isin(self.synop_dates)].index

        if self.config.experiment.stl_decompose:
            self.synop_decompose()
            features_to_normalize = self.synop_feature_names
        else:
            # do not normalize periodic features
            features_to_normalize = [name for name in self.synop_feature_names if name not in
                                     [*get_feature_names_after_periodic_reduction(
                                         [f['column'][1] for f in SYNOP_PERIODIC_FEATURES])]]

        if self.config.experiment.load_gfs_data:
            self.prepare_dataset_for_gfs()

        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_mean, self.synop_std, self.synop_min, self.synop_max = normalize_data_for_training(
            self.synop_data, self.data_indices, features_to_normalize,
            self.sequence_length + self.prediction_offset + self.future_sequence_length,
            self.normalization_type)

        log.info(f"Synop target mean: {self.synop_mean[self.target_param]}")
        log.info(f"Synop target std: {self.synop_std[self.target_param]}")

    def after_synop_loaded(self):
        self.synop_dates = get_correct_dates_for_sequence(self.synop_data, self.sequence_length,
                                                          self.future_sequence_length,
                                                          self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            self.log_dataset_info()
            return
        if self.get_from_cache(stage):
            self.log_dataset_info()
            return

        if self.config.experiment.load_gfs_data:
            synop_dataset = Sequence2SequenceSynopDataset(self.config, self.synop_data, self.data_indices,
                                                          self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)

            gfs_dataset = Sequence2SequenceGFSDataset(self.config, self.gfs_data, self.data_indices,
                                                      self.gfs_features_names)
            gfs_dataset.set_mean(self.gfs_mean)
            gfs_dataset.set_std(self.gfs_std)
            gfs_dataset.set_min(self.gfs_min)
            gfs_dataset.set_max(self.gfs_max)
            dataset = ConcatDatasets(synop_dataset, gfs_dataset)
        else:
            synop_dataset = Sequence2SequenceSynopDataset(self.config, self.synop_data, self.data_indices,
                                                          self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)
            dataset = synop_dataset

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        self.split_dataset(self.config, dataset, self.sequence_length)
        self.log_dataset_info()

        if self.config.experiment._tags_[0] == 'GFS':
            self.eliminate_gfs_bias()

    def prepare_dataset_for_gfs(self):
        log.info("Preparing the GFS dataset")
        # match GFS and synop sequences
        self.data_indices, self.gfs_data = self.gfs_util.match_gfs_with_synop_sequence2sequence(
            self.synop_data,
            self.data_indices)

        if all([f in self.gfs_features_names for f in self.gfs_wind_parameters]):
            self.gfs_data = self.prepare_gfs_data_with_wind_components(self.gfs_data)

        self.gfs_exploration()

        if self.config.experiment.stl_decompose:
            self.gfs_features_names = self.gfs_decompose()
            features_to_normalize = [*self.gfs_features_names]
        else:
            # do not normalize periodic features
            features_to_normalize = [name for name in self.gfs_features_names if name not in ["wind-sin", "wind-cos"]]

        features_to_normalize.remove(self.gfs_target_param)

        # normalize GFS parameters data
        self.gfs_data, self.gfs_mean, self.gfs_std, self.gfs_min, self.gfs_max = normalize_data_for_training(
            self.gfs_data, self.data_indices, features_to_normalize,
            self.sequence_length + self.prediction_offset + self.future_sequence_length,
            self.normalization_type)

        target_data = self.gfs_data[self.gfs_target_param]

        if self.target_param in [LOWER_CLOUDS[1], CLOUD_COVER[1]]:
            self.gfs_data[self.gfs_target_param] = target_data / 100
        else:
            if self.normalization_type == NormalizationType.STANDARD:
                self.gfs_mean[self.gfs_target_param] = target_data.mean(axis=0)
                self.gfs_std[self.gfs_target_param] = target_data.std(axis=0)
                self.gfs_data[self.gfs_target_param] = (target_data - target_data.mean(axis=0)) / target_data.std(
                    axis=0)
            else:
                self.gfs_min[self.gfs_target_param] = target_data.min(axis=0)
                self.gfs_max[self.gfs_target_param] = target_data.max(axis=0)
                self.gfs_data[self.gfs_target_param] = (target_data - target_data.min(axis=0)) / (
                            target_data.max(axis=0) - target_data.min(axis=0))

        if self.target_param == "wind_direction":
            # TODO handle this case - as for now we do not use this parameter as target
            pass

    def resolve_all_synop_data(self):
        synop_inputs = []
        all_synop_targets = []
        synop_data_dates = self.synop_data['date']
        train_params = list(list(zip(*self.synop_train_params))[1])
        # all_targets and dates - dates are needed for matching the labels against GFS dates
        all_targets_and_labels = pd.concat([synop_data_dates, self.synop_data[train_params]], axis=1)

        for index in tqdm(self.data_indices):
            synop_inputs.append(
                self.synop_data.iloc[index:index + self.sequence_length][[*train_params, 'date']])
            all_synop_targets.append(all_targets_and_labels.iloc[
                                     index + self.sequence_length + self.prediction_offset:index + self.sequence_length + self.prediction_offset + self.future_sequence_length])

        return synop_inputs, all_synop_targets

    def prepare_gfs_data_with_wind_components(self, gfs_data: pd.DataFrame):
        gfs_wind_data = gfs_data[self.gfs_wind_parameters]
        gfs_data.drop(columns=self.gfs_wind_parameters, inplace=True)
        velocity, sin, cos = extend_wind_components(gfs_wind_data.values)
        gfs_data["wind-velocity"] = velocity
        gfs_data["wind-sin"] = sin
        gfs_data["wind-cos"] = cos
        self.gfs_features_names.remove(self.gfs_wind_parameters[0])
        self.gfs_features_names.remove(self.gfs_wind_parameters[1])
        self.gfs_features_names.extend(["wind-velocity", "wind-sin", "wind-cos"])
        return gfs_data

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=len(self.dataset_train) if self.batch_size == 0 else self.batch_size,
                          shuffle=self.shuffle,
                          collate_fn=self.collate_fn, num_workers=self.config.experiment.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=len(self.dataset_val) if self.batch_size == 0 else self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.config.experiment.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=len(self.dataset_test) if self.batch_size == 0 else self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=self.config.experiment.num_workers)

    def collate_fn(self, x: List[Tuple]):
        if self.config.experiment.load_gfs_data:
            synop_data, gfs_data = [item[0] for item in x], [item[1] for item in x]
            synop_data, dates = [[*item[:4], *item[6:]] for item in synop_data], [item[4:6] for item in synop_data]
            all_data = [*default_collate(synop_data), *default_collate(gfs_data), *list(zip(*dates))]
        else:
            synop_data, dates = [[*item[:4], *item[6:]] for item in x], [item[4:6] for item in x]
            all_data = [*default_collate(synop_data), *list(zip(*dates))]

        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.SYNOP_FUTURE_Y.value: all_data[2],
            BatchKeys.SYNOP_FUTURE_X.value: all_data[3],
            BatchKeys.DATES_PAST.value: all_data[-2],
            BatchKeys.DATES_FUTURE.value: all_data[-1]
        }

        if self.config.experiment.load_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_PAST_Y.value] = all_data[5]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[7]

            if self.config.experiment.differential_forecast:
                target_mean = self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset").mean[
                    self.target_param]
                target_std = self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset").std[
                    self.target_param]
                gfs_past_y = dict_data[BatchKeys.GFS_PAST_Y.value] * target_std + target_mean
                gfs_future_y = dict_data[BatchKeys.GFS_FUTURE_Y.value] * target_std + target_mean
                synop_past_y = dict_data[BatchKeys.SYNOP_PAST_Y.value].unsqueeze(-1) * target_std + target_mean
                synop_future_y = dict_data[BatchKeys.SYNOP_FUTURE_Y.value].unsqueeze(-1) * target_std + target_mean
                diff_past = gfs_past_y - synop_past_y
                diff_future = gfs_future_y - synop_future_y
                dict_data[BatchKeys.GFS_SYNOP_PAST_DIFF.value] = diff_past / target_std
                dict_data[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value] = diff_future / target_std

        return dict_data

    def synop_decompose(self):
        target_param_series = self.synop_data[self.target_param]
        self.synop_data = decompose_data(self.synop_data, self.synop_feature_names)
        self.synop_data[self.target_param] = target_param_series
        self.synop_feature_names = list(
            chain.from_iterable(
                (f"{feature}_T", f"{feature}_S", f"{feature}_R") for feature in self.synop_feature_names))
        self.synop_feature_names.append(self.target_param)

    def gfs_decompose(self):
        target_param_series = self.gfs_data[self.gfs_target_param]
        self.gfs_data = decompose_gfs_data(self.gfs_data, self.gfs_features_names)
        self.gfs_data[self.gfs_target_param] = target_param_series
        new_gfs_features_names = list(
            chain.from_iterable(
                (f"{feature}_T", f"{feature}_S", f"{feature}_R") for feature in self.gfs_features_names))
        new_gfs_features_names.append(self.gfs_target_param)
        return new_gfs_features_names

    # we can check what is the mean GFS error and just add it to target values to improve performance. We assume we know only train data
    def eliminate_gfs_bias(self):
        train_concat_dataset = self.dataset_train.dataset
        train_synop_dataset = train_concat_dataset.get_dataset("Sequence2SequenceSynopDataset")
        target_mean = train_synop_dataset.mean[self.target_param]
        target_std = train_synop_dataset.std[self.target_param]

        unscaled_diff = self.get_gfs_train_unscaled_diff()
        self.plot_diff(unscaled_diff, 'train')

        bias = unscaled_diff.mean(axis=0)
        test_synop_dataset = self.dataset_test.dataset.get_dataset("Sequence2SequenceSynopDataset")
        test_synop_data = test_synop_dataset.synop_data
        test_gfs_dataset = self.dataset_test.dataset.get_dataset("Sequence2SequenceGFSDataset")
        test_gfs_data = test_gfs_dataset.gfs_data

        unscaled_gfs_targets = test_gfs_data[self.gfs_target_param] * target_std + target_mean
        fixed_gfs_targets = unscaled_gfs_targets + bias
        test_gfs_data[self.gfs_target_param] = (fixed_gfs_targets - target_mean) / target_std

        test_indices = [test_synop_dataset.data[index] for index in self.dataset_test.indices]
        all_gfs_data = resolve_indices(test_gfs_data, test_indices,
                                       self.sequence_length + self.prediction_offset + self.future_sequence_length)
        gfs_test_targets = all_gfs_data[self.gfs_target_param].values
        unscaled_gfs_test_targets = gfs_test_targets * target_std + target_mean

        all_synop_data = resolve_indices(test_synop_data, test_indices,
                                         self.sequence_length + self.prediction_offset + self.future_sequence_length)
        synop_targets = all_synop_data[self.target_param].values

        test_diff = (synop_targets * target_std + target_mean - unscaled_gfs_test_targets)

        self.plot_diff(test_diff, 'test')

    def get_gfs_train_unscaled_diff(self):
        train_concat_dataset = self.dataset_train.dataset
        train_synop_dataset = train_concat_dataset.get_dataset("Sequence2SequenceSynopDataset")
        train_synop_data = train_synop_dataset.synop_data
        train_gfs_dataset = train_concat_dataset.get_dataset("Sequence2SequenceGFSDataset")
        train_gfs_data = train_gfs_dataset.gfs_data
        target_mean = train_synop_dataset.mean[self.target_param]
        target_std = train_synop_dataset.std[self.target_param]

        train_indices = [train_synop_dataset.data[index] for index in self.dataset_train.indices]
        all_gfs_data = resolve_indices(train_gfs_data, train_indices,
                                       self.sequence_length + self.prediction_offset + self.future_sequence_length)
        gfs_targets = all_gfs_data[self.gfs_target_param].values
        all_synop_data = resolve_indices(train_synop_data, train_indices,
                                         self.sequence_length + self.prediction_offset + self.future_sequence_length)
        synop_targets = all_synop_data[self.target_param].values

        unscaled_gfs_train_targets = gfs_targets * target_std + target_mean

        return synop_targets * target_std + target_mean - unscaled_gfs_train_targets

    def plot_diff(self, diff: list, dataset_name: str):
        plt.figure(figsize=(20, 10))
        plt.tight_layout()
        sns.displot(diff, bins=100, kde=True)
        plt.ylabel('Liczebność')
        plt.xlabel('Różnica')

        os.makedirs(os.path.join(Path(__file__).parent, "plots"), exist_ok=True)
        plt.savefig(
            os.path.join(Path(__file__).parent, "plots",
                         f"gfs_diff_{self.config.experiment.target_parameter}_{dataset_name}.png"),
            dpi=200, bbox_inches='tight')

    def log_dataset_info(self):
        log.info('Dataset train len: ' + str(len(self.dataset_train)))
        log.info('Dataset val len: ' + ('0' if self.dataset_val is None else str(len(self.dataset_val))))
        log.info('Dataset test len: ' + str(len(self.dataset_test)))

        log.info('Dataset train first date: ' +
                 str(self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                         self.dataset_train.indices[0]][4][0]))
        log.info('Dataset train last date: ' +
                 str(self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                         self.dataset_train.indices[-1]][5][-1]))
        if self.dataset_val is not None:
            log.info('Dataset val first date: ' +
                     str(self.dataset_val.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                             self.dataset_val.indices[0]][4][0]))
            log.info('Dataset val last date: ' +
                     str(self.dataset_val.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                             self.dataset_val.indices[-1]][5][-1]))
        log.info('Dataset test first date: ' +
                 str(self.dataset_test.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                         self.dataset_test.indices[0]][4][0]))
        log.info('Dataset test last date: ' +
                 str(self.dataset_test.dataset.get_dataset("Sequence2SequenceSynopDataset")[
                         self.dataset_test.indices[-1]][5][-1]))

    def gfs_exploration(self):
        gfs_data = resolve_indices(self.gfs_data, self.data_indices,
                                   self.sequence_length + self.prediction_offset + self.future_sequence_length)
        synop_data = resolve_indices(self.synop_data, self.data_indices,
                                     self.sequence_length + self.prediction_offset + self.future_sequence_length)
        explore_data_for_each_gfs_param(gfs_data, self.gfs_features_names)
        synop_data = synop_data.rename(
            columns=dict(zip([f[2] for f in self.synop_feature_names], [f[1] for f in self.synop_feature_names])))
        add_angle_from_sin_cos_to_df(synop_data)
        gfs_data[get_gfs_target_param(TEMPERATURE[1])] -= 273.15
        gfs_data[get_gfs_target_param(PRESSURE[1])] /= 100
        explore_data_bias(synop_data, gfs_data,
                          [(TEMPERATURE[1], 'TMP_HTGL_2'),
                           (VELOCITY_COLUMN[1], 'wind-velocity'),
                           (PRESSURE[1], 'PRES_SFC_0'),
                           (LOWER_CLOUDS[1], 'T CDC_LCY_0')
                           ],
                          ['gfs_diff_temperature',
                           'gfs_diff_wind_velocity',
                           'gfs_diff_pressure',
                           'gfs_diff_lower_clouds']
                          )

    def gfs_exploration_test_dataset(self):
        gfs_data = resolve_indices(self.gfs_data, self.data_indices,
                                   self.sequence_length + self.prediction_offset + self.future_sequence_length)
        synop_data = resolve_indices(self.synop_data, self.data_indices,
                                     self.sequence_length + self.prediction_offset + self.future_sequence_length)
        explore_data_for_each_gfs_param(gfs_data, self.gfs_features_names)
        synop_data = synop_data.rename(
            columns=dict(zip([f[2] for f in self.synop_feature_names], [f[1] for f in self.synop_feature_names])))
        add_angle_from_sin_cos_to_df(synop_data)
        gfs_data[get_gfs_target_param(TEMPERATURE[1])] -= 273.15
        gfs_data[get_gfs_target_param(PRESSURE[1])] /= 100
        explore_data_bias(synop_data, gfs_data,
                          [(TEMPERATURE[1], 'TMP_HTGL_2'),
                           (VELOCITY_COLUMN[1], 'wind-velocity'),
                           (PRESSURE[1], 'PRES_SFC_0'),
                           (LOWER_CLOUDS[1], 'T CDC_LCY_0')
                           ],
                          ['gfs_diff_temperature',
                           'gfs_diff_wind_velocity',
                           'gfs_diff_pressure',
                           'gfs_diff_lower_clouds']
                          )
