from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset


class SequenceDataModule(LightningDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__()
        self.config = config
        self.val_split = config.experiment.val_split
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file,
                                                                             list(list(zip(*self.train_params))[1]),
                                                                             dataset_dir=SYNOP_DATASETS_DIRECTORY)

        target_param_index = [x[1] for x in self.train_params].index(self.target_param)
        print(self.label_mean[target_param_index])
        print(self.label_std[target_param_index])

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = SequenceDataset(config=self.config, labels=self.labels, train=True)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = SequenceDataset(config=self.config, labels=self.labels, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)