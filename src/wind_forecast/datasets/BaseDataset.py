import torch


class BaseDataset(torch.utils.data.Dataset):
    SYNOP_PAST_Y_INDEX = ...
    SYNOP_PAST_X_INDEX = ...
    SYNOP_FUTURE_Y_INDEX = ...
    SYNOP_FUTURE_X_INDEX = ...
    GFS_PAST_X_INDEX = ...
    GFS_PAST_Y_INDEX = ...
    GFS_FUTURE_X_INDEX = ...
    GFS_FUTURE_Y_INDEX = ...
    DATES_PAST_INDEX = ...
    DATES_FUTURE_INDEX = ...

    def __init__(self):
        super().__init__()
        self.mean = self.std = self.min = self.max = None

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def set_min(self, min):
        self.min = min

    def set_max(self, max):
        self.max = max