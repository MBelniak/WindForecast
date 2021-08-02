import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.utils import weight_norm
from wind_forecast.config.register import Config


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, stride=1, dropout=0.5):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, 0), dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, 0), dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = len(config.experiment.synop_train_features) if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        layers += [
            nn.Flatten(),
            nn.Linear(in_features=num_channels[-1] * config.experiment.sequence_length, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
        return self.network(x).squeeze()
