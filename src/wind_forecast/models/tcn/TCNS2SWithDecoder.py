from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.tcn.TCNModel import TemporalBlock
from wind_forecast.models.tcn.TCNS2SModel import TemporalConvNetS2S


class TemporalConvNetS2SWithDecoder(TemporalConvNetS2S):
    def __init__(self, config: Config):
        super().__init__(config)

        self.kernel_size = config.experiment.tcn_kernel_size

        tcn_layers = []
        in_channels = config.experiment.tcn_channels[-1]
        for i in range(self.num_levels):
            dilation_size = 2 ** (self.num_levels - i)
            out_channels = self.tcn_channels[-(i+2)] if i < self.num_levels - 1 else self.embed_dim
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=self.embed_dim + (1 if self.use_gfs else 0), out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, False)
        if self.use_gfs:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        x = self.encoder(input_elements.permute(0, 2, 1))
        y = self.decoder(x).permute(0, 2, 1)[:, -self.future_sequence_length:, :]

        if self.use_gfs:
            return self.classification_head(torch.cat([y, gfs_targets], -1)).squeeze(-1)
        return self.classification_head(y).squeeze(-1)
