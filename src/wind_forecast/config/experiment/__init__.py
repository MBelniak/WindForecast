from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf.omegaconf import MISSING

# Experiment settings validation schema & default values
from wind_forecast.preprocess.synop.consts import SYNOP_TRAIN_FEATURES
from wind_forecast.util.utils import NormalizationType


@dataclass
class ExperimentSettings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1

    # Path to resume from. Two formats are supported:
    # - local checkpoints: path to checkpoint relative from run (results) directory
    # - wandb artifacts: wandb://ARTIFACT_PATH/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    resume_checkpoint: Optional[str] = None

    # Enable checkpoint saving
    save_checkpoints: bool = True

    # Enable initial validation before training
    validate_before_training: bool = False

    # ----------------------------------------------------------------------------------------------
    # Data loading settings
    # ----------------------------------------------------------------------------------------------
    # Training batch size
    batch_size: int = 128

    # Enable dataset shuffling
    shuffle: bool = True

    # Number of dataloader workers
    num_workers: int = 8

    # Model to use
    model: Any = MISSING

    system: Any = MISSING

    # ----------------------------------------------------------------------------------------------
    # Dataset specific settings
    # ----------------------------------------------------------------------------------------------
    # Lightning DataModule to use
    datamodule: Any = MISSING

    normalization_type: NormalizationType = NormalizationType.STANDARD

    val_split: float = .2

    gfs_dataset_dir: str = os.path.join("D:\\WindForecast", "output_np2")

    synop_file: str = "KOZIENICE_488_data.csv"

    synop_from_year: int = 2019

    target_parameter: str = "temperature"

    prediction_offset: int = 3

    data_dim_x: int = 53

    data_dim_y: int = 33

    train_parameters_config_file: str = "CNNConfig.json"

    cnn_input_size: Any = (33, 53)

    cnn_ff_input_dim: List = field(default_factory=lambda: [1600, 256])

    cnn_filters: List = field(default_factory=lambda: [32, 64])

    epochs: int = 100

    synop_train_features: List = field(default_factory=lambda: SYNOP_TRAIN_FEATURES)

    sequence_length: int = 24

    future_sequence_length: int = 24

    target_coords: List = field(default_factory=lambda: [52.1831174, 20.9875259])

    tcn_channels: List = field(default_factory=lambda: [64])

    tcn_kernel_size: int = 3

    tcn_input_features: int = 1600

    tcn_cnn_ff_input_dim: int = 1536

    subregion_nlat: float = 53

    subregion_slat: float = 51

    subregion_elon: float = 22

    subregion_wlon: float = 20

    dropout: float = 0.1

    teacher_forcing_epoch_num: int = 10

    gradual_teacher_forcing: bool = True

    time2vec_embedding_size: int = 5

    transformer_ff_dim: int = 1024

    transformer_attention_layers: int = 6

    transformer_attention_heads: int = 1

    cmax_sample_size: Any = (900, 900)