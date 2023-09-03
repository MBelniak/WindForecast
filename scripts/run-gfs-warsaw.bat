REM NBEATSX

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.epochs=20

