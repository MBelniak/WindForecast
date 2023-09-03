REM python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
REM experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.train_parameters_config_file="CommonOperGFSConfig.json" experiment.epochs=3 experiment.target_coords=[54.5884822,16.8542177] experiment.synop_file="USTKA_354160115_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.train_parameters_config_file="CommonOperGFSConfig.json" experiment.epochs=3 experiment.target_coords=[54.829480,18.335405] experiment.synop_file="ROZEWIE_254180010_data.csv"

REM python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
REM experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.train_parameters_config_file="CommonOperGFSConfig.json" experiment.epochs=3 experiment.target_coords=[54.753647,17.534823] experiment.synop_file="LEBA_354170120_data.csv"
