import os
from typing import cast

import hydra
import optuna
import pytorch_lightning as pl
import setproctitle
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config, register_configs, get_tags
from wind_forecast.util.callbacks import CustomCheckpointer, get_resume_checkpoint
from wind_forecast.util.logging import log
from wind_forecast.util.plots import plot_results
from wind_forecast.util.rundir import setup_rundir

from wind_forecast.util.common_util import wandb_logger
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_tune(cfg: Config):

    def objective(trial: optuna.trial.Trial):
        config = {}
        for param in cfg.tune.params.keys():
            config[param] = trial.suggest_categorical(param, list(cfg.tune.params[param]))
        config['dropout'] = trial.suggest_uniform('dropout', 0.1, 0.6)

        for param in config.keys():
            cfg.experiment.__setattr__(param, config[param])

        if 'lambda_lr' in cfg.optim:
            cfg.optim.__setattr__('starting_lr', trial.suggest_loguniform('starting_lr', 0.000001, 0.001))
            cfg.optim.__setattr__('final_lr', trial.suggest_loguniform('final_lr', 0.000001, 0.001))
            cfg.optim.__setattr__('warmup_epochs', trial.suggest_int('warmup_epochs', 0, cfg.experiment.epochs))
            cfg.optim.__setattr__('decay_epochs', trial.suggest_int('decay_epochs', 0, cfg.experiment.epochs))

        cfg.optim.__setattr__('base_lr', trial.suggest_loguniform('base_lr', 0.000001, 0.001))

        # Create main system (system = models + training regime)
        system: LightningModule = instantiate(cfg.experiment.system, cfg)
        log.info(f'[bold yellow]\\[init] System architecture:')
        log.info(system)
        dm = instantiate(cfg.experiment.datamodule, cfg)

        trainer: pl.Trainer = instantiate(
            cfg.lightning,
            logger=True,
            max_epochs=cfg.experiment.epochs,
            gpus=cfg.lightning.gpus,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="ptl/val_loss")],
            checkpoint_callback=False,
            num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
            check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch
        )
        trainer.fit(system, dm)

        return trainer.logged_metrics["ptl/val_loss"]

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=cfg.tune.trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def run_training(cfg):
    RUN_NAME = os.getenv('RUN_NAME')
    log.info(f'[bold yellow]\\[init] Run name --> {RUN_NAME}')

    run: Run = wandb_logger.experiment  # type: ignore

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    run.tags = tags
    run.notes = str(cfg.notes)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    log.info(f'[bold yellow][{RUN_NAME} / {run.id}]: [bold white]{",".join(tags)}')

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    log.info(f'[bold white]Overriding cfg.lightning settings with derived values:')
    log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}\n')

    # Create main system (system = models + training regime)
    system: LightningModule = instantiate(cfg.experiment.system, cfg)
    log.info(f'[bold yellow]\\[init] System architecture:')
    log.info(system)
    # Prepare data using datamodules
    datamodule: LightningDataModule = instantiate(cfg.experiment.datamodule, cfg)

    resume_path = get_resume_checkpoint(cfg, wandb_logger)
    if resume_path is not None:
        log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

    checkpointer = CustomCheckpointer(
        dirpath='checkpoints',
        filename='{epoch}'
    )

    trainer: pl.Trainer = instantiate(
        cfg.lightning,
        logger=wandb_logger,
        max_epochs=cfg.experiment.epochs,
        callbacks=[checkpointer],
        resume_from_checkpoint=resume_path,
        checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
        num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch
    )

    if not cfg.experiment.skip_training:
        trainer.fit(system, datamodule=datamodule)

    trainer.test(system, datamodule=datamodule)

    metrics = {
        'train_dataset_length': len(datamodule.dataset_train),
        'test_dataset_length': len(datamodule.dataset_test)
    }

    mean = datamodule.dataset_test.mean
    std = datamodule.dataset_test.std
    if cfg.experiment.use_gfs_data:
        gfs_mean = datamodule.dataset_test.gfs_mean
        gfs_std = datamodule.dataset_test.gfs_std
    else:
        gfs_mean = None
        gfs_std = None

    if mean is not None:
        if type(mean) == list:
            for index, m in enumerate(mean):
                metrics[f"target_mean_{str(index)}"] = m
        else:
            metrics['target_mean'] = mean
    if std is not None:
        if type(std) == list:
            for index, s in enumerate(std):
                metrics[f"target_std_{str(index)}"] = s
        else:
            metrics['target_std'] = std

    wandb_logger.log_metrics(metrics, step=system.current_epoch)

    if cfg.experiment.view_test_result:
        plot_results(system, cfg, mean, std, gfs_mean, gfs_std)

    if trainer.interrupted:  # type: ignore
        log.info(f'[bold red]>>> Training interrupted.')
        run.finish(exit_code=255)


@hydra.main(config_path='config', config_name='default')
def main(cfg: Config):
    RUN_MODE = os.getenv('RUN_MODE', '').lower()
    if RUN_MODE == 'debug':
        cfg.debug_mode = True

    log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')

    pl.seed_everything(cfg.experiment.seed)

    cfg.experiment.train_parameters_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               'config', 'train_parameters',
                                                               cfg.experiment.train_parameters_config_file)

    if RUN_MODE in ['tune', 'tune_debug']:
        cfg.tune_mode = True
        if RUN_MODE == 'tune_debug':
            cfg.debug_mode = True
        run_tune(cfg)
    else:
        run_training(cfg)


if __name__ == '__main__':
    setup_rundir()

    wandb.init(project=os.getenv('WANDB_PROJECT'),
               entity=os.getenv('WANDB_ENTITY'),
               name=os.getenv('RUN_NAME'))

    # Init logger from source dir (code base) before switching to run dir (results)
    wandb_logger.experiment  # type: ignore

    # Instantiate default Hydra config with environment variables & switch working dir
    register_configs()
    main()
