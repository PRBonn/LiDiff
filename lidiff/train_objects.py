import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME

import lidiff.datasets.datasets_objects as datasets_objects
import lidiff.models.models_objects as models_objects

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, weights, checkpoint, test):
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')

    #Load data and model
    if weights is None:
        model = models_objects.DiffusionPoints(cfg)
    else:
        if test:
            ckpt_cfg = yaml.safe_load(open(config))
            cfg = ckpt_cfg

        model = models_objects.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
        print(model.hparams)

    data = datasets_objects.dataloaders[cfg['data']['dataloader']](cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                dirpath='checkpoints/',
                                filename=cfg['experiment']['id']+'_{epoch:02d}',
                                save_last=True
                            )

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    #Setup trainer
    if torch.cuda.device_count() > 1:
        cfg['train']['n_gpus'] = torch.cuda.device_count()
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(
                        devices=cfg['train']['n_gpus'],
                        logger=tb_logger,
                        log_every_n_steps=100,
                        resume_from_checkpoint=checkpoint,
                        max_epochs= cfg['train']['max_epoch'],
                        callbacks=[lr_monitor, checkpoint_saver],
                        check_val_every_n_epoch=5,
                        accelerator='gpu',
                        strategy="ddp",
                        )
    else:
        trainer = Trainer(
                        accelerator='gpu',
                        devices=cfg['train']['n_gpus'],
                        logger=tb_logger,
                        log_every_n_steps=100,
                        resume_from_checkpoint=checkpoint,
                        max_epochs= cfg['train']['max_epoch'],
                        callbacks=[lr_monitor, checkpoint_saver],
                        check_val_every_n_epoch=5,
                        )


    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
