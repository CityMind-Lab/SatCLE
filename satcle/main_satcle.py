import argparse
import os
from datetime import datetime
import lightning.pytorch
import torch
from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from loss import SatCLELoss
from model import SatCLE
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')

class SatCLELightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        in_channels=13,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,        
    ) -> None:
        super().__init__()

        self.model = SatCLE(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
        )

        self.loss_fun = SatCLELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        images = batch[0]["image"]
        t_points = batch[0]["point"].float()
        filenames = batch[1]
        logits_per_image, logits_per_coord = self.model(images, t_points, filenames)
        return self.loss_fun(logits_per_image, logits_per_coord)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },
            ],
            lr=self.learning_rate,
        )
        
        return optimizer

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")
        parser.add_argument("--time", type=str, default=None)


def cli_main(default_config_filename="/configs/default.yaml",
             ):
    save_path = 'outputs/satcle'
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,         
        verbose=True,        
        mode='min',          
    )


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_last= True,
        dirpath=save_path,  
        filename='{epoch:02d}-{val_loss:.2f}', 
    )
    
    tensorboard_logger = {
        "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
        "init_args": {
            "name": "satclip_fuseneighbourlocation_causalmixup_disadj",
            "version": "satclip_fuseneighbourlocation_causalmixup_disadj",
            "save_dir": save_path,
        }
    }
    
    cli = MyLightningCLI(
        model_class=SatCLELightningModule,
        datamodule_class=S2GeoDataModule,
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 10,
            "callbacks": [early_stop_callback, checkpoint_callback],
            "precision": 16, 
            "logger": [tensorboard_logger],
            "strategy": "ddp_find_unused_parameters_true"  
        },
        parser_kwargs={
            "default_config_files": [default_config_filename]
        },
        seed_everything_default=0,
        run=False,
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"SatCLIP_S2_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    cli.trainer.callbacks[4].dirpath = os.path.join(cli.trainer.callbacks[4].dirpath, cli.config.time)
    new_save_dir = os.path.join(tensorboard_logger['init_args']['save_dir'], cli.config.time)

    new_tensorboard_logger = TensorBoardLogger(
        name=tensorboard_logger['init_args']['name'],
        version=tensorboard_logger['init_args']['version'],
        save_dir=new_save_dir,
    )

    cli.trainer.logger = new_tensorboard_logger
    
    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    
    config_fn = './satcle.yaml'
    if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
        torch.backends.cuda.matmul.allow_tf32 = True
        print('Superfastmode! 🚀')
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        
    cli_main(
        default_config_filename=config_fn,
    )