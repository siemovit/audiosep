import os
from pathlib import Path
from typing import cast

from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
import audiosep.data  # pylint: disable=W0611 # noqa: F401 used by the CLI to find datamodules
import audiosep.models  # pylint: disable=W0611 # noqa: F401 used by the CLI to find models
import wandb


class CustomSaveConfigCallback(SaveConfigCallback):
    """# from https://github.com/Lightning-AI/pytorch-lightning/issues/19728"""

    # Saves full training configuration
    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                cast(WandbLogger, logger).watch(pl_module, log="all")
                config = self.config.as_dict()
                logger.log_hyperparams({"config": config})
        return super().save_config(trainer, pl_module, stage)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--checkpoint_reference")

    def before_instantiate_classes(self):
        api = wandb.Api()
        cmd: str = cast(str, self.subcommand)
        if cmd == "test" or cmd == "predict":
            model_artifact = api.artifact(
                self.config[cmd]["checkpoint_reference"], type="model"
            )
            model_artifact_dir = model_artifact.download()
            self.config[cmd]["ckpt_path"] = os.path.join(
                Path(model_artifact_dir).as_posix(), "model.ckpt"
            )
        dataset_artifact = api.artifact(
            "simon-yannis/audiosep/data:latest", type="raw_data"
        )
        if not os.path.exists("./data"):
            dataset_artifact.download(root="./data")


def cli_main():
    MyLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    wandb.finish()  # really important


if __name__ == "__main__":
    cli_main()
