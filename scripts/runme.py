import os
from pathlib import Path
from typing import cast

from lightning.pytorch.cli import LightningCLI

import audiosep.data  # pylint: disable=W0611 # noqa: F401 used by the CLI to find datamodules
import audiosep.models  # pylint: disable=W0611 # noqa: F401 used by the CLI to find models
import wandb

# simple demo classes for your convenience


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
    MyLightningCLI(save_config_callback=None)
    wandb.finish()  # really important


if __name__ == "__main__":
    cli_main()
