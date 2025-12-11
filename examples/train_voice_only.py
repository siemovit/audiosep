import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from audiosep.data import OriginalVoiceNoiseDatamodule
from audiosep.models import SpectroUNetOriginal


wandb_logger = WandbLogger(
    project="audiosep", name="spectro_UNetOriginal_run", log_model="all"
)
if not os.path.exists("./data"):
    wandb_logger.download_artifact(
        "simon-yannis/audiosep/data:latest", save_dir="./data", artifact_type="raw_data"
    )

dm = OriginalVoiceNoiseDatamodule(
    train_data_dir="data/train",
    test_data_dir="data/test",
    batch_size=128,
    num_workers=0,
    seed=42,
)


# Model
model = SpectroUNetOriginal()
wandb_logger.watch(model, log="all")
# Trainer
callback = ModelCheckpoint(every_n_epochs=5, filename="{epoch:02d}")
# Logging every 10 steps to ensure training logs appear even when epoch has fewer batches
trainer = Trainer(
    max_epochs=100,
    accelerator="auto",
    logger=wandb_logger,
    callbacks=[callback],
    log_every_n_steps=10,
)
trainer.fit(model, dm)
