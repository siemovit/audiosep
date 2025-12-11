from audiosep.data import WaveDatamodule
from audiosep.models import WaveUNet
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os
import torch
import argparse
import wandb
from wandb import Api

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--exp_name', type=str, default="", help='Additional experiment name info')
parser.add_argument('--depth', type=int, default=5, help='Depth of the WaveUNet model')
parser.add_argument('--base_filters', type=int, default=24, help='Number of base filters in WaveUNet model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')

args = parser.parse_args()

# Experiment name
config = f"L{args.depth}_f{args.base_filters}_b{args.batch_size}_ep{args.max_epochs}"
exp_suffix = "_" + args.exp_name if args.exp_name else ""
exp_name = "Wave_U-Net_" + config + exp_suffix

# Logger and download data
logger = WandbLogger(project="audiosep", name=exp_name, log_model="all")
if not os.path.exists("data"):
    logger.download_artifact(
        "simon-yannis/audiosep/data:latest", save_dir="data", artifact_type="raw_data"
    )

# Adjust max_len based on device capabilities
device = "mps" if torch.backends.mps.is_available() else "cpu"
max_len=32000 if device == "mps" else 80000

# DataModule
dm = WaveDatamodule(
    train_data_dir="data/train",
    test_data_dir="data/test",
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    seed=42,
    max_len=max_len,
)

# Model
model = WaveUNet(in_channels=1, out_channels=2, depth=args.depth, base_filters=args.base_filters)

# Logger checkpoint dir
run = logger.experiment  # should be a wandb Run
run_dir = getattr(run, 'dir', None) or getattr(run, 'run_dir', None)
ckpt_dir = os.path.join(run_dir)
os.makedirs(ckpt_dir, exist_ok=True)

# Trainer
callback = ModelCheckpoint(every_n_epochs=5, dirpath=ckpt_dir, filename="{epoch:02d}")
trainer = Trainer(max_epochs=args.max_epochs, accelerator="auto", logger=logger, callbacks=[callback], log_every_n_steps=10)
trainer.fit(model, dm)