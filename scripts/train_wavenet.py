from audiosep.data import WaveDatamodule
from audiosep.models import WaveUNet
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=50, help='Number of training epochs')
args = parser.parse_args()

# Run in another terminal: tensorboard --logdir lightning_logs --port 6006
# logger = TensorBoardLogger("lightning_logs", name="tb_run")
logger = WandbLogger(project="audiosep", name="wave_unet_run", log_model="all")

device = "mps" if torch.backends.mps.is_available() else "cpu"
# Adjust max_len based on device capabilities
max_len=32000 if device == "mps" else 80000
    
dm = WaveDatamodule(
    train_data_dir="data/train",
    test_data_dir="data/test",
    batch_size=args.batch_size,
    num_workers=0,
    seed=42,
    max_len=max_len,
)

# W&B checkpoint dir
run = logger.experiment  # should be a wandb Run
run_dir = getattr(run, 'dir', None) or getattr(run, 'run_dir', None)
ckpt_dir = os.path.join(run_dir)
os.makedirs(ckpt_dir, exist_ok=True)
print(f"Saving checkpoints to: {ckpt_dir}")

# Model
model = WaveUNet(in_channels=1, out_channels=2, depth=5, base_filters=24, max_filters=512)

# Trainer
callback = ModelCheckpoint(every_n_epochs=5, dirpath=ckpt_dir, filename="{epoch:02d}")
# Logging every 10 steps to ensure training logs appear even when epoch has fewer batches
trainer = Trainer(max_epochs=args.max_epochs, accelerator="auto", logger=logger, callbacks=[callback], log_every_n_steps=10)
trainer.fit(model, dm)