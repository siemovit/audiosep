from audiosep.data import WaveDatamodule
from audiosep.models import WaveUNet
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os

# Run in another terminal: tensorboard --logdir lightning_logs --port 6006
# logger = TensorBoardLogger("lightning_logs", name="tb_run")
logger = WandbLogger(project="audiosep", name="wave_unet_run", log_model="all")

dm = WaveDatamodule(
    train_data_dir="data/train",
    test_data_dir="data/test",
    batch_size=8,
    num_workers=0,
    seed=42,
)

# W&B checkpoint dir
run = logger.experiment  # should be a wandb Run
run_dir = getattr(run, 'dir', None) or getattr(run, 'run_dir', None)
ckpt_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"Saving checkpoints to: {ckpt_dir}")

# Model
# Increased capacity: depth=5, base_filters=24 (standard Wave-U-Net often uses 24)
model = WaveUNet(in_channels=1, out_channels=2, depth=5, base_filters=24, max_filters=512)

# Trainer
callback = ModelCheckpoint(every_n_epochs=5, dirpath=ckpt_dir, filename="{epoch:02d}")
# Logging every 10 steps to ensure training logs appear even when epoch has fewer batches
trainer = Trainer(max_epochs=100, accelerator="auto", logger=logger, callbacks=[callback], log_every_n_steps=10)
trainer.fit(model, dm)