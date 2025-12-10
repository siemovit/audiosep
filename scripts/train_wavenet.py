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

wandb.init(project="audiosep")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--exp_name', type=str, default='', help='Additional experiment name info')

args = parser.parse_args()

if not os.path.exists("data"):
    api = Api()  # se base sur WANDB_API_KEY ou ton login wandb actif
    artifact = api.artifact("simon-yannis/audiosep/data:latest")  # <entity>/<project>/<name>:<version>
    dst = artifact.download(root="data")  # télécharge dans ./data
    print("Artifact downloaded to:", dst)
    
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Adjust max_len based on device capabilities
max_len=32000 if device == "mps" else 80000
    
dm = WaveDatamodule(
    train_data_dir="data/train_small",
    test_data_dir="data/test_small",
    batch_size=args.batch_size,
    num_workers=0,
    seed=42,
    max_len=max_len,
)

# Model
model = WaveUNet(in_channels=1, out_channels=2, depth=5, base_filters=24)

def get_exp_name(model, args):
    run_number = wandb.run.name.split("-")[-1]  
    return run_number + "_" + model.__repr__() + args.exp_name

exp_name = get_exp_name(model, args)
print(f"Experiment name: {exp_name}")

# initialize W&B run explicitly with the desired name
wandb_run = wandb.init(project="audiosep", name=exp_name, reinit=True)

logger = WandbLogger(project="audiosep", name=exp_name, log_model="all")
run = logger.experiment  # should be a wandb Run
run_dir = getattr(run, 'dir', None) or getattr(run, 'run_dir', None)
ckpt_dir = os.path.join(run_dir)
os.makedirs(ckpt_dir, exist_ok=True)

# Trainer
callback = ModelCheckpoint(every_n_epochs=5, dirpath=ckpt_dir, filename="{epoch:02d}")
# Logging every 10 steps to ensure training logs appear even when epoch has fewer batches
trainer = Trainer(max_epochs=args.max_epochs, accelerator="auto", logger=logger, callbacks=[callback], log_every_n_steps=10)
# trainer.fit(model, dm)