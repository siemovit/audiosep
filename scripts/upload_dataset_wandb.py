import wandb

wandb.init(project="audiosep", job_type="upload")
# path to my remote data directory in Google Cloud Storage
# create a regular artifact
dataset_at = wandb.Artifact("data", type="raw_data")
# creates a checksum for each file and adds a reference to the bucket
# instead of uploading all of the contents
dataset_at.add_dir("data")
wandb.log_artifact(dataset_at)
