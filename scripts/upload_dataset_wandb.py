import wandb
run = wandb.init(project="audiosep", job_type="upload")
# path to my local dataset
dir = "file:///Users/yannis/audiosep/data"
# create a regular artifact
dataset_at = wandb.Artifact('data',type="raw_data")
# creates a checksum for each file and adds a reference to the bucket
# instead of uploading all of the contents
dataset_at.add_reference(dir)
run.log_artifact(dataset_at)
