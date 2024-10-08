#!/bin/bash

#SBATCH -J template-unattended
#SBATCH -t 0:30:00
#SBATCH --partition h100
#SBATCH --gpus 1

# Only for Kuma temporarily

# If not done already in your bashrc (depends on the cluster so better write that logic there.)
# export SCRATCH=/scratch/moalla

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$SCRATCH/thought-transfer/run
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1

srun \
  --container-image=$CONTAINER_IMAGES/claire+moalla+thought-transfer+amd64-cuda-root-latest.sqsh \
  --container-mounts=$SCRATCH:$SCRATCH \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  -G 1 \
  /opt/template-entrypoints/pre-entrypoint.sh \
  python -m thought_transfer.template_experiment some_arg=some_value wandb.mode=offline

# additional options
# --container-env to override environment variables defined in the container
