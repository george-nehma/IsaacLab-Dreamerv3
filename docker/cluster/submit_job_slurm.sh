#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash 

#SBATCH --job-name=isaaclab
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=isaaclab_6dof.log
#SBATCH --error=job_error_6dof.log

source /etc/profile.d/modules.sh

# Load apptainer
module load apptainer

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
