#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<'EOT' > job.sh
#!/bin/bash 

#SBATCH --job-name=isaaclab
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=isaaclab_6dof.log
#SBATCH --error=job_error_6dof.log

source /etc/profile.d/modules.sh

echo "=============================="
echo " Job information"
echo " Node:        $SLURMD_NODENAME"
echo " Job ID:      $SLURM_JOB_ID"
echo " GPUs:        $SLURM_JOB_GPUS"
echo " Hostname:    $(hostname)"
echo "=============================="

# Load apptainer
module unload apptainer || true
module load apptainer #/1.3.4-gcc-14.2.0-g7o5w4g

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"

EOT

sbatch job.sh  "$1" "$2" "${@:3}"
rm job.sh
