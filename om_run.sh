#!/bin/bash
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --array=0-4
#SBATCH --job-name=dqn
#SBATCH --mem=32GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 24:00:00
#SBATCH -D /om2/user/amineh/rat_exp/dqn_gym/log/
#SBATCH --partition=cbmm

cd /om2/user/amineh/rat_agent/

hostname

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python3 main.py --agent dqn_gym --host_filesystem om2 --run train --experiment_index ${SLURM_ARRAY_TASK_ID}

