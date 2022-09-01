#!/bin/bash -l

#SBATCH --job-name=bert-squad-finetune
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=<account>

module load daint-gpu
module load sarus
module load OpenMPI/4.1.2   # installed locally by the user

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun sarus run --mount=type=bind,source=$SCRATCH,destination=$SCRATCH \
             nvcr.io/nvidia/pytorch:22.08-py3 \
             bash -c '
             cd $SCRATCH/UserLabDay/2022/containers/ngc-bert-squad
             . deepspeed-env/bin/activate;
             python bert_squad_deepspeed_train.py --deepspeed_config ds_config.json'
