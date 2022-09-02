#!/bin/bash
#SBATCH --job-name="tensorflow_hvd_cnn_job"
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0:10:0
#SBATCH --account=csstaff
#SBATCH --constraint=gpu

module load daint-gpu
module load Horovod
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0

srun python tf2_hvd_synthetic_benchmark.py --batch-size=128 \
    --model=ResNet50 --num-iters=10
