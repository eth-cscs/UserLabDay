#!/bin/bash -l
#SBATCH --job-name=by_test_job
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=cscsci
#SBATCH --constraint=gpu
#SBATCH --output=test_job.out
#SBATCH --error=test_job.err

date
module list
