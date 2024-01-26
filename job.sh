!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=db8682@princeton.edu

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

# Activate conda environment
conda activate langevin_rotation

# Script
# (your script here!)
