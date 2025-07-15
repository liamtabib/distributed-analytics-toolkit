#!/bin/bash -l
#SBATCH -A uppmax2023-2-36 # project name 
#SBATCH -M snowy # name of system 
#SBATCH -p node # request a full node 
#SBATCH -N 1 # request 1 node #SBATCH -t 1:00:00 # job takes at most 1 hour 
#SBATCH --gres=gpu:1 --gpus-per-node=1 # use the GPU nodes
#SBATCH -J stream_cuda # name of the job 
#SBATCH -D ./ # stay in current working directory 

# Build the program if not already built
if [ ! -f "./memory-bandwidth-benchmark" ]; then
    make
fi

# Create output directory
mkdir -p output

# Run the program and save output
nvidia-smi
./memory-bandwidth-benchmark > output/results.txt
