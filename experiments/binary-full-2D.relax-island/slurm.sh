#!/bin/bash
#SBATCH --job-name bampl1D_AB

#SBATCH -q express
#SBATCH -p ezfhp
#SBATCH --nodelist=zfh15

##SBATCH -q primary
##SBATCH -q secondary
##SBATCH -q debug

#SBATCH -N 1             # number of nodes used
#SBATCH -n 4             # number of cpus
#SBATCH --tasks-per-node=4 --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -t 2-0:0:0      # -t [min] OR -t [days-hh:mm:ss]

##SBATCH --constraint=intel

#SBATCH -o output_%A_%a.dat
#SBATCH -e errors_%A_%a.dat
##SBATCH --mail-type=ALL
#SBATCH --mail-user=eric.stimpson@wayne.edu
# #SBATCH --array=1-136
# #SBATCH --array=35-136

# for bash: use export
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

hostname
date

# Run the job
cd /wsu/home/bf/bf55/bf5524/PFC/experiments/binary-full-2D.relax-island/rawout/
pwd

# Run the code
run='/wsu/home/bf/bf55/bf5524/PFC/experiments/binary-full-2D.relax-island/source/exp.out'
echo "run $run"
time $run
