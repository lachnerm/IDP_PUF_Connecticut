#!/bin/bash
#SBATCH --partition=gpu                           # Name of Partition
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END                             # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=michael.lachner@outlook.de      # Destination email address
#SBATCH --job-name realSV7
#SBATCH --ntasks=16
#SBATCH -o or.out
module load sqlite/3.18.0
module load tcl/8.6.6.8606
module load gcc/5.4.0-alt
module load libffi/3.2.1
module load python/3.6.3

python3 -m print_real
