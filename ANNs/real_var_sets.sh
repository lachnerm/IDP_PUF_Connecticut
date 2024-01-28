#!/bin/bash
#SBATCH --partition=gpu                           # Name of Partition
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END                             # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=michael.lachner@outlook.de      # Destination email address
#SBATCH --job-name real_circle2_first1k
#SBATCH --cpus-per-task=16
#SBATCH -o o_var_sets.out
module load sqlite/3.18.0
module load tcl/8.6.6.8606
module load gcc/5.4.0-alt
module load libffi/3.2.1
module load python/3.6.3
module load lzma
module load cuda
module load glib/2.40.0

python3 -m pip install matplotlib
python3 -m main_real_var_sets --f="$1"