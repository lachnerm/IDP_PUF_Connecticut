#!/bin/bash
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --mail-type=END                             # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=michael.lachner@outlook.de      # Destination email address
#SBATCH --job-name store_preds
#SBATCH --cpus-per-task=8
#SBATCH -o o_preds.out

python3 -m main_real_store_preds --f="$1"