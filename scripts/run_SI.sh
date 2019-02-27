#!/bin/bash

#SBATCH --mem-per-cpu=1000M
#SBATCH --time=8:00:00
#SBATCH --job-name=run_model_SI
#SBATCH --error=model_error_SI.txt
#SBATCH --output=model_output_SI.txt
#SBATCH --ntasks=1

module load python2

source $HOME/venv/bin/activate

id=$SLURM_ARRAY_TASK_ID

python optimize_model.py > POP_SI_"$id"_results.txt