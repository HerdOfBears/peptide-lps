#!/bin/bash
#SBATCH --job-name=bayesopt
#SBATCH --account=def-rmansbac          # Replace with your Compute Canada allocation
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=h100:1

set -euo pipefail

EXPT_DIR=$SCRATCH/peptide-lps/bayesopt_expt1_smoke_test/

# Load required modules 
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6
module load vmd/1.9.4a57
module load gromacs/2026.1 openmm/8.4.0
module load rdkit/2025.09.4
module load wham/2.1.0

# Activate the virtual environment
source .venv/bin/activate

# Replace this with your Bayesian optimization command.
python scripts/bayesopt_driver.py --smoke_test --expt_dir $EXPT_DIR
echo "Bayesian optimization job completed."
