#!/bin/bash

set -euo pipefail

source ~/venvs/venv-openmm/bin/activate
pip install .

#INPUT_FILE="./input_files/gl13k.csv"
#INPUT_FILE="./input_files/negative_control.csv"
#INPUT_FILE="./input_files/real_negative_control.csv"
#INPUT_FILE="./input_files/bahl_et_al_negative_controls.csv"
#INPUT_FILE="./input_files/bahl_B59s.csv"
#INPUT_FILE="./input_files/park_et_al_2024.csv"
INPUT_FILE="./input_files/bahl_et_al_positive_controls.csv"

echo "starting process_sequences.py"
python scripts/process_sequences.py \
        --input_file $INPUT_FILE \
        --wdir "outputs/us_test" \
	--n_windows 26 \
	--run_umbrella \
        --ff_toppar_path "ff_files/toppar" \
        --accelerator "gpu" \
        --override
