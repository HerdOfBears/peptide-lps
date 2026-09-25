import shutil
import subprocess
import sys
from pathlib import Path

import os
import argparse
import logging

import pandas as pd
import numpy as np

from peplps.boltz_helpers import run_boltz_predict, write_yaml_input


def args_parser():
    pass

def main(params):
    
    df = pd.read_csv(params.input_file)
    pep_ids     = df["pep-id"]
    sequences   = df["sequence"]
    
    for pep_id, seq in zip(pep_ids, sequences):
        # logging.info(f"Processing peptide {pep_id} with sequence {seq}")
        print(f"Processing peptide {pep_id} with sequence {seq}")
        _odir = Path(params.wdir) / f"peptide-{pep_id}"
        if not _odir.exists():
            _odir.mkdir(parents=True)

        ##################################################
        # Run Boltz 2
        ##################################################
        yaml_path = _odir / f"pep-{pep_id}-input.yaml"
        write_yaml_input(seq, pep_id, yaml_path)

        out_dir = _odir / f"pep-{pep_id}-boltz2-prediction"
        _output_fmt = "pdb"
        run_boltz_predict(
            yaml_path=yaml_path,
            out_dir=out_dir,
            accelerator=params.accelerator,
            recycling_steps=params.recycling_steps,
            diffusion_samples=params.diffusion_samples,
            output_format=_output_fmt,
            override=params.override
        )

        ####
        # get the predicted structure file path
        pred_struct_path = (
            out_dir / f"boltz_results_pep-{pep_id}-input" /\
                "predictions" /f"pep-{pep_id}-input" /\
                f"pep-{pep_id}-input_model_0.{_output_fmt}"
        )

        ##################################################
        # run SMD simulations using the predicted structure
        ##################################################

        # 1. use subprocess to call bash_scripts/setup_peptide_lps_system.sh
        #   with arguments: pred_struct_path, pep_id, wdir
        setup_script = Path("bash_scripts/setup_peptide_lps_system.sh")
        subprocess.run([
            str(setup_script),
            str(pred_struct_path),
            str(pep_id),
            str(_odir),
        ], check=True)

        # 2. use subprocess to call scripts/run_smd.py
        #  with arguments: 
        #    --pepid pep_id
        #    --psf path_to_psf_file (which will be in wdir/peptide-pep_id/solvated_ionized.psf)
        #    --pdb path_to_pdb_file (which will be in wdir/peptide-pep_id/solvated_ionized.pdb)
        #    --wdir wdir
        #    --odir wdir/peptide-pep_id/
        #    --ff_toppar_path path_to_toppar (which will be in wdir/)
        #    --rseed 42
        _psf = Path(params.wdir) / f"peptide-{pep_id}" / "solvated_ionized.psf"
        _pdb = Path(params.wdir) / f"peptide-{pep_id}" / "solvated_ionized.pdb"
        run_smd_script = Path("scripts/run_smd.py")
        subprocess.run([
            sys.executable,  # Use the current Python interpreter
            str(run_smd_script),
            "--pepid", str(pep_id),
            "--psf", str(_psf),
            "--pdb", str(_pdb),
            "--wdir", str(params.wdir),
            "--odir", _odir,
            "--ff_toppar_path", str(params.ff_toppar_path),
            "--rseed", "42",
            "--init_fmax_cutoff", "3500",
            "--n_windows", str(params.n_windows),
            "--run_umbrella" if params.run_umbrella else ""
        ], check=True)

        print(f"Finished processing peptide {pep_id}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input CSV file containing peptide IDs and sequences")
    parser.add_argument("--wdir", required=True, help="Working directory path")
    parser.add_argument("--n_windows",required=True, help="Number of frames to save for US along the SMD pull")
    parser.add_argument("--run_umbrella", required=False, action='store_true',help="Whether or not to run US and WHAM")

    # simulation argument
    parser.add_argument("--ff_toppar_path", required=True, help="Path to toppar/ containing forcefield parameter files")
    
    # Boltz 2 arguments
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="cpu", help="Device to run Boltz-2 on (default: cpu).")
    parser.add_argument("--recycling_steps", type=int, default=3, help="Number of recycling steps for Boltz-2 (default: 3).")
    parser.add_argument("--diffusion_samples", type=int, default=1, help="Number of diffusion samples to generate with Boltz-2 (default: 1).")
    parser.add_argument("--override", action='store_true', help="Whether to override existing output directories/files for each peptide.")
    
    params = parser.parse_args()

    main(params)