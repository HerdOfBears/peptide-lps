import shutil
import subprocess
import sys
from pathlib import Path

import os
import argparse
import logging
import pickle as pkl

import pandas as pd
import numpy as np
import torch

from peplps.boltz_helpers import run_boltz_predict, write_yaml_input
from peplps.evo_bo import EVO_BO_2026, _NResEncoder

def high_fidelity_score(seq, pep_id, params=None):
    """
    Implements the umbrella sampling set up and pipeline
    """
    print(f"Processing peptide {pep_id} with sequence {seq}")
    n_arginines = seq.count("R")
    return n_arginines

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
    pass

def main(params):

    # get initial data, then initialize optimizer
    if os.path.exists(params['initial_data_fpath']):
        _data = pd.read_csv(params['initial_data_fpath'])
        train_x = _data['sequence']
        train_x = train_x.to_numpy().reshape(-1,1)

        if params["FIDELITIES"] == [1.0]:
            train_x = np.concat(
                (train_x, np.ones_like(train_x)), 
                axis=1
            )
        else:
            train_fids = _data['fidelity'].to_numpy().reshape(-1,1)
            train_x = np.concat( (train_x, train_fids), axis=1)

        train_obj = _data['objective'].to_numpy().reshape(-1,1)
    else:
        train_x   = None
        train_obj = None

    print(f"[info] {train_x.shape=}, {train_obj.shape=}")
    bayesOpt = EVO_BO_2026(train_x, train_obj, params)

    cumulative_cost = 0.0
    id_counter = 0
    results = {
        "iteration":[],
        "designs":[],
        "scores":[],
        "best_design":[],
        "best_design_score":[]
    }
    N_ITER = params.get('N_ITER',10)
    print(f"Running BayesOpt loop {N_ITER=} times")
    for _iter in range(N_ITER):
        print(f"{_iter+1=}/{N_ITER}")
        new_x, cost = bayesOpt.suggest()
        new_obj = torch.empty(new_x.shape[0], 1, dtype=torch.double)
        for i, row in enumerate(new_x):
            results['iteration'].append(_iter+1)

            design = row[0] # tensor of design vars (seq, fidelity)
            results["designs"].append(design)

            _fidelity_val = row[bayesOpt.fidelity_col]
            if not isinstance(_fidelity_val, float):
                _fidelity_val = _fidelity_val.item()

            if _fidelity_val >= 0.5:
                _score = high_fidelity_score(design, f"pep-{id_counter}")
                new_obj[i] = _score

                results['scores'].append(_score)
            else:
                raise NotImplementedError("low fidelity scores have not been implemented yet")
                # new_obj[i] = low_fidelity_score(design)
            id_counter += 1
        bayesOpt.register_observations(new_x, new_obj)

        _current_best = bayesOpt.get_recommendation().flatten()[0]
        _current_best_score = bayesOpt._incumbent()

        print(f"{_current_best}, {_current_best_score}")
        results["best_design"]       += [_current_best      ]*len(new_x)
        results["best_design_score"] += [_current_best_score]*len(new_x)

        cumulative_cost += cost

        os.makedirs(params['expt_dir'], exist_ok=True)
        with open(
            os.path.join(params["expt_dir"], "results.pkl"),
            "wb"
        ) as fo:
            pkl.dump(results, fo)


    best = bayesOpt.get_recommendation()

if __name__=="__main__":

    params = {}
    params["expt_dir"] = "outputs/us_test"
    params["N_ITER"]=30
    params["FIDELITIES"] = [1.0]
    params['MIN_MUTATIONS'] = 1 # how few mutations are allowed when sampling seqs 
    params['MAX_MUTATIONS'] = 3 # what is the max number of mutations allowed when sampling seqs

    params['ENCODER'] = _NResEncoder()
    # number of sequences to suggest at each BayesOpt iteration
    params['BATCH_SIZE'] = 1 

    params['initial_data_fpath'] = "./input_files/bo_initial_data.csv"
    main(params)