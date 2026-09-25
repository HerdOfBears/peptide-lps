
import os
import time 

from openmm import unit, Vec3
import openmm as mm
import openmm.app as app
from openmm import MonteCarloBarostat

import parmed as pmd
import logging

import sys
import math
import numpy as np
import argparse

import logging

from structSim import calculate_state_PE_and_maxForce
from structSim.equilibration import staged_soft_energy_minimization
from structSim.openmm_helpers import add_backbone_posres

from peplps.helpers import (
    get_indices, 
    set_cubic_box_from_positions,
    build_system,
    build_simulation,
    convert_for_wham
)
from peplps.steered_md import (
    run_smd,
    run_window, 
    plot_histograms, 
    run_wham, 
    plot_pmf,
    analyze_pmf,
)

if __name__=="__main__":

    # set up parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--pepid", required=True, 
                        help="ID for the peptide")
    parser.add_argument("--psf", required=True,
                        help="filepath to psf file")
    parser.add_argument("--pdb", required=True, 
                        help="filepath to associated pdb file")
    parser.add_argument("--wdir", required=True,
                        help="Working directory path")
    parser.add_argument("--odir", required=True,
                        help="output file directory path")
    parser.add_argument("--ff_toppar_path", required=True, 
                      help="Path to toppar/ containing forcefield parameter files")
    parser.add_argument("--rseed", default=42, required=False,
                        help="Random number seed")
    parser.add_argument("--init_fmax_cutoff", default=3_000, type=float, required=False,
                        help="Maximum force cutoff for successful energy minimization (default: 3000 kJ/mol/nm)")
    parser.add_argument("--n_windows", default=None, type=int, help="Number of windows to save during SMD and analyze in Umbrella Sampling")
    parser.add_argument("--run_umbrella", default=False, action="store_true",help="Whether or not to do umbrella sampling, or just SMD")
    args = parser.parse_args()

    params = vars(args)
    params['nonbondedCutoff'] = 1.2 # nm
    if isinstance(params["rseed"], str):
        try:
            params["rseed"] = int(params["rseed"])
        except ValueError:
            raise ValueError(f"Invalid rseed value: {params['rseed']}. Must be an integer.")

    platform = mm.Platform.getPlatformByName("OpenCL") if mm.Platform.getNumPlatforms() else None

    if not os.path.exists(params["odir"]):
        os.makedirs(params["odir"])

    logging.basicConfig(
        filename=os.path.join(params["odir"], f"pep-{params['pepid']}-smd.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    output_state_fpath = os.path.join(
        params["odir"], 
        f"pep-{params['pepid']}-smd-state.csv"
    )
    output_traj_fpath = os.path.join(
        params["odir"], 
        f"pep-{params['pepid']}-smd.dcd"
    )

    n_windows = params['n_windows']
    RUN_UMBRELLA = params['run_umbrella']
    #####################################
    # load forcefield parameters
    #####################################
    # param_files = []
    # for _f in os.listdir(params["ff_toppar_path"]):
    #     if _f =="toppar.str":
    #         continue
    #     if _f.endswith(".str") or _f.endswith(".prm") or _f.endswith(".rtf"):
    #         param_files.append( os.path.join(params["ff_toppar_path"], _f) ) 

    # ff_params = app.CharmmParameterSet(*param_files)

    #####################################
    # load the solvated systems
    solvated_psf = app.CharmmPsfFile(params["psf"])
    struct_crds  = pmd.load_file(    params["pdb"])
    # struct crds positions are in Angstrom
    #####################################
    
    
    # grab indices for the lipid A and peptide atoms
    # compute their COM-COM distance for r0; 
    # this will be the CV we pull on in SMD
    lipidA_indices, peptide_indices = get_indices(solvated_psf.topology)
    
    com_pep = pmd.geometry.center_of_mass(
        struct_crds[peptide_indices].coordinates/10.0, # convert from Angstrom to nm
        np.array([struct_crds.atoms[i].mass for i in peptide_indices])
    ) 
    
    com_lipidA = pmd.geometry.center_of_mass(
        struct_crds[lipidA_indices].coordinates/10.0, # convert from Angstrom to nm
        np.array([struct_crds.atoms[i].mass for i in lipidA_indices])
    ) 

    r0 = np.linalg.norm(com_pep - com_lipidA) * unit.nanometer

    # rf = 3.0*unit.nanometer # final COM-COM distance to pull to in SMD
    rf = 2.0*r0 # final COM-COM distance to pull to in SMD
    logging.info(f"Peptide COM (nm): {com_pep}")
    logging.info(f"Lipid A COM (nm): {com_lipidA}")
    logging.info(f"Initial COM-COM distance r0 (nm): {r0.value_in_unit(unit.nanometer)}")
    logging.info(f"Final   COM-COM distance rf (nm): {rf.value_in_unit(unit.nanometer)}")

    L = set_cubic_box_from_positions(solvated_psf, struct_crds.positions.in_units_of(unit.angstrom), padding_nm=0.0)
    logging.info(f"Set cubic box length (nm): {L} (also done in build_system)")

    #####################################
    # parameters
    #####################################
    params['temperature'] = 303.15
    params['dt'] = 2.0
    restrain_carbon_nitrogen_atoms = False
    temperature = params['temperature']*unit.kelvin
    EVERY_N_STEP = 1000
    # CONSTANT_V_PULLING = 0.02*unit.nanometer/unit.picosecond # nm/ps
    CONSTANT_V_PULLING = 0.01*unit.nanometer/unit.picosecond # nm/ps
    INIT_FORCE_MAX_CUTOFF = params["init_fmax_cutoff"] #* unit.kilojoules_per_mole/unit.nanometer

    dt = params['dt']*unit.femtoseconds
    total_nvt_equilibriation_steps = math.ceil((50*unit.picoseconds)/dt)
    total_npt_equilibriation_steps = math.ceil((50*unit.picoseconds)/dt)


    total_production_time = (rf-r0)/(CONSTANT_V_PULLING) #50*unit.nanoseconds
    total_n_production_steps= math.ceil(total_production_time/dt)
    params["total_steps"] = total_n_production_steps

    # # set up system
    # system = solvated_psf.createSystem(
    #     ff_params,
    #     nonbondedMethod=app.PME,
    #     nonbondedCutoff=1.2*unit.nanometer,
    #     constraints=app.HBonds,
    #     ewaldErrorTolerance=1e-4,
    # )

    # integrator = mm.LangevinMiddleIntegrator(
    #     temperature,
    #     1.0/unit.picosecond,
    #     dt
    # )
    # integrator.setRandomNumberSeed( int(params["rseed"]) )

    # simulation = app.Simulation(solvated_psf.topology, system, integrator, platform)
    
    # logging.info("setting positions ")
    # simulation.context.setPositions(struct_crds.positions)

    system, solvated_psf, struct_crds = build_system(params)

    simulation = build_simulation(system, solvated_psf, struct_crds, params, platform=platform)

    ###################################
    # energy minimization
    ###################################
    logging.info("performing energy minimization...")
    _pe, _fmax = calculate_state_PE_and_maxForce(simulation)
    _counter=0
    _condn = ( (_pe<0) and (_fmax<INIT_FORCE_MAX_CUTOFF) )

    staged_soft_energy_minimization(simulation)
    _pe, _fmax = calculate_state_PE_and_maxForce(simulation)
    _condn = ( (_pe<0) and (_fmax<INIT_FORCE_MAX_CUTOFF) )
    if not _condn:
        logging.info(f"Failed to minimize energy sufficiently with staged_soft_energy_min")
        logging.info(f"Failed with: {_pe=}, {_fmax=}")
        sys.exit(1)

    logging.info(f"Done energy minimization with: {_pe=}, {_fmax=}, {_counter=}")
    
    _state_reporter = app.StateDataReporter(
            output_state_fpath, 
            EVERY_N_STEP, 
            step=True, 
            time=True, 
            potentialEnergy=True, 
            temperature=True,
            volume=True, 
            density=True, 
            progress=True,
            remainingTime=True, 
            speed=True, 
            totalSteps=total_n_production_steps, 
            separator='\t',
        )
    simulation.reporters.append(_state_reporter)
    
    _dcd_reporter = app.DCDReporter(output_traj_fpath, EVERY_N_STEP, enforcePeriodicBox=False)
    simulation.reporters.append(_dcd_reporter)

    ##############
    # optionally Restrain protein and LPS backbone
    ##############
    logging.info(f"setting velocities to T = {temperature}")
    simulation.context.setVelocitiesToTemperature(temperature, params["rseed"])

    if restrain_carbon_nitrogen_atoms:
        logging.info("Restrain carbon atoms and nitrogen atoms...")
        add_backbone_posres(system, struct_crds, 10.0, periodic_boundaries=True)                
    simulation.context.reinitialize(preserveState=True) # reinitialize context with additional force

    ##############
    # run NVT equilibriation
    ##############
    logging.info(f"Running NVT equil..")
    logging.info(f"Running NVT equilibriation for {total_nvt_equilibriation_steps} steps...")
    simulation.step(total_nvt_equilibriation_steps)

    ##############
    # run NPT equilibriation
    ##############
    logging.info(f"Running NPT equil..")
    logging.info(f"Running NPT equilibriation for {total_npt_equilibriation_steps} steps...")
    _barostat = MonteCarloBarostat(1.0*unit.bar, temperature)
    _barostat.setRandomNumberSeed(params["rseed"])
    system.addForce(
            _barostat
        )
    simulation.context.reinitialize(preserveState=True) # reinitialize context with additional force

    #########################
    # now do the SMD pulling
    #########################
    _additional_params = {
        'fc_pull': 1000.0, # kJ/mol/nm^2
        'v_pulling': CONSTANT_V_PULLING,
        'increment_steps': 10,
        'window_prefix': 'window',
        "total_steps": total_n_production_steps, 
    }
    for _k, _v in _additional_params.items():
        params[_k]=_v

    t0  = time.time()

    _, log = run_smd(
        system=system,
        simulation=simulation,
        rvals=(r0, rf),
        params=params,
        n_windows=n_windows,
        peptide_indices=peptide_indices,
        lps_indices=lipidA_indices
    )

    logging.info(f"{time.time()-t0}s elapsed for run_smd")

    from peplps.steered_md import compute_work
    work = compute_work(log)
    logging.info(f"Total work done during SMD pulling (kJ/mol): {work[-1]:.2f}")
    print(f"Total work done during SMD pulling (kJ/mol): {work[-1]:.2f}")

    with open(os.path.join(params["odir"], f"pep-{params['pepid']}-smd-pulling-log.csv"), 'w') as f:
        f.write("step,r0(nm),cv(nm),force(kJ/mol/nm)\n")
        for step, r0_val, cv_val, force_val in zip(log['step'], log['r0'], log['cv'], log['force']):
            f.write(f"{step},{r0_val:.4f},{cv_val:.4f},{force_val:.2f}\n")

    with open(os.path.join(params["odir"], f"pep-{params['pepid']}-smd-work.csv"), 'w') as f:
        f.write("step,work(kJ/mol)\n")
        for step, work_val in zip(log['step'], work):
            f.write(f"{step},{work_val:.2f}\n")

    ########################
    ########################
    # Do umbrella sampling
    ########################
    ########################
    if (n_windows is not None) and (RUN_UMBRELLA):
        params['window_npt_time']=1.0
        t0 = time.time()
        logging.info(f"Running windows.")
        for i in range(n_windows):
            logging.info(f"On {i+1}/{n_windows}")
            run_window(simulation, i, params)

        logging.info(f"{time.time()-t0}s elapsed for running windows")

        logging.info(f"Plotting histogram and building metadata file for WHAM")
        plot_histograms(params)

        logging.info(f"Quick conversion of cv files for wham")
        convert_for_wham(
            os.path.join( params['odir'], "metafile.txt")
        )

        logging.info(f"Running WHAM")
        wham_return_code = run_wham(
            r0.value_in_unit(unit.nanometer), 
            rf.value_in_unit(unit.nanometer), 
            params
        )

        if wham_return_code==0:

            logging.info(f"Plotting PMF obtained from WHAM")
            plot_pmf(params)

            logging.info(f"Compute estimate of free-energy of binding")
            _free_energy_binding = analyze_pmf(params)

            _header = "pep-id,energy\n"
            _line   = f"{params['pepid']},{_free_energy_binding}"
            with open(os.path.join(params['odir'],"free_energy.csv"), 'w') as f:
                f.writelines(
                    [_header,
                     _line]
                )
        else:
            logging.info(f"WHAM failed")