import openmm as omm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd

import logging
import os
import subprocess
import shutil

import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as spytegrate
from peplps.helpers import build_simulation, build_system, get_force_by_name

# Standard amino acid residue names plus common CHARMM protonation variants.
# Extend this set if your peptide contains nonstandard residues.
_STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # CHARMM histidine protonation states
    'HSD', 'HSE', 'HSP',
    # CHARMM neutral / alternate protonation variants
    'LSN', 'ASPP', 'GLUP', 'CYM',
}

def _build_com_distance_cv(peptide_indices, lps_indices, system):
    """
    Build a CustomCentroidBondForce that measures the COM-COM distance
    between the peptide and LPS, mass-weighted.
    """
    cv = omm.CustomCentroidBondForce(2, 'distance(g1, g2)')

    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton)
         for i in range(system.getNumParticles())]
    )

    g1_weights = [float(masses[i]) for i in peptide_indices]
    g2_weights = [float(masses[i]) for i in lps_indices]

    g1 = cv.addGroup(list(peptide_indices), g1_weights)
    g2 = cv.addGroup(list(lps_indices), g2_weights)
    cv.addBond([g1, g2], [])
    return cv


def run_smd(system: omm.System,
            simulation: app.Simulation,
            rvals: tuple,
            params: dict,
            peptide_indices,
            lps_indices,
            n_windows=None,
            ):
    """
    Run constant-velocity steered MD pulling the COM-COM distance between
    a peptide and an LPS molecule from rvals[0] to rvals[1].

    Parameters
    ----------
    system : openmm.System
        The system. The pulling force will be added to it.
    simulation : openmm.app.Simulation
        The simulation. Must already have positions set and (ideally) be
        equilibrated.
    rvals : (Quantity, Quantity)
        (r0_initial, rf) — the starting and final COM-COM distance.
    params : dict
        Required keys:
            'fc_pull'     : float, kJ/mol/nm^2 — restraint force constant
            'total_steps' : int, total MD steps for the SMD pull
        Optional keys:
            'v_pulling'   : Quantity, default 0.02 nm/ps
            'increment_steps' : int, default 10
            'window_prefix'   : str, default 'window'
    peptide_indices, lps_indices : list[int]
        Atom indices defining the two COM groups.
    n_windows : int or None
        If given, save N evenly spaced configurations along the pull as
        starting structures for umbrella sampling.

    Returns
    -------
    pullingForce : openmm.CustomCVForce
        The restraint force that was added to the system. Keep this around
        if you want to query forces or compute work afterwards.
    log : dict
        Time series with keys 'step', 'r0' (nm), 'cv' (nm), 'force' (kJ/mol/nm).
    """

    r0 = rvals[0]
    rf = rvals[1]

    total_steps = params['total_steps']
    fc_pull     = params['fc_pull'] * unit.kilojoules_per_mole / unit.nanometers**2
    v_pulling   = params.get('v_pulling',
                           0.02 * unit.nanometer / unit.picosecond)
    increment_steps = params.get('increment_steps', 10)
    window_prefix   = params.get('window_prefix', 'window')
    # window_dir      = params['window_dir']

    dt = simulation.integrator.getStepSize()

    # Build the COM-COM distance CV
    cv = _build_com_distance_cv(peptide_indices, lps_indices, system)

    pullingForce = omm.CustomCVForce('0.5 * fc_pull * (cv-r0)^2')
    pullingForce.addGlobalParameter('fc_pull', fc_pull)
    pullingForce.addGlobalParameter('r0', r0)
    pullingForce.addCollectiveVariable("cv", cv)
    pullingForce.setName("umbrella")

    system.addForce(pullingForce)
    simulation.context.reinitialize(preserveState=True)

    # Window bookkeeping
    if n_windows is not None:
        windows = np.linspace(
            r0.value_in_unit(unit.nanometer),
            rf.value_in_unit(unit.nanometer),
            n_windows,
        )
        window_coords = []
        window_pbc_vecs=[]
        window_rvals  = []
        window_index = 0

    structure = pmd.openmm.load_topology(simulation.topology, system=system)

    # Log of (step, r0, cv, restraint force magnitude) for later analysis
    log = {'step': [], 'r0': [], 'cv': [], 'force': []}

    # save xml of the system set up
    # simulation.saveState(
    #     os.path.join(params['odir'], "system.xml")
    # )
    with open(os.path.join(params['odir'], "system.xml"), 'w') as f:
        f.write(omm.XmlSerializer.serialize(system))

    n_loops = total_steps // increment_steps
    for i in range(n_loops):
        simulation.step(increment_steps)
        current_cv_value = pullingForce.getCollectiveVariableValues(
            simulation.context
        )[0]
        step_count = (i + 1) * increment_steps

        r0_nm = r0.value_in_unit(unit.nanometer)
        fc_val = fc_pull.value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer**2
        )
        # F = -dU/dcv = -k*(cv - r0); we log magnitude
        force_mag = abs(fc_val * (current_cv_value - r0_nm))

        log['step'].append(step_count)
        log['r0'].append(r0_nm)
        log['cv'].append(current_cv_value)
        log['force'].append(force_mag)

        if step_count % 5000 == 0:
            print(f"step {step_count:>8d}  r0 = {r0_nm:.4f} nm  "
                  f"cv = {current_cv_value:.4f} nm  "
                  f"|F| = {force_mag:.2f} kJ/mol/nm")

        # advance the restraint center
        r0 = r0 + v_pulling * dt * increment_steps
        simulation.context.setParameter('r0', r0)

        # save window starting structures based on r0 (monotonic), not cv
        if n_windows is not None:
            r0_now = r0.value_in_unit(unit.nanometer)
            while (window_index < len(windows)
                   and r0_now >= windows[window_index]):
                state = simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=False
                )
                window_coords.append(state.getPositions())
                window_pbc_vecs.append(state.getPeriodicBoxVectors())
                window_rvals.append(r0_now)
                window_index += 1

    if n_windows is not None:
        os.makedirs(
            os.path.join( params['odir'], "windows/"),
            exist_ok=True
        )
        for i, coords in enumerate(window_coords):
            _pbc_vecs = window_pbc_vecs[i]
            structure.positions = coords
            structure.box_vectors = _pbc_vecs
            structure.save(
                os.path.join( params['odir'],f'windows/{window_prefix}_{i}.pdb')
            )
            # with open( os.path.join( params['odir'],f'windows/{window_prefix}_{i}.pdb'), 'w') as outfile:
            #     app.PDBFile.writeFile(simulation.topology, coords, outfile)

            with open(os.path.join( params['odir'],f"windows/window_rvals.csv"), 'a') as fobj:
                fobj.write( f"{window_rvals[i]}\n" )

    return pullingForce, log


def get_spring_force(simulation, pullingForce, fc_pull):
    """
    Return a snapshot of the current restraint state.

    Parameters
    ----------
    simulation : app.Simulation
    pullingForce : openmm.CustomCVForce
        The restraint force returned by run_smd.
    fc_pull : Quantity
        The force constant used for the restraint (kJ/mol/nm^2).

    Returns
    -------
    (cv_value, r0, force_magnitude)
        cv_value         : float, current COM-COM distance (nm)
        r0               : float, current restraint center (nm)
        force_magnitude  : float, |k*(cv-r0)| (kJ/mol/nm)
    """
    cv_value = pullingForce.getCollectiveVariableValues(
        simulation.context
    )[0]

    # r0 was registered as a global parameter. It may come back as a plain
    # float (its value in the parameter's stored unit, nm here).
    r0_param = simulation.context.getParameter('r0')
    if unit.is_quantity(r0_param):
        r0 = r0_param.value_in_unit(unit.nanometer)
    else:
        r0 = float(r0_param)

    fc_val = fc_pull.value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer**2
    )
    force_magnitude = abs(fc_val * (cv_value - r0))

    return cv_value, r0, force_magnitude


def compute_work(log):
    """
    Compute the work done by the moving restraint along an SMD trajectory.

    For constant-velocity SMD with bias w(x, t) = 0.5 k (x - r0(t))^2,
    the work done on the system by the moving restraint is

        W = integral over t of (dw/dr0)(dr0/dt) dt
          = integral over r0 of dw/dr0 dr0
          = integral of -k (cv - r0) dr0

    We approximate this by the trapezoidal rule using the recorded series.

    Parameters
    ----------
    log : dict
        The log dict returned by run_smd (contains 'r0' and 'cv' arrays in nm).

    Returns
    -------
    work_series : np.ndarray
        Cumulative work (kJ/mol) at each logged step. work_series[-1] is
        the total work over the pull. Note this comes from a single
        non-equilibrium trajectory and is only a rough estimate of binding
        free energy — for a proper estimate use Jarzynski-averaging over
        many independent pulls or run umbrella sampling + WHAM with the
        windows produced by run_smd.
    """
    r0 = np.asarray(log['r0'])
    cv = np.asarray(log['cv'])

    # need fc_pull. We don't store it directly in the log because we log
    # |F|; recover k from |F| and (cv - r0). Guard against div-by-zero.
    force = np.asarray(log['force'])
    diff = np.abs(cv - r0)
    # Use the first sample where diff is meaningfully nonzero
    mask = diff > 1e-8
    if not np.any(mask):
        return np.zeros_like(r0)
    k = np.median(force[mask] / diff[mask])

    # dW = -k * (cv - r0) * dr0
    integrand = -k * (cv - r0)
    dr0 = np.diff(r0)
    # trapezoidal rule
    avg_integrand = 0.5 * (integrand[:-1] + integrand[1:])
    increments = avg_integrand * dr0
    work = np.concatenate([[0.0], np.cumsum(increments)])
    return work

def run_window(simulation, window_ix:int, params:dict=None):

    centers = np.loadtxt(os.path.join(params['odir'], 'windows/window_rvals.csv'), delimiter=",")
    if window_ix >= len(centers):
        logging.info(f"{window_ix=} > {len(centers)} ")
        return 0
    r0 = centers[window_ix] * unit.nanometer
    K  = 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2

    #####
    # 1. rebuild the identical system from a single serialized copy
    # topology = app.PDBFile( params['system_pdb'] ).topology
    _system_fpath = os.path.join(params['odir'], 'system.xml')
    if not os.path.exists(_system_fpath):
        _system_fpath = params['system_xml']
    system   = omm.XmlSerializer.deserialize(
        open( _system_fpath ).read()
    )
    system = omm.XmlSerializer.deserialize(open(_system_fpath).read())
    if not isinstance(system, omm.System):
        raise TypeError(
            f'{_system_fpath} is a {type(system).__name__}, not a System'
        )

    system_psf = app.CharmmPsfFile(params['psf'])

    # set positions
    _window_dir = os.path.join(params['odir'], 'windows')
    _window_fpath = os.path.join(
        _window_dir, 
        f"window_{window_ix}.pdb"
    )
    # _window_pdb = app.PDBFile(_window_fpath)
    _window_pdb = pmd.load_file(_window_fpath)
    _pbc_box_vectors = _window_pdb.box_vectors

    for idx, f in enumerate(system.getForces()):
        f.setForceGroup(idx)

    simulation = build_simulation(system, system_psf, _window_pdb, params)

    simulation.context.setPeriodicBoxVectors(*_pbc_box_vectors)

    # set global parameters to window specific values
    simulation.context.setParameter('r0', r0)
    simulation.context.setParameter('fc_pull', K)

    # reset velocities 
    simulation.context.setVelocitiesToTemperature(
        params['temperature']*unit.kelvin, 
        params["rseed"]
    )

    e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f'window {window_ix} initial PE: {e}')

    for idx, f in enumerate(system.getForces()):
        e = simulation.context.getState(getEnergy=True, groups={idx}).getPotentialEnergy()
        print(f'{idx:2d} {f.getName():30s} {e}')


    # run short equilibration with new positions and r0
    simulation.step(1000)

    ##### 
    # run the data collection

    dt_ps = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
    total_ps = params['window_npt_time'] * 1000.0          # ns -> ps
    total_steps = int(round(total_ps / dt_ps))

    # frequency to record the current CV value
    record_steps = 1_000

    umbrellaForce = get_force_by_name(system, "umbrella")
    # run the simulation and record the value of the CV.
    cv_values=[]
    for i in range(total_steps//record_steps):
        simulation.step(record_steps)

        # get the current value of the cv
        current_cv_value = umbrellaForce.getCollectiveVariableValues(simulation.context)
        current_potential_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        cv_values.append([
                (i+1)*record_steps*params.get("dt",2), # assumes in femtoseconds
                current_cv_value[0], 
                current_potential_energy.value_in_unit(unit.kilojoule_per_mole)
            ]
        )
    # save the CV timeseries to a file so we can postprocess
    os.makedirs(
        os.path.join(
            params['odir'],
            "cv_values"
        ),
        exist_ok=True
    )
    np.savetxt(
        os.path.join(
            params['odir'],
            f'cv_values/cv_values_window_{window_ix}.csv'
        ), 
        np.array(cv_values),
        delimiter=","
    )

    print('Completed window', window_ix)

def plot_histograms(params:dict=None):
    """
    Plots a histogram of the collective variable values obtained
    throughout umbrella sampling
    """
    centers = np.loadtxt(os.path.join(params['odir'], 'windows/window_rvals.csv'), delimiter=",")

    n_windows = len(centers)
    K=1000 # spring constant

    # plot the histograms
    metafilelines = []
    for i in range(n_windows):
        data = np.loadtxt(
            os.path.join(
                params['odir'],
                f'cv_values/cv_values_window_{i}.csv'
            ), 
            delimiter=","
        )
        plt.hist(data[:,1])
        metafileline = f'cv_values/cv_values_window_{i}.csv {centers[i]} {K}\n'
        metafilelines.append(metafileline)

    plt.xlabel("r (nm)")
    plt.ylabel("count")

    plt.savefig(os.path.join(params['odir'],"cv_vals_hist.png"),dpi=200)
    
    with open(os.path.join(params['odir'],"metafile.txt"), "w") as f:
        f.writelines(metafilelines)

    plt.close()

def run_wham(r0, rf, params, nbins=50, tol=1e-6, temperature=303.15, numpad=0):
    """
    Run Grossfield WHAM on metafile_wham.txt and write pmf.txt.

    Returns 0 on success, nonzero on failure.
    """
    odir = params['odir']
    if ((not odir.startswith("./")) and (not odir.startswith("/"))):
        odir = os.path.join("./", odir)
    metafile = os.path.join(odir, 'metafile_wham.txt')
    pmf_out  = os.path.join(odir, 'pmf.txt')
    log_out  = os.path.join(odir, 'wham_log.txt')

    exe = params.get('wham_exe', 'wham')
    if shutil.which(exe) is None:
        print(f'run_wham: executable {exe!r} not found on PATH')
        return 2

    if not os.path.exists(metafile):
        print(f'run_wham: missing {metafile}')
        return 3

    # convert to abspath
    metafile = os.path.abspath(metafile)
    pmf_out  = os.path.abspath(pmf_out)

    cmd = [
        exe,                # program
        f'{float(r0):.6f}', # lower bound of CVvals
        f'{float(rf):.6f}', # upper bound of CVvals
        str(int(nbins)),    # number of bins for WHAM
        f'{float(tol):g}',  # error tolerance
        f'{float(temperature):.4f}', # temperature system was at in K
        str(int(numpad)),
        metafile,           # file pointing to all CV data 'streams' 
        pmf_out,            # output file name/path
        f"{10}",            # number of Monte Carlo bootstrap samples
        f"{42}"             # random seed for Monte Carlo bootstrapping
    ]

    try:
        with open(log_out, 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, cwd=odir, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(f'run_wham: wham exited {e.returncode}; see {log_out}')
        return e.returncode or 1
    except subprocess.TimeoutExpired:
        print('run_wham: wham timed out')
        return 4
    except OSError as e:
        print(f'run_wham: {e}')
        return 5

    if not os.path.exists(pmf_out) or os.path.getsize(pmf_out) == 0:
        print(f'run_wham: no output written to {pmf_out}')
        return 6

    return 0

def plot_pmf(params:dict=None):

    pmf = np.loadtxt(
        os.path.join(params['odir'], "pmf.txt")
    )

    # plt.plot(pmf[:,0], pmf[:,1])
    plt.errorbar(
        pmf[:,0],
        pmf[:,1],
        yerr=pmf[:,2]
    )
    plt.xlabel("r (nm)", fontsize=18)
    plt.ylabel("PMF (kJ/mol)", fontsize=18)
    plt.savefig(
        os.path.join(params['odir'], "pmf.png"),dpi=200
    )
    plt.close()


def analyze_pmf(params:dict=None):
    """
    Takes a pmf and estimate difference in free energy of binding.
    using Eq. 19 from 
    Reif and Zacharias' 
    Computational Tools for Accurate Binding Free-Energy Prediction

    Take a pmf, split into 'bound' and 'unbound'
    stages. Compute the integral over each of them.
    """
    T = params['temperature'] # in kelvin
    # RT = (8.314)*T    # J/(mol)
    RT = (8.314e-3)*T # kJ/mol

    _pmf_fpath = os.path.join(params['odir'],'pmf.txt')
    _data = np.loadtxt(_pmf_fpath)
    _rvals, _pmf_raw = _data[:,0], _data[:,1]

    _n_rvals = len(_rvals)

    # assume first ~third is   'bound'
    _rvals_bound    = _rvals[  :int(_n_rvals/3)]
    _pmf_raw_bound  = _pmf_raw[:int(_n_rvals/3)]
    _boltzmann_weighted_bound = np.exp(-(1/RT)*_pmf_raw_bound)
    _integrated_bound = spytegrate.trapezoid(
        _boltzmann_weighted_bound,
        _rvals_bound
    )

    # assume last  ~third is 'unbound'
    _rvals_unbound    = _rvals[  -int(_n_rvals/3):]
    _pmf_raw_unbound  = _pmf_raw[-int(_n_rvals/3):]
    _boltzmann_weighted_unbound = np.exp(-(1/RT)*_pmf_raw_unbound)
    _integrated_unbound = spytegrate.trapezoid(
        _boltzmann_weighted_unbound,
        _rvals_unbound
    )
    _length_unbound = spytegrate.trapezoid(
        np.ones_like(_rvals_unbound),
        _rvals_unbound
    )

    _ratio_bound_unbound = _integrated_bound/(_integrated_unbound/_length_unbound)

    # 'raw' Free energy of binding
    # technically needs to have corrections added
    # Eq. 19 from Computational Tools for Accurate Binding Free-Energy Prediction (2022)
    _raw_free_energy_of_binding = -RT*np.log(_ratio_bound_unbound)

    return _raw_free_energy_of_binding