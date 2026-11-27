from openmm.app import *
from openmm import *
from openmm.unit import *

import time
import math
import os
import logging
import numpy as np
from peplps.helpers import build_system, set_cubic_box_from_positions

# Function to add backbone position restraints
def add_backbone_posres(system, pdb, restraint_force, periodic_boundaries=True):
    if periodic_boundaries:
        force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    else:
        force = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')        

    force_amount = restraint_force * kilocalories_per_mole/angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    # restrain all Calpha atoms
    for atom in pdb.topology.atoms():
        _carbon_condn = (atom.name.startswith("C"))
        _nitro_condn  = (atom.name.startswith("N"))
        _heavy_condn  = (atom.name in ('CA', 'C', 'N') )
        if _carbon_condn or _nitro_condn or _heavy_condn: # all heavy atoms
            force.addParticle(atom.index, pdb.positions[atom.index])

    system.addForce(force)

def calculate_state_PE_and_maxForce(simulation:simulation.Simulation):
    """
    Calculates the system's potential energy (PE) and max force.

    Returns:
        pe: the potential energy of the system
        max_force: the maximum force exerted on an atom
    """
    _state = simulation.context.getState(getEnergy=True, getForces=True)
    pe = _state.getPotentialEnergy()
    pe = pe._value
    _forces = _state.getForces(asNumpy=True)
    max_force = np.max(np.linalg.norm(_forces, axis=1))
    return pe, max_force

def preminimize_with_stock_openmm(params, positions, platform, properties,
                                  tolerance, max_iterations):
    """
    Minimize with standard OpenMM forces (no cuPotential plugin).

    The stock system can use OpenMM's CPU fallback for very large forces,
    which plugin forces cannot. Use this before minimizing a plugin
    simulation, then pass the returned positions to the plugin Context.

    Parameters
    ----------
    params : dict
        The configuration dictionary passed to ``build_system`` (psf, pdb,
        ff_toppar_path, nonbondedCutoff, ...). ``use_cuPotential`` is
        overridden to False here, so the original dict is not modified.
    positions : list of openmm.Vec3 or Quantity
        Starting coordinates, in the same atom order as the PSF (e.g.
        ``struct_crds.positions`` or the positions from the plugin Context).
    platform : openmm.Platform
        Platform for the temporary simulation, normally CUDA.
    properties : dict
        Platform properties, e.g. ``{"Precision": "mixed"}``. Use ``{}``
        for the CPU platform.
    tolerance : openmm.unit.Quantity
        Convergence criterion: minimization stops when the root-mean-square
        force falls below this value, in force units (kJ/mol/nm).
    max_iterations : int
        Maximum number of minimizer iterations; 0 means no limit (run until
        the tolerance is reached).

    Returns
    -------
    list of openmm.Vec3 (Quantity, nm)
        Minimized positions, ready for ``simulation.context.setPositions()``.
    """
    stock_system, psf, _ = build_system({**params, "use_cuPotential": False})
    sim = Simulation(psf.topology, stock_system,
                         VerletIntegrator(0.001 * picoseconds),
                         platform, properties)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    print("Stock pre-minimization energy:", state.getPotentialEnergy())
    del sim                                   # free GPU memory
    return state.getPositions()

def staged_soft_energy_minimization(simulation:simulation.Simulation):
    """
    Performs up to four stages of 'soft' energy minimization.
    Tries to bring the potential energy below 0 and the 
    maximum force value below 15,000

    When a system has *very* high potential energy and force values, 
    we have to perform energy minimization with high thresholds 
    then progressively lower the threshold value. 

    Parameters
    ----------
    simulation: openmm.app.simulation.Simulation
        The system whose energy we want to minimize.

    Returns
    -------
    None
        Acts 'in-place' on the simulation.

    Notes:
    - This may not always succeed in sufficiently minimizing the energy.
    """
    _pe, _max_force = calculate_state_PE_and_maxForce(simulation)

    _stage_to_energy_tolerance = {
        "stage1": 10_000,
        "stage2":  1_000,
        "stage3":    100,
        "stage4":     10,
        "stage5":      5
    }
    _stage = "stage1"
    _max_iterations = 500

    _condn_pe = (_pe < 0 )
    _condn_max_force = (_max_force < 5_000)
    if (not _condn_pe) or (not _condn_max_force):
        logging.info(f"PE and maxForce still quite high. Doing soft energy minimization")
        logging.info(f"PE={_pe} kJ/mol and maxForce={_max_force}")

        _counter = 0
        while ((not _condn_pe) or (not _condn_max_force)):
            logging.info(f"staged soft energy minimization at stage={_stage}")
            t0 = time.time()
            energy_tolerance = _stage_to_energy_tolerance[_stage]

            simulation.minimizeEnergy(
                tolerance=energy_tolerance*(kilojoules_per_mole/nanometer),
                maxIterations=_max_iterations
            )

            logging.info(f"softer energy minimization duration: {round(time.time() - t0)}s per run of .minimizeEnergy ")

            _pe, _max_force = calculate_state_PE_and_maxForce(simulation)
            logging.info(f"PE = {_pe} (kJ/mol), maxForce = {_max_force}")

            _condn_pe = (_pe < 0)
            _condn_max_force = (_max_force < 2_000)

            _counter += 1
            if _counter in [2,3]:
                _stage = "stage2"
            elif _counter in [4,5]:
                _max_iterations = 500
                _stage = "stage3"
            elif _counter in [6,7]:
                _max_iterations = 1000
                _stage = "stage4"
            elif _counter in [8,9]:
                _max_iterations = 1500
                _stage = "stage5"
            elif _counter > 9:
                logging.info(f"forceably leaving energy minimization loop {_counter=} > 8")
                break
