import openmm as mm
import openmm.app as app
import openmm.unit as unit

from Bio.PDB import PDBIO
from pdbfixer import PDBFixer
from io import StringIO
import numpy as np

def biopython_to_pdbfixer_stringio(structure):
    """Convert BioPython Structure to PDBFixer without temp file"""
    
    # Write BioPython structure to string
    io = PDBIO()
    io.set_structure(structure)
    
    pdb_string = StringIO()
    io.save(pdb_string)
    pdb_string.seek(0)  # Reset to beginning
    
    # Load with PDBFixer from string
    fixer = PDBFixer(pdbfile=pdb_string)
    
    return fixer

def pdbfixer_workflow(fixer):
    """
    Standard PDBFixer workflow.


    """
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    return fixer

def setup_and_minimize(topology, positions, padding=1.0*unit.nanometers, 
                       max_iterations=100, tolerance=10.0, verbose=True):
    """
    Set up periodic boundary conditions and perform energy minimization.
    
    Parameters
    ----------
    topology : openmm.app.Topology
        The molecular topology
    positions : list or np.ndarray
        Atomic positions (with or without units)
    padding : openmm.unit.Quantity, optional
        Padding around the bounding box (default: 1.0 nm)
    max_iterations : int, optional
        Maximum minimization iterations (default: 100)
    tolerance : float, optional
        Energy tolerance in kJ/mol (default: 10.0)
    verbose : bool, optional
        Print progress information (default: True)
    
    Returns
    -------
    topology : openmm.app.Topology
        Topology with updated periodic box vectors
    minimized_positions : openmm.unit.Quantity
        Energy-minimized positions
    """
    
    # Ensure positions have units
    if not hasattr(positions, 'unit'):
        positions = positions * unit.nanometers
    
    # Convert to numpy array for calculations
    pos_array = np.array(positions.value_in_unit(unit.nanometers))
    
    # Find axis-aligned bounding box
    min_coords = pos_array.min(axis=0)
    max_coords = pos_array.max(axis=0)
    box_size = max_coords - min_coords
    
    if verbose:
        print(f"Bounding box size: {box_size} nm")
        print(f"Min coords: {min_coords} nm")
        print(f"Max coords: {max_coords} nm")
    
    # Add padding
    padding_value = padding.value_in_unit(unit.nanometers)
    padded_box = box_size + 2 * padding_value
    
    # Set periodic box vectors (axis-aligned cubic/rectangular box)
    box_vectors = np.diag(padded_box) * unit.nanometers
    topology.setPeriodicBoxVectors(box_vectors)
    
    if verbose:
        print(f"Padded box size: {padded_box} nm")
        print(f"Box vectors set:\n{box_vectors}")
    
    # Center the molecule in the box
    center_offset = min_coords - padding_value
    centered_positions = (pos_array - center_offset) * unit.nanometers
    
    # Create a force field for energy minimization
    # CHARMM36m is excellent for peptides, including random sequences
    forcefield = app.ForceField('charmm36_2024.xml', 'charmm36_2024/water.xml')
    
    # Create system
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometers,
        constraints=app.HBonds
    )
    
    # Set up integrator (not used for minimization, but required)
    integrator = mm.LangevinIntegrator(
        300*unit.kelvin,
        1.0/unit.picoseconds,
        2.0*unit.femtoseconds
    )
    
    # Create simulation
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(centered_positions)
    
    # Get initial energy
    if verbose:
        state = simulation.context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy()
        print(f"\nInitial potential energy: {initial_energy}")
    
    # Perform energy minimization
    if verbose:
        print(f"Minimizing (max_iterations={max_iterations}, tolerance={tolerance} kJ/mol)...")
    
    simulation.minimizeEnergy(
        tolerance=tolerance*(unit.kilojoules/(unit.nanometer*unit.mole)),
        maxIterations=max_iterations
    )
    
    # Get final energy and positions
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    
    if verbose:
        final_energy = state.getPotentialEnergy()
        print(f"Final potential energy: {final_energy}")
        print(f"Energy change: {final_energy - initial_energy}")
        print("Minimization complete!")
    
    return topology, minimized_positions