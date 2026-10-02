import shutil
import warnings
import Bio.PDB
import PeptideBuilder

import os
import numpy as np

from peplps.seq2struct_helpers import (
    biopython_to_pdbfixer_stringio, 
    pdbfixer_workflow,
    setup_and_minimize
)

import mdtraj as mdt
import logging
import argparse

import openmm as omm
import openmm.app as app
from openmm.app import PDBFile, Modeller
from openmm.unit import *

def center_on_origin(topology, positions, tolerance=0.01):
    """
    Check if the structure is centered on the origin and center it if needed.
    
    Parameters:
    -----------
    topology : openmm.app.Topology
        The molecular topology
    positions : list of Vec3
        Atomic positions with units
    tolerance : float, optional
        Distance threshold in nm to determine if structure is already centered
        Default is 0.01 nm (0.1 Angstrom)
    
    Returns:
    --------
    shifted_positions : list of Vec3 or None
        Shifted positions if centering was needed, None if already centered
    """
    if isinstance(positions, list):
        _incoming_unit = positions[0].unit
    else:
        _incoming_unit = positions.unit
        
    # Convert positions to numpy array (strip units for calculation)
    pos_array = np.array([[p.x, p.y, p.z] for p in positions])
    
    # Calculate centroid
    centroid = np.mean(pos_array, axis=0)
    distance_from_origin = np.linalg.norm(centroid)
    
    logging.info(f"\nCentroid position: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}) nm")
    logging.info(f"Distance from origin: {distance_from_origin:.4f} nm")
    
    # Check if already centered within tolerance
    if distance_from_origin < tolerance:
        logging.info(f"Structure is already centered (within {tolerance} nm tolerance)")
        return None
    
    # Need to center - shift all positions
    logging.info(f"Centering structure (shift needed: {distance_from_origin:.4f} nm)")
    shifted_array = pos_array - centroid
    
    # Convert back to OpenMM format with units
    shifted_positions = [omm.Vec3(p[0], p[1], p[2])* _incoming_unit for p in shifted_array]
    
    return shifted_positions

def openmm_to_mdtraj_topology(omm_top):
    return mdt.Topology.from_openmm(omm_top)

def main(params):

    sequence    = params["sequence"]
    sequence_id = params["sequence_id"]
    arrangement = params["arrangement"]
    output_dir  = params["output_dir"]

    if arrangement == "parallel":
        _phi, _psi = -120, 115 # from wikipedia page: https://en.wikipedia.org/wiki/Beta_sheet#Geometry
        _phi, _psi = -119, 113 # from https://bio.libretexts.org/Bookshelves/Biochemistry/Fundamentals_of_Biochemistry_(Jakubowski_and_Flatt)/01%3A_Unit_I-_Structure_and_Catalysis/04%3A_The_Three-Dimensional_Structure_of_Proteins/4.02%3A_Secondary_Structure_and_Loops
    elif arrangement == "antiparallel":
        _phi, _psi = -140, 135 # from wikipedia page: https://en.wikipedia.org/wiki/Beta_sheet#Geometry
        _phi, _psi = -139, 135 # from same bio.libretext
    elif arrangement == "extended":
        _phi, _psi = 180, 180  # https://bio.libretexts.org/Bookshelves/Biochemistry/Fundamentals_of_Biochemistry_(Jakubowski_and_Flatt)/01%3A_Unit_I-_Structure_and_Catalysis/04%3A_The_Three-Dimensional_Structure_of_Proteins/4.02%3A_Secondary_Structure_and_Loops

    phis = [_phi]*(len(sequence)-1)
    psis = [_psi]*(len(sequence)-1)


    structure = PeptideBuilder.make_structure(
        AA_chain=sequence,
        phi = phis, 
        psi_im1 = psis
    )

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save("temp.pdb")

    fixer = biopython_to_pdbfixer_stringio(structure)

    fixed = pdbfixer_workflow(fixer)

    topology  = fixed.topology
    positions = fixed.positions

    _, positions = setup_and_minimize(topology, positions)

    # save the minimized single monomer structure to a pdb
    with open(os.path.join(output_dir, f"peptide_aa.pdb"), "w") as f:
        PDBFile.writeFile(topology, positions, f)
    logging.info(f"Saved minimized structure for {sequence_id} to {output_dir}/{sequence_id}.pdb")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Generate a structure from a sequence and arrangement")
    parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence (1-letter code)")
    parser.add_argument("--sequence_id", type=str, required=True, help="Identifier for the sequence (used in output filename)")
    parser.add_argument("--arrangement", type=str, choices=['extended', "parallel", "antiparallel"], default="parallel", help="Arrangement of beta strands")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output PDB files")

    args = parser.parse_args()
    params = vars(args)

    if args.sequence_id not in args.output_dir:
        output_dir = os.path.join(args.output_dir, args.sequence_id)
        params["output_dir"] = output_dir
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(output_dir, "sequence_to_structure.log"),
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting sequence to structure generation for {args.sequence_id} with arrangement {args.arrangement}")
    logging.info(f"Input sequence: {args.sequence}")
    main(params)
