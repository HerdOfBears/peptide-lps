import os
import sys
import logging

import openmm as omm
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3

import numpy as np
import parmed as pmd

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

def get_lipid_a_indices(topology, resname='ECLIL', chain_id=None,
                        heavy_only=False):
    """
    Return atom indices belonging to the lipid A region of an LPS molecule.

    Designed for CHARMM-GUI LPS Modeler output, where lipid A is a single
    residue (default name 'ECLI' for E. coli lipid A) containing the
    diglucosamine backbone, both phosphates, and all acyl chains. The KDO
    sugars, inner/outer core, and O-antigen are separate residues and are
    excluded.

    If your build uses a different residue name for lipid A (e.g., 'LIPA',
    'LPA', or a custom name), pass it via `resname`. If multiple LPS
    molecules are present, distinguish them with `chain_id` (assign each
    LPS to its own chain in your topology before calling).

    Parameters
    ----------
    topology : openmm.app.Topology
        The system topology.
    resname : str or iterable of str, default 'ECLIL'
        Residue name(s) that constitute lipid A. Matched case-insensitively.
        For E. coli LPS from CHARMM-GUI this is 'ECLIL'.
    chain_id : str or None
        If given, only include atoms whose chain.id matches this value.
        Useful when multiple LPS molecules are present.
    heavy_only : bool, default False
        If True, exclude hydrogens. COM is mass-weighted so this rarely
        changes the COM noticeably, but it can speed up CV evaluation
        slightly.

    Returns
    -------
    indices : list[int]
        Sorted list of atom indices for lipid A.

    Raises
    ------
    ValueError
        If no atoms match — usually means the residue name is wrong for
        your build.

    Notes
    -----
    OpenMM's PDB parser does NOT expose the SEGID column on Residue
    objects, so segment-based filtering is not supported. If your system
    has multiple LPS molecules, give each its own chain ID in the
    topology before calling this function.
    """
    if isinstance(resname, str):
        wanted = {resname.upper()}
    else:
        wanted = {r.upper() for r in resname}

    indices = []
    for residue in topology.residues():
        if residue.name.upper() not in wanted:
            continue
        if chain_id is not None and residue.chain.id != chain_id:
            continue
        for atom in residue.atoms():
            if heavy_only and atom.element is not None \
                    and atom.element.symbol == 'H':
                continue
            indices.append(atom.index)

    if not indices:
        raise ValueError(
            f"No atoms found for lipid A (resname={resname!r}, "
            f"chain_id={chain_id!r}). "
            f"Check your topology's residue names — CHARMM-GUI usually "
            f"writes 'ECLI' for E. coli lipid A but other LPS variants "
            f"use different names (e.g., 'SLIA', 'PLIA')."
        )

    return sorted(indices)

def get_indices(topology,
                lipid_a_resname='ECLIPA',
                lipid_a_chain_id=None,
                peptide_chain_id=None,
                peptide_resnames=None,
                heavy_only=False):
    """
    Convenience wrapper that returns (lipid_a_indices, peptide_indices)
    for a combined solvated system.

    Lipid A atoms are selected by residue name (default 'ECLIL' for
    CHARMM-GUI E. coli LPS). Peptide atoms are selected either by chain
    ID (preferred when available) or by residue-name membership in a set
    of standard amino acids.

    Parameters
    ----------
    topology : openmm.app.Topology
        The combined system topology (peptide + LPS + solvent + ions).
    lipid_a_resname : str or iterable of str, default 'ECLIL'
        Residue name(s) defining lipid A. Forwarded to get_lipid_a_indices.
    lipid_a_chain_id : str or None
        If given, restrict lipid A search to this chain.
    peptide_chain_id : str or None
        If given, the peptide is identified as ALL atoms on this chain.
        This is the most reliable selector when your topology has clean
        chain IDs.
    peptide_resnames : iterable of str or None
        If given, used instead of the default standard-AA set. Useful if
        your peptide contains nonstandard residues (D-amino acids,
        modified termini, etc.). Ignored if peptide_chain_id is given.
    heavy_only : bool, default False
        If True, exclude hydrogens from BOTH index lists.

    Returns
    -------
    (lipid_a_indices, peptide_indices) : (list[int], list[int])
        Sorted atom indices into the given topology.

    Raises
    ------
    ValueError
        If either selection is empty.
    """
    lipid_a_indices = get_lipid_a_indices(
        topology,
        resname=lipid_a_resname,
        chain_id=lipid_a_chain_id,
        heavy_only=heavy_only,
    )

    if peptide_chain_id is not None:
        peptide_indices = []
        for residue in topology.residues():
            if residue.chain.id != peptide_chain_id:
                continue
            for atom in residue.atoms():
                if heavy_only and atom.element is not None \
                        and atom.element.symbol == 'H':
                    continue
                peptide_indices.append(atom.index)

        if not peptide_indices:
            raise ValueError(
                f"No peptide atoms found on chain {peptide_chain_id!r}. "
                f"Available chain IDs: "
                f"{sorted({c.id for c in topology.chains()})}"
            )
    else:
        if peptide_resnames is None:
            wanted = _STANDARD_AA
        else:
            wanted = {r.upper() for r in peptide_resnames}

        peptide_indices = []
        for residue in topology.residues():
            if residue.name.upper() not in wanted:
                continue
            for atom in residue.atoms():
                if heavy_only and atom.element is not None \
                        and atom.element.symbol == 'H':
                    continue
                peptide_indices.append(atom.index)

        if not peptide_indices:
            raise ValueError(
                "No peptide atoms found by residue name. "
                "If your peptide uses nonstandard residue names, pass "
                "them via peptide_resnames, or use peptide_chain_id."
            )

    # Sanity check: the two selections should be disjoint. Overlap would
    # mean a residue is being claimed by both selectors (e.g., an unusual
    # naming collision) and would silently corrupt downstream COM math.
    overlap = set(lipid_a_indices) & set(peptide_indices)
    if overlap:
        raise ValueError(
            f"Lipid A and peptide selections overlap on "
            f"{len(overlap)} atom(s). Check your selectors — likely a "
            f"residue-name or chain-ID collision."
        )

    return sorted(lipid_a_indices), sorted(peptide_indices)

def compute_box_size(peptide_indices, lps_indices, positions, rf, padding):
    """
    Compute box vectors large enough to accommodate the fully-pulled system
    plus a solvent/cutoff buffer on each side.

    The pulling axis is taken to be z. The box is sized so that the two
    molecules can be separated by `rf` along z without either molecule
    seeing its own periodic image (within the given padding).

    Parameters
    ----------
    peptide_indices : list[int]
        Atom indices of the peptide.
    lps_indices : list[int]
        Atom indices of the LPS.
    positions : Quantity (N, 3)
        Positions of the system (e.g. from a PDB / modeller). Used only
        to measure the spatial extents of each molecule.
    rf : Quantity (length)
        Maximum CV value (final COM-COM distance) the SMD will pull to.
    padding : Quantity (length)
        Buffer added on every side of every box dimension.

    Returns
    -------
    box_vectors : tuple of three Vec3 Quantities
        Orthorhombic box vectors suitable for system.setDefaultPeriodicBoxVectors.
    """
    pos = np.array(positions.value_in_unit(unit.nanometer))
    rf_nm = rf.value_in_unit(unit.nanometer)
    pad_nm = padding.value_in_unit(unit.nanometer)

    pep = pos[list(peptide_indices)]
    lps = pos[list(lps_indices)]

    # extents (max - min along each axis) for each molecule
    pep_extent = pep.max(axis=0) - pep.min(axis=0)
    lps_extent = lps.max(axis=0) - lps.min(axis=0)

    # x and y: fit the wider of the two molecules + padding on both sides
    lx = max(pep_extent[0], lps_extent[0]) + 2 * pad_nm
    ly = max(pep_extent[1], lps_extent[1]) + 2 * pad_nm

    # z (pulling axis): both molecules must fit end-to-end with the COMs
    # separated by rf, plus padding on each end. Approximate each molecule's
    # contribution along z as half its z-extent (COM-to-edge).
    lz = (pep_extent[2] / 2.0) + rf_nm + (lps_extent[2] / 2.0) + 2 * pad_nm

    a = omm.Vec3(lx, 0.0, 0.0) * unit.nanometer
    b = omm.Vec3(0.0, ly, 0.0) * unit.nanometer
    c = omm.Vec3(0.0, 0.0, lz) * unit.nanometer
    return (a, b, c)

def set_cubic_box_from_positions(psf, positions, padding_nm=1.5):
    """Set periodic box vectors based on bounding box of positions + padding."""
    X = np.array([[p.x, p.y, p.z] for p in positions], dtype=float)  # nm
    mins = X.min(axis=0)/10
    maxs = X.max(axis=0)/10
    lengths = (maxs - mins) + 2.0*padding_nm  # nm
    print(lengths)
    # Use a cube for simplicity
    L = float(np.max(lengths))
    a = Vec3(lengths[0], 0, 0) * unit.nanometer
    b = Vec3(0, lengths[1], 0) * unit.nanometer
    c = Vec3(0, 0, lengths[2]) * unit.nanometer

    if type(psf)==pmd.structure.Structure:
        _v = a+b+c
        # psf.topology.setPeriodicBoxVectors(_v)
        psf.box = [lengths[0], lengths[1], lengths[2], 90, 90, 90]  # ParmEd uses Å
    else:
        psf.setBox(lengths[0], lengths[1], lengths[2])
    
    return L

def build_system(params):
    """
    Build an OpenMM System from a configuration dictionary.

    Parameters
    ----------
    params : dict
        Configuration dictionary with keys:
            'psf' : str, path to solvated system CHARMM PSF file,
            'pdb' : str, path to solvated system coordinates (PDB or CHARMM CRD),
            'ff_toppar_path' : str, path to CHARMM force field toppar directory,
    """

    #####################################
    # load forcefield parameters
    #####################################
    param_files = []
    for _f in os.listdir(params["ff_toppar_path"]):
        if _f =="toppar.str":
            continue
        if _f.endswith(".str") or _f.endswith(".prm") or _f.endswith(".rtf"):
            param_files.append( os.path.join(params["ff_toppar_path"], _f) ) 

    ff_params = app.CharmmParameterSet(*param_files)

    #####################################
    # load the solvated systems
    solvated_psf = app.CharmmPsfFile(params["psf"])
    struct_crds  = pmd.load_file(    params["pdb"])

    # struct crds positions are in Angstrom
    L = set_cubic_box_from_positions(solvated_psf, struct_crds.positions.in_units_of(unit.angstrom), padding_nm=0.0)
    # logging.info(f"Set cubic box length (nm): {L}")

    # set up system
    system = solvated_psf.createSystem(
        ff_params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=params['nonbondedCutoff']*unit.nanometer,
        constraints=app.HBonds,
        ewaldErrorTolerance=1e-4,
    )

    return system, solvated_psf, struct_crds

def build_simulation(system, solvated_psf, struct_crds, params, platform=None):
    """
    Build an OpenMM Simulation from a configuration dictionary.

    Parameters
    ----------
    system : openmm.System
        The OpenMM System object.
    solvated_psf : openmm.app.CharmmPsfFile
        The PSF file object for the solvated system.
    struct_crds : parmed.Structure
        The ParmEd Structure object containing coordinates.
    params : dict
        Configuration dictionary with keys:
            'rseed' : int, random seed for Langevin integrator,
            'temperature' : float, temperature in Kelvin,
            'nonbondedCutoff' : float, nonbonded cutoff in nanometers.
    platform : openmm.Platform = None
        What hardware platform to run on.

    Returns
    -------

    """

    if platform is None:
        platform = omm.Platform.getPlatformByName("OpenCL") if omm.Platform.getNumPlatforms() else None


    integrator = omm.LangevinMiddleIntegrator(
        params['temperature']*unit.kelvin,
        1.0/unit.picosecond,
        params['dt']*unit.femtoseconds
    )
    integrator.setRandomNumberSeed( int(params["rseed"]) )

    print(type(system), type(integrator), type(platform))
    logging.info(f"{type(system)}, {type(integrator)}, {type(platform)}")
    simulation = app.Simulation(solvated_psf.topology, system, integrator, platform)
    print(system.getNumParticles(),
      struct_crds.topology.getNumAtoms(),
      len(struct_crds.positions),
      solvated_psf.topology.getNumAtoms()
    )
    simulation.context.setPositions(struct_crds.positions)

    # simulation.context.setVelocitiesToTemperature(
    #     params['temperature']*unit.kelvin, 
    #     params.get('rseed',42)
    # )
    return simulation

def get_force_by_name(system, name):
    """
    Iterate through forces in the system object looking for
    one named 'name'.

    Parameters:
    -----------
    system: omm.System
        An OpenMM system
    name: str
        The name of the force to return
    
    Returns:
    --------
    force: omm.Force (could be CustomCVForce e.g.)
        The force with the given name 
    """
    for f in system.getForces():
        if f.getName() == name:
            return f
    raise ValueError(f'no force named {name!r}')


##################################
# helpers to convert some files for wham
###################################
def read_metafile(path):
    """Parse a WHAM metafile into a list of (filepath, center, k) tuples."""
    entries = []
    base = os.path.dirname(os.path.abspath(path))

    with open(path) as fobj:
        for lineno, line in enumerate(fobj, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f'{path}:{lineno}: expected at least 3 fields, got {len(fields)}'
                )

            fpath, center, k = fields[0], float(fields[1]), float(fields[2])

            # resolve relative paths against the metafile's own directory
            if not os.path.isabs(fpath):
                fpath = os.path.join(base, fpath)

            entries.append((os.path.normpath(fpath), center, k))

    return entries


def convert_timeseries(csv_path):
    """
    Read a comma-delimited cv_values file and write the whitespace-delimited
    two-column version next to it. Returns the new path.
    """
    data = np.loadtxt(csv_path, delimiter=',', ndmin=2)

    if data.shape[1] < 2:
        raise ValueError(f'{csv_path}: expected >=2 columns, got {data.shape[1]}')

    out = data[:, :2]                       # keep index/time and CV only
    txt_path = os.path.splitext(csv_path)[0] + '.txt'

    np.savetxt(txt_path, out, fmt='%.6f', delimiter=' ')
    return txt_path, len(out)


def convert_for_wham(metafile, HALVE_K=False):
    entries = read_metafile(metafile)
    if not entries:
        sys.exit(f'No usable entries found in {metafile}')

    out_lines = []
    for csv_path, center, k in entries:
        if not os.path.exists(csv_path):
            sys.exit(f'Missing timeseries file: {csv_path}')

        txt_path, nrows = convert_timeseries(csv_path)
        k_out = k / 2.0 if HALVE_K else k

        out_lines.append(f'{txt_path} {center:.6f} {k_out:.6f}\n')
        print(f'{os.path.basename(csv_path)} -> {os.path.basename(txt_path)} '
              f'({nrows} rows, center={center:.4f}, k={k_out:.1f})')

    new_metafile = os.path.join(
        os.path.dirname(os.path.abspath(metafile)), 'metafile_wham.txt'
    )
    with open(new_metafile, 'w') as fobj:
        fobj.writelines(out_lines)

    print(f'\nWrote {new_metafile} ({len(out_lines)} windows)')
    if HALVE_K:
        print('Force constants halved for WHAM\'s k*(r-r0)^2 convention.')
