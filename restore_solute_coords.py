#!/usr/bin/env python
"""
Copy solute coordinates back into a solvated PDB, matching atoms by ORDER.

Why: VMD's solvate/autoionize rebuild the system with psfgen, which matches
coordinates by atom name. PDB files only hold 4-character names, so LPS atoms
with 5-character names (H810R, H810S, ...) lose their coordinates and end up at
(0,0,0). The solute atoms are still the first atoms of the solvated system, in
the same order, so their coordinates can be copied back by position.

Usage:
    python restore_solute_coords.py combined.pdb solvated_ionized.pdb [-o out.pdb]

Without -o the solvated PDB is updated in place.
"""

import argparse
from pathlib import Path

import numpy as np


def read_pdb(path):
    """Return (all lines, indices of atom lines, Nx3 coordinates, box or None)."""
    lines = Path(path).read_text().splitlines(keepends=True)
    atom_idx = [i for i, l in enumerate(lines) if l.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])]
                    for i in atom_idx])
    box = None
    for l in lines:
        if l.startswith("CRYST1"):
            box = np.array([float(l[6:15]), float(l[15:24]), float(l[24:33])])
            break
    return lines, atom_idx, xyz, box


def ident(line):
    """Atom name, residue name and segment ID: what must match between files."""
    return line[12:16].strip(), line[17:21].strip(), line[72:76].strip()


def is_unset(line):
    """psfgen marks atoms without coordinates with occupancy -1.00 (and 0,0,0)."""
    occ = line[54:60].strip()
    xyz_zero = all(float(line[a:b]) == 0.0 for a, b in ((30, 38), (38, 46), (46, 54)))
    return (occ and float(occ) < 0) or xyz_zero


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solute_pdb", help="PDB of the solute before solvation (e.g. combined.pdb)")
    ap.add_argument("solvated_pdb", help="PDB after solvate/autoionize")
    ap.add_argument("-o", "--output", help="output PDB (default: overwrite solvated_pdb)")
    args = ap.parse_args()

    s_lines, s_idx, s_xyz, _ = read_pdb(args.solute_pdb)
    v_lines, v_idx, v_xyz, box = read_pdb(args.solvated_pdb)
    n = len(s_idx)
    if n > len(v_idx):
        raise SystemExit(f"ERROR: solute has {n} atoms but solvated file only {len(v_idx)}")

    # 1. The first n atoms must be the same atoms, in the same order
    for k in range(n):
        if ident(s_lines[s_idx[k]]) != ident(v_lines[v_idx[k]]):
            raise SystemExit(f"ERROR: atom {k + 1} differs: solute {ident(s_lines[s_idx[k]])} "
                             f"vs solvated {ident(v_lines[v_idx[k]])}. Order not preserved.")

    # 2. Atoms that kept their coordinates must not have moved (catches shifts/reordering)
    unset = np.array([is_unset(v_lines[v_idx[k]]) for k in range(n)])
    kept = ~unset
    if kept.any():
        max_shift = np.abs(s_xyz[kept] - v_xyz[:n][kept]).max()
        if max_shift > 0.002:
            raise SystemExit(f"ERROR: solute atoms moved by up to {max_shift:.3f} A during "
                             "solvation; copying coordinates back would be unsafe.")

    # 3. Copy coordinates back (all solute atoms) and clear the 'unset' flag
    for k in range(n):
        li = v_idx[k]
        line = v_lines[li]
        x, y, z = s_xyz[k]
        occ = "  1.00" if unset[k] else line[54:60]
        v_lines[li] = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{occ}{line[60:]}"

    out = Path(args.output or args.solvated_pdb)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(v_lines))
    tmp.replace(out)
    print(f"Restored coordinates for {unset.sum()} solute atoms (of {n}); wrote {out}")

    # 4. Report how close the restored atoms now are to solvent/ions
    if unset.any():
        fixed = s_xyz[unset]
        others = v_xyz[n:]
        d = fixed[:, None, :] - others[None, :, :]
        if box is not None:
            d -= box * np.round(d / box)          # nearest periodic image
        dmin = np.sqrt((d ** 2).sum(-1)).min()
        print(f"Closest solvent/ion atom to a restored atom: {dmin:.2f} A"
              + ("  (below 1.0 A: real overlap)" if dmin < 1.0 else
                 "  (minimization will relax this)" if dmin < 2.0 else ""))


if __name__ == "__main__":
    main()
