#!/bin/bash
set -euo pipefail

PEP_PDB=$1
PEP_ID=$2
wDIR=${3:-"outputs/"}

LPS_PSF="lps_modeler.psf"
LPS_PDB="lps_modeler.pdb"

SCRIPT_DIR="bash_scripts/"

# make protein psf
# outputs peptide.pdb peptide.psf
echo "making peptide.psf and peptide.pdb.."
bash "${SCRIPT_DIR}psfgen_protein.sh" "${PEP_PDB}"

# combine peptide and LPS 
# place peptide 1.2nm away from LPS
# outputs combined.psf combined.pdb
echo "combining peptide and LPS, and placing peptide near LPS.."
bash "${SCRIPT_DIR}psfgen_and_move_with_vmd.sh" "$LPS_PSF" "$LPS_PDB" "peptide.psf" "peptide.pdb" "$wDIR/combined"

# solvate+ionize
# outputs solvated_ionized.psf and solvated_ionized.pdb
echo "add water molecules and ionize to 0.15 M"
bash "${SCRIPT_DIR}solvate_with_vmd.sh" "$wDIR"

echo "Done combining and solvating ${PEP_PDB}"


