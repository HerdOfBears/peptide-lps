#!/bin/bash 
#
PEP_PDB=$1
FFfiles_dir="ff_files"

TCL_SCRIPT=$(mktemp /tmp/psfgen_merge_XXXXXX.tcl)

cat > "$TCL_SCRIPT" << EOF
#
package require psfgen
resetpsf

# Point to your CHARMM36 topology file
topology $FFfiles_dir/toppar/top_all36_prot.rtf
topology $FFfiles_dir/toppar/toppar_water_ions.str

# Build segment from PDB
segment PEP {
    pdb $PEP_PDB
}

# Read coordinates
coordpdb $PEP_PDB PEP

# Guess any missing coordinates
guesscoord

writepsf peptide.psf
writepdb peptide.pdb
exit
EOF

echo "Running VMD psfgen..."
vmd -dispdev none -e "$TCL_SCRIPT"
EXIT_CODE=$?

rm -f "$TCL_SCRIPT"


