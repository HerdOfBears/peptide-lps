#!/bin/bash
set -euo pipefail

oDIR=$1
system_name="combined"
PSF=$oDIR/"${system_name}.psf"
PDB=$oDIR/"${system_name}.pdb"

PAD_A=10      # Å padding
SALT_M=0.15   # mol/L

SOLV_PREFIX="solvated"
ION_PREFIX="solvated_ionized"

OUT_PSF="${ION_PREFIX}.psf"
OUT_PDB="${ION_PREFIX}.pdb"
OUT_CRD="${ION_PREFIX}.crd"

vmd -dispdev text -eofexit << EOF
package require solvate
package require autoionize

#################################################
# load the system
#################################################
# set mid [mol new "${PSF}" type psf waitfor all]
# mol addfile "${PDB}" type pdb waitfor all molid \$mid

#################################################
# measure COM-COM distance between peptide and ECLIPA
#################################################
mol load psf {${PSF}} pdb {${PDB}}

set lipa_sel [atomselect top "resname ECLIPA"]
set lipa_com [measure center \$lipa_sel weight mass]

set pep_sel [atomselect top "segid PEP"]
set pep_com [measure center \$pep_sel weight mass]

# compute the distance vector and its magnitude
set dx [expr {[lindex \$pep_com 0] - [lindex \$lipa_com 0]}]
set dy [expr {[lindex \$pep_com 1] - [lindex \$lipa_com 1]}]
set dz [expr {[lindex \$pep_com 2] - [lindex \$lipa_com 2]}]

set dist_lipa_pep [expr {sqrt(\$dx*\$dx + \$dy*\$dy + \$dz*\$dz)}]

puts "Initial peptide-lipid A COM distance: [format %.2f \$dist_lipa_pep] A"
#################################################
# compute box dimensions for solvation
#################################################
# set sel [atomselect \$mid all]
set sel [atomselect top all]
set mm  [measure minmax \$sel]          ;# {{xmin ymin zmin} {xmax ymax zmax}}
set mn  [lindex \$mm 0]
set mx  [lindex \$mm 1]

set xmin [lindex \$mn 0]; set ymin [lindex \$mn 1]; set zmin [lindex \$mn 2]
set xmax [lindex \$mx 0]; set ymax [lindex \$mx 1]; set zmax [lindex \$mx 2]

set dx [expr {\$xmax - \$xmin}]
set dy [expr {\$ymax - \$ymin}]
set dz [expr {\$zmax - \$zmin}]

# max(x,y) extent
set dxy [expr {(\$dx > \$dy) ? \$dx : \$dy}]

# (pad on EACH SIDE):
set Lxy [expr {\$dxy + 2.0*${PAD_A}}]
set Lz  [expr {\$dz  + 2.0*${PAD_A}}]

# Center on solute
set cx [expr {0.5*(\$xmin + \$xmax)}]
set cy [expr {0.5*(\$ymin + \$ymax)}]
set cz [expr {0.5*(\$zmin + \$zmax)}]

set bxmin [expr {\$cx - 0.5*\$Lxy}]
set bymin [expr {\$cy - 0.5*\$Lxy}]
set bzmin [expr {\$cz - 0.5*\$Lz}]
set bxmax [expr {\$cx + 0.5*\$Lxy}]
set bzmax [expr {\$cz + 0.5*\$Lz}]

# extend box in y 2.0x pep-lipa COM distance, 
# to ensure sufficient space for SMD pulling
# 2.0x allows for doubling the COM-COM distance
# during pulling, while still keeping the peptide from 
# the periodic image of the lipid A COM.
set bymax [expr {\$cy + 0.5*\$Lxy + 2.0*\$dist_lipa_pep}]


set minmax [list [list \$bxmin \$bymin \$bzmin] [list \$bxmax \$bymax \$bzmax]]
puts "Measured minmax: \$mm"
puts "Solvate minmax:  \$minmax"

# Clean up *this* molecule safely
\$sel delete
# mol delete \$mid

# --- Solvate with explicit dimensions ---
solvate ${PSF} ${PDB} -minmax \$minmax -o ${SOLV_PREFIX}

# Ionize (IMPORTANT: set output prefix)

# autoionize -psf ${SOLV_PREFIX}.psf -pdb ${SOLV_PREFIX}.pdb -neutralize -salt ${SALT_M} -o ${ION_PREFIX}
autoionize -psf ${SOLV_PREFIX}.psf -pdb ${SOLV_PREFIX}.pdb -sc ${SALT_M} -o ${ION_PREFIX}

# Write CRD (if you really want it)
mol new ${ION_PREFIX}.psf
mol addfile ${ION_PREFIX}.pdb
set sel [atomselect top all]
\$sel writecrd ${ION_PREFIX}.crd

quit
EOF


NATOM=$(grep -c "^ATOM\|^HETATM" solvated_ionized.pdb)

cat > solvated_ionized_fixed.crd << EOF
* Generated from solvated_ionized.pdb
*
    ${NATOM}  EXT
EOF
cat solvated_ionized.crd >> solvated_ionized_fixed.crd

echo "Wrote:"
ls -lh "${OUT_PSF}" "${OUT_PDB}" "${OUT_CRD}"

echo "moving to ${oDIR}"
mv "${OUT_PSF}" "${oDIR}"
mv "${OUT_PDB}" "${oDIR}"
mv "${OUT_CRD}" "${oDIR}"
