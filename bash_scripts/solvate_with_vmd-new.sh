#!/bin/bash
# solvate_ionize.sh
# Solvate and ionize <oDIR>/combined.psf/.pdb with VMD, writing a box that is
# safe for periodic boundary conditions.
#
# Usage: ./solvate_ionize.sh <oDIR>
set -euo pipefail

oDIR=$1
system_name="combined"
PSF="${oDIR}/${system_name}.psf"
PDB="${oDIR}/${system_name}.pdb"

PAD_A=10        # A padding on each side of the solute
SALT_M=0.15     # mol/L
WATER_GAP=2.0   # A of the box left without water (split over both faces), so
                # water on opposite faces doesn't overlap across the boundary

SOLV="${oDIR}/solvated"
ION="${oDIR}/solvated_ionized"
LOG="${oDIR}/solvate_vmd.log"

for f in "$PSF" "$PDB"; do
    [ -f "$f" ] || { echo "ERROR: File not found: $f"; exit 1; }
done

# Stop early if the input already has atoms without coordinates
bad_atoms() {   # count atoms at (0,0,0) or with occupancy < 0 (psfgen's "unset" flag)
    awk '/^(ATOM|HETATM)/ {
            x=substr($0,31,8)+0; y=substr($0,39,8)+0; z=substr($0,47,8)+0; occ=substr($0,55,6)+0
            if ((x==0 && y==0 && z==0) || occ<0) n++ }
         END {print n+0}' "$1"
}
n=$(bad_atoms "$PDB")
[ "$n" -eq 0 ] || { echo "ERROR: $n atoms in $PDB have no coordinates; fix the merge step first"; exit 1; }

vmd -dispdev text -eofexit << EOF 2>&1 | tee "$LOG"
package require solvate
package require autoionize

if {[catch {

    mol new {${PSF}} type psf waitfor all
    mol addfile {${PDB}} type pdb waitfor all

    # --- peptide - lipid A COM distance (used to lengthen the box in y) ---
    set lipa_com [measure center [atomselect top "resname ECLIPA"] weight mass]
    set pep_com  [measure center [atomselect top "segid PEP"] weight mass]
    set dist_lipa_pep [vecdist \$pep_com \$lipa_com]
    puts "Initial peptide-lipid A COM distance: [format %.2f \$dist_lipa_pep] A"

    # --- box dimensions ---
    set mm [measure minmax [atomselect top all]]
    lassign [lindex \$mm 0] xmin ymin zmin
    lassign [lindex \$mm 1] xmax ymax zmax

    set dxy [expr {max(\$xmax - \$xmin, \$ymax - \$ymin)}]
    set Lxy [expr {\$dxy + 2.0*${PAD_A}}]
    set Lz  [expr {(\$zmax - \$zmin) + 2.0*${PAD_A}}]

    # centre on the solute
    set cx [expr {0.5*(\$xmin + \$xmax)}]
    set cy [expr {0.5*(\$ymin + \$ymax)}]
    set cz [expr {0.5*(\$zmin + \$zmax)}]

    set bxmin [expr {\$cx - 0.5*\$Lxy}];  set bxmax [expr {\$cx + 0.5*\$Lxy}]
    set bzmin [expr {\$cz - 0.5*\$Lz}];   set bzmax [expr {\$cz + 0.5*\$Lz}]
    set bymin [expr {\$cy - 0.5*\$Lxy}]
    # extend +y by 2x the COM-COM distance so the peptide can be pulled away
    # (SMD) without meeting the periodic image of the LPS
    set bymax [expr {\$cy + 0.5*\$Lxy + 2.0*\$dist_lipa_pep}]

    # This is the periodic box we want
    set Lx [expr {\$bxmax - \$bxmin}]
    set Ly [expr {\$bymax - \$bymin}]
    set Lz [expr {\$bzmax - \$bzmin}]

    # Fill a slightly smaller region with water: WATER_GAP/2 in from every face.
    # Without this, water touching opposite faces overlaps through the
    # periodic boundary (seen as <1 A water-water contacts).
    set g [expr {${WATER_GAP}/2.0}]
    set solv_minmax [list [list [expr {\$bxmin+\$g}] [expr {\$bymin+\$g}] [expr {\$bzmin+\$g}]] \
                          [list [expr {\$bxmax-\$g}] [expr {\$bymax-\$g}] [expr {\$bzmax-\$g}]]]
    puts "Solute minmax:   \$mm"
    puts "Water region:    \$solv_minmax"
    puts "Periodic box:    [format {%.3f %.3f %.3f} \$Lx \$Ly \$Lz] A"

    mol delete top

    # --- solvate and ionize ---
    solvate {${PSF}} {${PDB}} -minmax \$solv_minmax -o {${SOLV}}
    autoionize -psf {${SOLV}.psf} -pdb {${SOLV}.pdb} -sc ${SALT_M} -o {${ION}}

    # --- write the periodic box into the final PDB (CRYST1) and box.txt ---
    mol new {${ION}.psf} type psf waitfor all
    mol addfile {${ION}.pdb} type pdb waitfor all
    molinfo top set {a b c alpha beta gamma} [list \$Lx \$Ly \$Lz 90 90 90]
    [atomselect top all] writepdb {${ION}.pdb}
    set fh [open {${oDIR}/box.txt} w]
    puts \$fh [format {%.4f %.4f %.4f} \$Lx \$Ly \$Lz]
    close \$fh

    puts "SOLVATE_OK"
} err]} {
    puts "ERROR: \$err"
}
quit
EOF

grep -q "SOLVATE_OK" "$LOG" || { echo "ERROR: solvation failed - see $LOG"; exit 1; }

# Check that no atom lost its coordinates during solvate/autoionize
for f in "${SOLV}.pdb" "${ION}.pdb"; do
    n=$(bad_atoms "$f")
    [ "$n" -eq 0 ] || { echo "ERROR: $n atoms in $f have no coordinates"; exit 1; }
done

# Optional: a proper CHARMM CRD file, written by ParmEd from the PSF + PDB
python -c "
import parmed as pmd
s = pmd.load_file('${ION}.psf', xyz='${ION}.pdb')
s.save('${ION}.crd', format='charmmcrd', overwrite=True)
"

echo "Wrote:"
ls -lh "${ION}.psf" "${ION}.pdb" "${ION}.crd" "${oDIR}/box.txt"
echo "Periodic box (A): $(cat "${oDIR}/box.txt")"