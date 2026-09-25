#!/bin/bash
# psfgen_and_move_with_vmd.sh
# Usage: ./psfgen_and_move_with_vmd.sh <lps.psf> <lps.pdb> <peptide.psf> <peptide.pdb> [output_prefix]

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <lps.psf> <lps.pdb> <peptide.psf> <peptide.pdb> [output_prefix]"
    exit 1
fi

LPS_PSF="$1"
LPS_PDB="$2"
PEP_PSF="$3"
PEP_PDB="$4"
OUT="${5:-combined}"
GAP=5.4 # + 3.4 for vdw

for f in "$LPS_PSF" "$LPS_PDB" "$PEP_PSF" "$PEP_PDB"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

echo "Merging:"
echo "  LPS:     $LPS_PSF / $LPS_PDB"
echo "  Peptide: $PEP_PSF / $PEP_PDB"
echo "  Output:  ${OUT}.psf / ${OUT}.pdb"

# Write Tcl script with bash variables substituted
TCL_SCRIPT=$(mktemp /tmp/psfgen_merge_XXXXXX.tcl)

cat > "$TCL_SCRIPT" << EOF
# --- Step 1: merge PSFs with psfgen ---
package require psfgen
resetpsf

readpsf  {${LPS_PSF}}
coordpdb {${LPS_PDB}}

set all [atomselect top all]
set zero_atoms [atomselect top "x == 0 and y == 0 and z == 0"]
puts "Atoms with zero coordinates: [\$zero_atoms num]"
\$zero_atoms delete
\$all delete

guesscoord

readpsf  {${PEP_PSF}}
coordpdb {${PEP_PDB}}
guesscoord

writepsf {${OUT}.psf}
writepdb {${OUT}.pdb}

puts "psfgen merge complete."

# --- Step 2: reposition peptide above LPS using VMD atomselect ---
################################################
################################################
# --- Lipid A: get max y and COM ---
mol load psf {${OUT}.psf} pdb {${OUT}.pdb}

# set lipa_sel [atomselect top "resname ECLIPA"]
# if { [\$lipa_sel num] == 0 } {
#     puts "ERROR: No atoms found with resname ECLIPA. Check residue name."
#     exit 1
# }
###############New below
set gap ${GAP}
set pep [atomselect top "segid PEP"]
set lps [atomselect top "resname ECLIPA"]

# 1. line up centers of mass in x and z
set pc [measure center \$pep weight mass]
set lc [measure center \$lps weight mass]
\$pep moveby [list [expr {[lindex \$lc 0] - [lindex \$pc 0]}] 0 \
                  [expr {[lindex \$lc 2] - [lindex \$pc 2]}]]

# 2. bracket: y_lo = buried (guaranteed contact), y_hi = bounding-box answer (guaranteed clear)
set y_lo [expr {[lindex \$lc 1] - [lindex \$pc 1]}]
set y_hi [expr {[lindex [measure minmax \$lps] 1 1] \
               - [lindex [measure minmax \$pep] 0 1] + \$gap + 1.0}]
\$pep moveby [list 0 \$y_hi 0]
set cur \$y_hi

# 3. bisect until the peptide just stops being within \$gap of the LPS
set probe [atomselect top "segid PEP and within \$gap of (not segid PEP)"]
while {\$y_hi - \$y_lo > 0.01} {
    set mid [expr {(\$y_lo + \$y_hi) / 2.0}]
    \$pep moveby [list 0 [expr {\$mid - \$cur}] 0]
    set cur \$mid
    \$probe update
    if {[\$probe num] > 0} { set y_lo \$mid } else { set y_hi \$mid }
}
# land on the clear side
\$pep moveby [list 0 [expr {\$y_hi - \$cur}] 0]

puts "Vertical offset applied: [format %.2f \$y_hi] A"
puts "Closest approach is now \$gap A (to within 0.01 A)"
\$probe delete
###############New above
# set lipa_y_list [\$lipa_sel get y]
# set lipa_max_y [lindex [lsort -real \$lipa_y_list] end]
# set lipa_com [measure center \$lipa_sel weight mass]
# set lipa_com_x [lindex \$lipa_com 0]
# set lipa_com_y [lindex \$lipa_com 1]
# set lipa_com_z [lindex \$lipa_com 2]
# puts "Lipid A max Y:   [format %.2f \$lipa_max_y] A"
# puts "Lipid A COM:     [format "%.2f %.2f %.2f" \$lipa_com_x \$lipa_com_y \$lipa_com_z] A"

# # --- Peptide: get min y and COM ---
# set pep_sel [atomselect top "segid PEP"]
# if { [\$pep_sel num] == 0 } {
#     puts "ERROR: No atoms found with segid PEP. Check segment IDs."
#     exit 1
# }
# set pep_y_list [\$pep_sel get y]
# set pep_min_y [lindex [lsort -real \$pep_y_list] 0]
# set pep_com [measure center \$pep_sel weight mass]
# set pep_com_x [lindex \$pep_com 0]
# set pep_com_y [lindex \$pep_com 1]
# set pep_com_z [lindex \$pep_com 2]
# puts "Peptide min Y:   [format %.2f \$pep_min_y] A"
# puts "Peptide COM:     [format "%.2f %.2f %.2f" \$pep_com_x \$pep_com_y \$pep_com_z] A"

# # --- Compute translation ---
# # x, z: line up peptide COM with lipid A COM
# # y: place peptide so its lowest atom is 10 A above lipid A's highest atom
# set gap ${GAP}
# set dx [expr {\$lipa_com_x - \$pep_com_x}]
# set dz [expr {\$lipa_com_z - \$pep_com_z}]
# set dy [expr {\$lipa_max_y + \$gap - \$pep_min_y}]
# puts "Translation:     dx=[format %.2f \$dx]  dy=[format %.2f \$dy]  dz=[format %.2f \$dz] A"

# # --- Apply ---
# \$pep_sel moveby [list \$dx \$dy \$dz]

# --- Verify ---
set pep_com_after [measure center \$pep_sel weight mass]
set pep_y_after [\$pep_sel get y]
puts "After move:"
puts "  Peptide COM:   [format "%.2f %.2f %.2f" \
    [lindex \$pep_com_after 0] [lindex \$pep_com_after 1] [lindex \$pep_com_after 2]] A"
puts "  Peptide min Y: [format %.2f [lindex [lsort -real \$pep_y_after] 0]] A"
puts "  Gap above LPS: [format %.2f \
    [expr {[lindex [lsort -real \$pep_y_after] 0] - \$lipa_max_y}]] A"

################################################
################################################
# Shift peptide above LPS
#set gap 10.0
#set shift_z [expr {\$max_z + \$gap - \$pep_z}]
#\$pep_sel moveby [list 0 \$shift_z -5]
#puts "Shifted peptide by [format %.2f \$shift_z] Angstroms in Z"

# Write final output
set all_sel [atomselect top all]
\$all_sel writepdb {${OUT}.pdb}

\$lipa_sel delete
#\$lps_sel delete
\$pep_sel delete
\$all_sel delete

puts "Done. Final output: ${OUT}.psf / ${OUT}.pdb"
exit
EOF

echo "Running VMD..."
vmd -dispdev none -e "$TCL_SCRIPT"
EXIT_CODE=$?

rm -f "$TCL_SCRIPT"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: VMD exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

if [ ! -f "${OUT}.psf" ] || [ ! -f "${OUT}.pdb" ]; then
    echo "ERROR: Output files not created."
    exit 1
fi

echo ""
echo "Success!"
echo "  ${OUT}.psf  ($(wc -l < "${OUT}.psf") lines)"
echo "  ${OUT}.pdb  ($(wc -l < "${OUT}.pdb") lines)"
