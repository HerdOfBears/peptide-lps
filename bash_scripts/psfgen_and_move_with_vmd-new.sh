#!/bin/bash
# psfgen_and_move_with_vmd.sh
# Merge an LPS and a peptide (PSF+PDB each) and place the peptide above the LPS
# with a fixed closest-approach gap.
#
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
GAP=5.4   # closest allowed peptide-LPS atom distance (A)

for f in "$LPS_PSF" "$LPS_PDB" "$PEP_PSF" "$PEP_PDB"; do
    [ -f "$f" ] || { echo "ERROR: File not found: $f"; exit 1; }
done

echo "Merging:"
echo "  LPS:     $LPS_PSF / $LPS_PDB"
echo "  Peptide: $PEP_PSF / $PEP_PDB"
echo "  Output:  ${OUT}.psf / ${OUT}.pdb"

# Temporary Tcl script, removed on exit even if something fails
TCL_SCRIPT=$(mktemp /tmp/psfgen_merge_XXXXXX.tcl)
LOG="${OUT}_vmd.log"
trap 'rm -f "$TCL_SCRIPT"' EXIT

cat > "$TCL_SCRIPT" << EOF
package require psfgen

# Any error inside this block is caught and reported instead of leaving VMD
# hanging at its prompt. The bash side checks for MERGE_OK in the log.
if {[catch {

    # --- Step 1: merge with psfgen ---------------------------------------
    # 'readpsf ... pdb ...' assigns coordinates by ATOM ORDER, not by name.
    # This matters for LPS: atom names like H810R/H810S have 5 characters,
    # which the PDB format truncates to 4 (both become H810). A name-based
    # 'coordpdb' then cannot match them and leaves them at (0,0,0).
    resetpsf
    readpsf {${LPS_PSF}} pdb {${LPS_PDB}}
    readpsf {${PEP_PSF}} pdb {${PEP_PDB}}
    writepsf {${OUT}.psf}
    writepdb {${OUT}.pdb}
    puts "psfgen merge complete."

    # --- Step 2: load the merged system into VMD -------------------------
    # (psfgen's structure is not a VMD molecule, so load it to select atoms)
    mol new {${OUT}.psf} type psf waitfor all
    mol addfile {${OUT}.pdb} type pdb waitfor all

    set all [atomselect top all]
    set pep [atomselect top "segid PEP"]
    set lps [atomselect top "not segid PEP"]
    if {[\$pep num] == 0} { error "no atoms with segid PEP" }
    if {[\$lps num] == 0} { error "no LPS atoms found" }

    # Refuse to continue if any atom still has no coordinates
    set unset [atomselect top "x == 0 and y == 0 and z == 0"]
    if {[\$unset num] > 0} {
        error "[\$unset num] atoms have no coordinates (at 0,0,0): [lsort -unique [\$unset get name]]"
    }

    # --- Step 3: place the peptide above the LPS (+y) --------------------
    set gap ${GAP}

    # 3a. line up centres of mass in x and z (LPS = whole LPS molecule)
    set pc [measure center \$pep weight mass]
    set lc [measure center \$lps weight mass]
    \$pep moveby [list [expr {[lindex \$lc 0] - [lindex \$pc 0]}] 0 \
                      [expr {[lindex \$lc 2] - [lindex \$pc 2]}]]

    # 3b. bracket the vertical offset:
    #     y_lo = centres overlapping (certainly in contact)
    #     y_hi = peptide's lowest atom above the LPS's highest atom + gap (certainly clear)
    set y_lo [expr {[lindex \$lc 1] - [lindex \$pc 1]}]
    set y_hi [expr {[lindex [measure minmax \$lps] 1 1] \
                   - [lindex [measure minmax \$pep] 0 1] + \$gap + 1.0}]
    \$pep moveby [list 0 \$y_hi 0]
    set cur \$y_hi

    # 3c. bisect until the peptide is just farther than \$gap from the LPS
    set probe [atomselect top "segid PEP and within \$gap of (not segid PEP)"]
    while {\$y_hi - \$y_lo > 0.01} {
        set mid [expr {(\$y_lo + \$y_hi) / 2.0}]
        \$pep moveby [list 0 [expr {\$mid - \$cur}] 0]
        set cur \$mid
        \$probe update
        if {[\$probe num] > 0} { set y_lo \$mid } else { set y_hi \$mid }
    }
    \$pep moveby [list 0 [expr {\$y_hi - \$cur}] 0]   ;# land on the clear side

    # --- Step 4: verify ----------------------------------------------------
    \$probe update
    if {[\$probe num] > 0} { error "peptide still within \$gap A of the LPS after placement" }
    set near [measure contacts [expr {\$gap + 0.1}] \$pep \$lps]
    puts "Vertical offset applied: [format %.2f \$y_hi] A"
    puts "Peptide atoms within [expr {\$gap + 0.1}] A of LPS: [llength [lindex \$near 0]] (should be > 0: touching distance)"
    set pc_after [measure center \$pep weight mass]
    puts "Peptide COM after move: [format {%.2f %.2f %.2f} {*}\$pc_after] A"

    # --- Step 5: write the moved coordinates (PSF is unchanged) ------------
    \$all writepdb {${OUT}.pdb}

    foreach s [list \$all \$pep \$lps \$unset \$probe] { \$s delete }
    puts "MERGE_OK"

} err]} {
    puts "ERROR: \$err"
}
exit
EOF

echo "Running VMD..."
# VMD usually exits with code 0 even after a Tcl error, so judge success from the log
vmd -dispdev none -e "$TCL_SCRIPT" 2>&1 | tee "$LOG"

if ! grep -q "MERGE_OK" "$LOG"; then
    echo "ERROR: merge/placement failed - see $LOG"
    exit 1
fi

# Independent check: psfgen writes occupancy -1.00 for atoms without coordinates
N_UNSET=$(awk '/^(ATOM|HETATM)/ && substr($0,55,6)+0 < 0' "${OUT}.pdb" | wc -l)
if [ "$N_UNSET" -gt 0 ]; then
    echo "ERROR: ${N_UNSET} atoms in ${OUT}.pdb have no coordinates"
    exit 1
fi

echo ""
echo "Success!"
echo "  ${OUT}.psf  ($(wc -l < "${OUT}.psf") lines)"
echo "  ${OUT}.pdb  ($(wc -l < "${OUT}.pdb") lines)"