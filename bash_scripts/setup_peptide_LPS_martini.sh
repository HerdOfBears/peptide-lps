#!/bin/bash
# Build a coarse-grained box of dispersed LPS (R1 core) + antimicrobial peptide,
# solvated with Martini water and neutralised with Ca2+ / NaCl.
#
# Prerequisites in $OUT_GRO_DIR:
#   martini_v300/        (force field directory)
#   LPSR1_noW.top        (topology with LPS includes, no solvent)
#   LPSR1_single.gro     (single LPS molecule)
#   peptide_aa.pdb       (atomistic peptide)
# Plus ions.mdp in the working directory.
#
# Usage: ./build_system.sh <secondary_structure> <output.gro>

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
PEP_SECONDARY_STRUCTURE=$1
OUT_GRO=$2

OUT_GRO_DIR=$(dirname "$OUT_GRO")

LPS_SINGLE="$OUT_GRO_DIR/LPSR1_single.gro"
PROTEIN_PDB="$OUT_GRO_DIR/peptide_aa.pdb"
PROTEIN_CG_PDB="$OUT_GRO_DIR/peptide_cg.pdb"
PROTEIN_ONLY_TOP="$OUT_GRO_DIR/peptide.top"

N_LPS_MOL=32
N_PEP_MOL=16
BOX_L=15                 # nm
SALT_CONCENTRATION=0.15  # mol/L

# ---------------------------------------------------------------------------
# Pre-requisite actions
# ---------------------------------------------------------------------------
cp -r data/martini_v300 "$OUT_GRO_DIR"
cp data/LPSR1_noW.top "$OUT_GRO_DIR"
cp data/LPSR1_single.gro "$OUT_GRO_DIR"
cp data/LPSR1.itp "$OUT_GRO_DIR"

# Work on a copy so LPSR1_noW.top stays pristine across reruns
SYSTEM_TOP="$OUT_GRO_DIR/LPSR1.top"
cp "$OUT_GRO_DIR/LPSR1_noW.top" "$SYSTEM_TOP"

# Martini bead radius for insertion overlap checks. GROMACS falls back to
# 0.105 nm for unknown bead names, which is far too small and produces
# overlapping beads that no minimiser can fix.
BEAD_RADIUS=0.23

# martinize2 settings (override by exporting before running this script)
# SECONDARY_STRUCTURE=${SECONDARY_STRUCTURE:?set SECONDARY_STRUCTURE, e.g. HHHHHHHHCCC}
SECONDARY_STRUCTURE=${PEP_SECONDARY_STRUCTURE}
EN_lower=${EN_lower:-0}
EN_upper=${EN_upper:-0.85}

# ---------------------------------------------------------------------------
# Stage 0: coarse-grain the peptide with martinize2
# Writes peptide_cg.pdb, peptide.top and molecule_*.itp into $OUT_GRO_DIR.
# ---------------------------------------------------------------------------
(
  cd "$OUT_GRO_DIR"
  martinize2 \
    -f "$(basename "$PROTEIN_PDB")" \
    -x "$(basename "$PROTEIN_CG_PDB")" \
    -o "$(basename "$PROTEIN_ONLY_TOP")" \
    -ff martini3001 -p backbone -ss "$SECONDARY_STRUCTURE" \
    -elastic -el "$EN_lower" -eu "$EN_upper" -noscfix -ignh
)

# ---------------------------------------------------------------------------
# Stage 1: create the box and insert LPS molecules at random positions
# ---------------------------------------------------------------------------
gmx insert-molecules \
  -box $BOX_L $BOX_L $BOX_L \
  -ci "$LPS_SINGLE" \
  -nmol $N_LPS_MOL \
  -radius $BEAD_RADIUS \
  -rot xyz \
  -o "$OUT_GRO_DIR/step1.gro"

sed -i -E "s/^(LPSR1[[:space:]]+)[0-9]+/\1$N_LPS_MOL/" "$SYSTEM_TOP"
# ---------------------------------------------------------------------------
# Stage 2: insert peptides into the same box, avoiding the LPS already there
# ---------------------------------------------------------------------------
gmx insert-molecules \
  -f "$OUT_GRO_DIR/step1.gro" \
  -ci "$PROTEIN_CG_PDB" \
  -nmol $N_PEP_MOL \
  -radius $BEAD_RADIUS \
  -rot xyz \
  -o "$OUT_GRO"

# ---------------------------------------------------------------------------
# Stage 3: add the peptide to the topology
# The #include goes before [ system ]; the molecule count is appended to
# [ molecules ] so the order matches the .gro: LPS, then peptide.
# ---------------------------------------------------------------------------
for itp in "$OUT_GRO_DIR"/molecule_*.itp; do
  sed -i "/^\[ *system *\]/i #include \"$(basename "$itp")\"" "$SYSTEM_TOP"
done

PEP_NAME=$(awk '/^\[ *molecules *\]/{f=1;next} f&&NF&&$1!~/^;/{print $1; exit}' \
  "$PROTEIN_ONLY_TOP")
# echo "$PEP_NAME  $N_PEP_MOL" >> "$SYSTEM_TOP"
printf "\n%s  %s\n" "$PEP_NAME" "$N_PEP_MOL" >> "$SYSTEM_TOP"
# ---------------------------------------------------------------------------
# Stage 4: solvate with Martini water using insane (lattice water, no salt)
# Ions are added later with genion so names and counts stay consistent.
# ---------------------------------------------------------------------------
insane -f "$OUT_GRO" \
  -o "$OUT_GRO_DIR/solvated.gro" \
  -p "$OUT_GRO_DIR/insane.top" \
  -pbc cubic -x $BOX_L -y $BOX_L -z $BOX_L \
  -sol W -center -d 0

# Take only the water line; insane's own Protein/lipid lines are wrong here
grep -E '^W[[:space:]]' "$OUT_GRO_DIR/insane.top" >> "$SYSTEM_TOP"

sed -i 's|#include "martini_|#include "martini_v300/martini_|g' "$SYSTEM_TOP"

# ---------------------------------------------------------------------------
# Stage 5: neutralise the LPS charge with Ca2+
# ---------------------------------------------------------------------------
gmx grompp -f ions.mdp \
  -c "$OUT_GRO_DIR/solvated.gro" \
  -p "$SYSTEM_TOP" \
  -o "$OUT_GRO_DIR/ions1.tpr" \
  -maxwarn 2

echo W | gmx genion \
  -s "$OUT_GRO_DIR/ions1.tpr" \
  -o "$OUT_GRO_DIR/ions1.gro" \
  -p "$SYSTEM_TOP" \
  -pname CA -pq 2 \
  -neutral

# ---------------------------------------------------------------------------
# Stage 6: add NaCl at the target concentration
# ---------------------------------------------------------------------------
gmx grompp -f ions.mdp \
  -c "$OUT_GRO_DIR/ions1.gro" \
  -p "$SYSTEM_TOP" \
  -o "$OUT_GRO_DIR/ions2.tpr" \
  -maxwarn 2

echo W | gmx genion \
  -s "$OUT_GRO_DIR/ions2.tpr" \
  -o "$OUT_GRO_DIR/final.gro" \
  -p "$SYSTEM_TOP" \
  -pname NA -nname CL \
  -conc $SALT_CONCENTRATION -neutral

# ---------------------------------------------------------------------------
# Stage 7: verify. No -maxwarn here, so a clean run means the system is
# neutral and the topology matches the coordinates.
# ---------------------------------------------------------------------------
gmx grompp -f ions.mdp \
  -c "$OUT_GRO_DIR/final.gro" \
  -p "$SYSTEM_TOP" \
  -o "$OUT_GRO_DIR/check.tpr"

# ---------------------------------------------------------------------------
# Clean up intermediates
# ---------------------------------------------------------------------------
rm -f "$OUT_GRO_DIR/ions1.gro" \
      "$OUT_GRO_DIR/ions1.tpr" \
      "$OUT_GRO_DIR/ions2.tpr" \
      "$OUT_GRO_DIR/step1.gro" \
      "$OUT_GRO_DIR/solvated.gro" \
      "$OUT_GRO_DIR/insane.top"
rm -f "$OUT_GRO_DIR"/#*

echo "Built $OUT_GRO_DIR/final.gro with $SYSTEM_TOP"