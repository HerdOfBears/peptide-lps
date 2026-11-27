"""
ESMFold2 structure prediction helper.

The ESMFold2 counterpart of `run_boltz_predict`. Unlike Boltz, ESMFold2 has a
Python API, so no subprocess is needed: the model is loaded once and called
directly, which is much faster inside an iterative fold -> MD loop.
"""

import json
import string
from functools import lru_cache
from pathlib import Path

import torch
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str):
    """
    Load ESMFold2 once and keep it in memory.

    Loading means reading several GB of weights, so it is slow. lru_cache
    remembers the result: the first call loads the model, and every later call
    with the same (model_name, device) returns the already-loaded model.
    """
    return EsmFold2Model.from_pretrained(model_name, device=device).eval()


def _read_fasta(fasta_path: Path) -> list[str]:
    """Return the sequences in a FASTA file, one string per '>' record."""
    sequences, current = [], []
    for line in fasta_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue                      # skip blank lines
        if line.startswith(">"):          # header: start a new record
            if current:
                sequences.append("".join(current))
                current = []
        else:                             # sequence line (may be wrapped)
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def run_esmfold2_predict(
    fasta_path: Path,
    out_dir: Path,
    device: str = "cuda",
    model_name: str = "biohub/ESMFold2",
    num_loops: int = 20,
    num_sampling_steps: int = 100,
    num_samples: int = 1,
    seed: int = 0,
    override: bool = True,
) -> list[Path]:
    """
    Predict a structure with ESMFold2 from a FASTA file.

    Each record in the FASTA file becomes one protein chain (A, B, C, ...) of a
    single complex, the same way one Boltz YAML file describes one complex.

    Parameters
    ----------
    fasta_path : Path
        Input FASTA file. One record = one chain.
    out_dir : Path
        Folder for the outputs (created if missing).
    device : str
        "cuda" (default) or "cpu". CPU works but is very slow for this model.
    model_name : str
        "biohub/ESMFold2" (full model) or "biohub/ESMFold2-Fast"
        (single-sequence, faster).
    num_loops, num_sampling_steps : int
        Passed straight to ESMFold2's fold(); defaults are the values from
        Biohub's README example.
    num_samples : int
        How many structures to predict. Each uses a different seed
        (seed, seed+1, ...).
    seed : int
        Random seed for the first sample, for reproducibility.
    override : bool
        If False and all output files already exist, skip the prediction.

    Returns
    -------
    list[Path]
        Paths of the predicted mmCIF files, one per sample. Next to each
        <name>_sampleN.cif is a <name>_sampleN.json with its confidence scores.
    """
    fasta_path = Path(fasta_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output names are based on the input file name, e.g. myprot_sample0.cif
    name = fasta_path.stem
    cif_paths = [out_dir / f"{name}_sample{i}.cif" for i in range(num_samples)]

    # Skip the work if the results are already there and we were told not to redo it
    if not override and all(p.exists() for p in cif_paths):
        print(f"[esmfold2] {name}: outputs already exist, skipping")
        return cif_paths

    # --- Build the model input: one ProteinInput per FASTA record ---
    sequences = _read_fasta(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    if len(sequences) > len(string.ascii_uppercase):
        raise ValueError("More than 26 chains; chain IDs A-Z would run out")

    chains = [
        ProteinInput(id=chain_id, sequence=seq.upper())
        for chain_id, seq in zip(string.ascii_uppercase, sequences)
    ]
    spi = StructurePredictionInput(sequences=chains)

    # --- Load the model (only slow the first time; cached afterwards) ---
    model = _load_model(model_name, device)
    builder = ESMFold2InputBuilder()

    # --- Predict each sample with its own seed and write the results ---
    for i, cif_path in enumerate(cif_paths):
        with torch.inference_mode():          # no gradients needed: saves memory
            result = builder.fold(
                model,
                spi,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1,
                seed=seed + i,
            )

        # Structure file
        cif_path.write_text(result.complex.to_mmcif())

        # Confidence scores. ipTM only means something for multi-chain complexes.
        scores = {
            "plddt_mean": float(result.plddt.mean()),
            "ptm": float(result.ptm),
        }
        if len(chains) > 1:
            scores["iptm"] = float(result.iptm)
        cif_path.with_suffix(".json").write_text(json.dumps(scores, indent=2))

        print(f"[esmfold2] wrote {cif_path.name}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in scores.items()))

    # Hand unused GPU memory back so the following OpenMM run has room
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return cif_paths


if __name__ == "__main__":
    # Example: python esmfold2_predict.py my_protein.fasta predictions/
    import sys
    run_esmfold2_predict(Path(sys.argv[1]), Path(sys.argv[2]))