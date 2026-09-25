import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Standard 20 amino acids (one-letter codes).
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence: str) -> str:
    """Validate and normalize the peptide sequence."""
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Peptide sequence is empty.")
    bad = sorted(set(seq) - VALID_AA)
    if bad:
        raise ValueError(
            f"Sequence contains non-standard amino acid characters: {bad}. "
            "Only the 20 standard one-letter codes are supported by this script."
        )
    return seq

def write_yaml_input(sequence: str, pep_id: int, yaml_path: Path) -> None:
    """
    Write a Boltz-2 YAML input file for a single peptide chain in
    single-sequence (no-MSA) mode.

    Setting `msa: empty` tells Boltz-2 to skip MSA usage entirely.
    """
    yaml_content = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}\n"
        "      msa: empty\n"
    )
    yaml_path.write_text(yaml_content)


def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    accelerator: str = "cpu",
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    output_format: str = "mmcif",
    override: bool = True,
) -> None:
    """Invoke the `boltz predict` CLI."""
    cmd = [
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--accelerator", accelerator,
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--output_format", output_format,
        "--no_kernels"
    ]
    if override:
        cmd.append("--override")

    # Make sure boltz is available on PATH.
    if shutil.which("boltz") is None:
        raise FileNotFoundError(
            "The `boltz` executable was not found on PATH. "
            "Install it with `pip install boltz -U` and try again."
        )

    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"`boltz predict` exited with code {result.returncode}. "
            "See the log above for details."
        )
