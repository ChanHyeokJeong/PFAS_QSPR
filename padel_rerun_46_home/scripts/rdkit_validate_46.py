from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


def validate_smiles(smiles: str) -> dict:
    result = {
        "rdkit_valid": False,
        "rdkit_canonical_smiles": "",
        "rdkit_error": "",
        "rdkit_atoms": 0,
        "rdkit_heavy_atoms": 0,
        "rdkit_rings": 0,
    }
    smiles = (smiles or "").strip()
    if not smiles:
        result["rdkit_error"] = "empty SMILES"
        return result

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["rdkit_error"] = "MolFromSmiles returned None"
            return result
        result.update(
            {
                "rdkit_valid": True,
                "rdkit_canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "rdkit_atoms": mol.GetNumAtoms(),
                "rdkit_heavy_atoms": mol.GetNumHeavyAtoms(),
                "rdkit_rings": mol.GetRingInfo().NumRings(),
            }
        )
    except Exception as exc:
        result["rdkit_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the 46 all-PaDEL-missing PFAS SMILES with RDKit.")
    parser.add_argument("--input", type=Path, default=Path("data/all_descriptor_missing_46.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rdkit_validation_46.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    for row in rows:
        profile = validate_smiles(row.get("smiles", ""))
        output_rows.append({**row, **profile})

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    valid_count = sum(1 for row in output_rows if row["rdkit_valid"])
    print(f"RDKit valid: {valid_count}/{len(output_rows)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
