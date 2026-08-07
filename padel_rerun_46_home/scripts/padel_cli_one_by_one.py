from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


def find_padel_jar(explicit: Path | None) -> Path:
    if explicit:
        jar = explicit
    else:
        import padelpy

        jar = Path(padelpy.__file__).resolve().parent / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar"
    if not jar.exists():
        raise FileNotFoundError(f"PaDEL-Descriptor.jar not found: {jar}")
    return jar


def rdkit_profile(smiles: str) -> dict:
    profile = {
        "rdkit_valid": False,
        "rdkit_canonical_smiles": "",
        "rdkit_error": "",
        "rdkit_atoms": 0,
        "rdkit_heavy_atoms": 0,
        "rdkit_rings": 0,
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            profile["rdkit_error"] = "MolFromSmiles returned None"
            return profile
        profile.update(
            {
                "rdkit_valid": True,
                "rdkit_canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "rdkit_atoms": mol.GetNumAtoms(),
                "rdkit_heavy_atoms": mol.GetNumHeavyAtoms(),
                "rdkit_rings": mol.GetRingInfo().NumRings(),
            }
        )
    except Exception as exc:
        profile["rdkit_error"] = f"{type(exc).__name__}: {exc}"
    return profile


def terminate_specific_padel(proc: subprocess.Popen, unique_token: str) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False)
        escaped = unique_token.replace("\\", "\\\\").replace("'", "''")
        subprocess.run(
            [
                "wmic",
                "process",
                "where",
                f"CommandLine like '%{escaped}%'",
                "call",
                "terminate",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.kill()


def count_descriptor_values(output_csv: Path) -> tuple[int, int]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return 0, 0
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return 0, 0
    row = rows[0]
    descriptor_keys = [key for key in row.keys() if key and key.lower() != "name"]
    nonmissing = 0
    for key in descriptor_keys:
        value = (row.get(key) or "").strip()
        if value and value.lower() not in {"na", "n/a", "nan", "null", "none"}:
            nonmissing += 1
    return len(descriptor_keys), nonmissing


def run_one(row: dict, args: argparse.Namespace, jar: Path, java_exe: str) -> dict:
    compound_id = row["compound_id"]
    smiles = row["smiles"].strip()
    profile = rdkit_profile(smiles)

    stamp = time.strftime("%Y%m%d%H%M%S")
    unique = f"{compound_id}_{stamp}"
    work_dir = args.output_dir / "work"
    log_dir = args.output_dir / "logs"
    desc_dir = args.output_dir / "descriptors"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)

    input_smi = work_dir / f"{unique}.smi"
    output_csv = desc_dir / f"{compound_id}.csv"
    log_path = log_dir / f"{compound_id}.log"
    input_smi.write_text(f"{smiles}\t{compound_id}\n", encoding="utf-8")

    command = [
        java_exe,
        "-Djava.awt.headless=true",
        "-jar",
        str(jar),
        "-maxruntime",
        str(args.padel_maxruntime_seconds),
        "-waitingjobs",
        "-1",
        "-threads",
        "1",
        "-maxcpdperfile",
        "1",
        "-2d",
        "-3d",
        "-convert3d",
        "-dir",
        str(input_smi),
        "-file",
        str(output_csv),
        "-retain3d",
        "-retainorder",
    ]

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(command) + "\n\n")
        proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            return_code = proc.wait(timeout=args.timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_specific_padel(proc, unique)
            return_code = None

    descriptor_count, nonmissing = count_descriptor_values(output_csv)
    elapsed = round(time.time() - started, 2)
    if nonmissing > 0:
        classification = "recovered"
    elif not profile["rdkit_valid"]:
        classification = "rdkit_invalid"
    elif timed_out:
        classification = "padel_timeout"
    else:
        classification = "padel_failed"

    return {
        "excel_row": row.get("excel_row", ""),
        "compound_index": row.get("compound_index", ""),
        "compound_id": compound_id,
        "cas": row.get("cas", ""),
        "value_mean": row.get("value_mean", ""),
        "raw_smiles": smiles,
        **profile,
        "classification": classification,
        "descriptor_count": descriptor_count,
        "nonmissing_descriptor_values": nonmissing,
        "timed_out": timed_out,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "descriptor_csv": str(output_csv),
        "log_file": str(log_path),
    }


def load_existing_status(status_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not status_jsonl.exists():
        return done
    with status_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["compound_id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaDEL for the 46 all-missing PFAS compounds one at a time. "
            "Default max-cases is 1 for CPU safety."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("data/all_descriptor_missing_46.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/padel_cli_one_by_one"))
    parser.add_argument("--padel-jar", type=Path, default=None)
    parser.add_argument("--compound-id", default=None)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--padel-maxruntime-seconds", type=int, default=90)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = args.output_dir / "status.jsonl"
    status_csv = args.output_dir / "status.csv"

    java_exe = shutil.which("java")
    if not java_exe:
        raise FileNotFoundError("java executable was not found on PATH.")
    jar = find_padel_jar(args.padel_jar)

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.compound_id:
        rows = [row for row in rows if row["compound_id"] == args.compound_id]
        if not rows:
            raise ValueError(f"compound-id not found: {args.compound_id}")

    done = set() if args.no_resume else load_existing_status(status_jsonl)
    todo = [row for row in rows if row["compound_id"] not in done]
    todo = todo[: max(0, args.max_cases)]

    print(f"PaDEL jar: {jar}")
    print(f"Java: {java_exe}")
    print(f"Running cases this pass: {len(todo)}")

    new_status = []
    for index, row in enumerate(todo, start=1):
        print(f"{index}/{len(todo)} {row['compound_id']} ...", flush=True)
        status = run_one(row, args, jar, java_exe)
        new_status.append(status)
        with status_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(
            f"  {status['classification']} nonmissing={status['nonmissing_descriptor_values']} "
            f"elapsed={status['elapsed_seconds']}s",
            flush=True,
        )

    all_status = []
    if status_jsonl.exists():
        with status_jsonl.open("r", encoding="utf-8") as handle:
            all_status = [json.loads(line) for line in handle if line.strip()]
    if all_status:
        with status_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_status[0].keys()))
            writer.writeheader()
            writer.writerows(all_status)
        print(f"Wrote: {status_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
