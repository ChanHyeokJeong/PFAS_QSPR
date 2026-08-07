# PFAS PaDEL Rerun Package for 46 All-Missing Compounds

This package is for verifying the 46 PFAS compounds whose original PaDEL output
contained no descriptor values in `LogP_descriptor_original.xlsx`.

It is intentionally small and does not include the original large Excel file.

## Contents

- `data/all_descriptor_missing_46.csv`: the 46 compound records extracted from the original workbook.
- `data/all_descriptor_missing_46.smi`: SMILES input file for the same 46 compounds.
- `data/all_descriptor_missing_46.json`: JSON copy of the same records.
- `data/partial_raw_screen_42_status.jsonl`: partial status from the office PC before reboot.
- `data/partial_raw_screen_42.log`: partial raw-screen log from the office PC.
- `scripts/rdkit_validate_46.py`: RDKit-only SMILES validation.
- `scripts/padel_cli_one_by_one.py`: safer PaDEL CLI rerun, defaulting to one compound per run.
- `scripts/summarize_status.py`: summarize rerun classifications.
- `scripts/emergency_stop_padel_related.ps1`: terminate only PaDEL-related java/python processes.

## Important Safety Note

Do not run all 46 compounds through PaDELPy in one batch. On the office PC,
PaDEL spawned Java processes that consumed CPU heavily. The partial screen
recorded 42 compounds before interruption; all 42 were RDKit-valid but timed
out in raw PaDEL screening at 45 seconds. This suggests molecule-level PaDEL
calculation timeout/failure rather than invalid SMILES, but the remaining cases
and longer-timeout behavior still need controlled verification.

## Setup

Recommended Conda environment:

```powershell
conda create -n pfas_padel_rerun python=3.11 -y
conda activate pfas_padel_rerun
pip install -r requirements.txt
```

Java must be available:

```powershell
java -version
```

PaDELPy should include `PaDEL-Descriptor.jar`. Quick check:

```powershell
python -c "import padelpy, pathlib; print(pathlib.Path(padelpy.__file__).resolve().parent / 'PaDEL-Descriptor' / 'PaDEL-Descriptor.jar')"
```

## Recommended Workflow

1. Validate all 46 SMILES with RDKit:

```powershell
python .\scripts\rdkit_validate_46.py
```

2. Run PaDEL for one compound only:

```powershell
python .\scripts\padel_cli_one_by_one.py --max-cases 1 --timeout-seconds 90 --padel-maxruntime-seconds 90
```

3. Check CPU and process state. If PaDEL is stuck, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\emergency_stop_padel_related.ps1
```

4. If stable, continue in very small chunks:

```powershell
python .\scripts\padel_cli_one_by_one.py --max-cases 3 --timeout-seconds 120 --padel-maxruntime-seconds 120
```

5. Summarize:

```powershell
python .\scripts\summarize_status.py
```

## Classification Logic

- `recovered`: PaDEL produced non-empty descriptor values.
- `rdkit_invalid`: RDKit could not parse the SMILES.
- `padel_timeout`: RDKit parsed the molecule, but PaDEL did not finish before the timeout.
- `padel_failed`: PaDEL returned without useful descriptors or failed for another reason.

For the manuscript response, do not call the 46 compounds "fundamentally
incalculable" based only on the original missing output. Treat them as
molecule-level PaDEL calculation/parsing/timeout failure candidates until the
controlled rerun is complete.
