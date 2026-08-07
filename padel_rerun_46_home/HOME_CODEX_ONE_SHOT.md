# One-Shot Prompt for Home Codex

Copy the prompt below into Codex at home after cloning or downloading this
repository.

```text
You are helping me verify 46 PFAS compounds whose original PaDEL output had all descriptor values missing.

Work inside the `padel_rerun_46_home` or `pfas_padel_rerun_46_home` folder. Do not use the original large Excel file unless I explicitly provide it. The required input is already in `data/all_descriptor_missing_46.csv`.

Important safety rules:
1. Do not run all 46 compounds in one PaDELPy batch.
2. Do not start multiple PaDEL/Java jobs in parallel.
3. First run RDKit-only validation.
4. Then run PaDEL for one compound only using the safe CLI script.
5. After every PaDEL attempt, check whether PaDEL-related Java/Python processes are still running.
6. If a PaDEL process is stuck or CPU remains high, run `scripts/emergency_stop_padel_related.ps1`.
7. Continue only in small chunks after confirming the PC is stable.

Step-by-step:

1. Inspect the folder:
   - list files under `data/` and `scripts/`
   - read `README.md`

2. Create or activate a Python environment:
   - if Conda is available, use:
     `conda create -n pfas_padel_rerun python=3.11 -y`
     `conda activate pfas_padel_rerun`
   - install dependencies:
     `pip install -r requirements.txt`
   - verify Java:
     `java -version`
   - verify PaDELPy jar path:
     `python -c "import padelpy, pathlib; print(pathlib.Path(padelpy.__file__).resolve().parent / 'PaDEL-Descriptor' / 'PaDEL-Descriptor.jar')"`

3. Run RDKit validation only:
   `python .\scripts\rdkit_validate_46.py`

4. Inspect `outputs/rdkit_validation_46.csv` and report:
   - how many of 46 are RDKit-valid
   - any invalid compound IDs and errors
   - atom count range for RDKit-valid molecules

5. Run a single PaDEL case:
   `python .\scripts\padel_cli_one_by_one.py --max-cases 1 --timeout-seconds 90 --padel-maxruntime-seconds 90`

6. Immediately check stability:
   - summarize `outputs/padel_cli_one_by_one/status.csv`
   - check whether any PaDEL-related Java/Python processes remain
   - if needed, run:
     `powershell -ExecutionPolicy Bypass -File .\scripts\emergency_stop_padel_related.ps1`

7. If the one-case run is stable, continue in small chunks only:
   `python .\scripts\padel_cli_one_by_one.py --max-cases 3 --timeout-seconds 120 --padel-maxruntime-seconds 120`

8. Summarize:
   `python .\scripts\summarize_status.py`

9. Final deliverables:
   - `outputs/rdkit_validation_46.csv`
   - `outputs/padel_cli_one_by_one/status.csv`
   - `outputs/padel_cli_one_by_one/summary.csv`
   - a short interpretation for manuscript revision:
     classify the 46 compounds into recovered, RDKit-invalid, PaDEL-timeout, and PaDEL-failed.

Use the partial office-PC evidence only as preliminary context:
`data/partial_raw_screen_42_status.jsonl` records 42 cases; all 42 were RDKit-valid and all 42 timed out in 45-second raw PaDEL screening. Do not overclaim from this partial screen.
```
