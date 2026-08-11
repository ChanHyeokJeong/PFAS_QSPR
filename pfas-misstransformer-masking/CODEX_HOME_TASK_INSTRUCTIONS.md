# Codex Home Task Instructions

Use this file as the first instruction document when continuing the PFAS
MissTransformer descriptor-masking benchmark on a home PC.

## Immediate Prompt To Give Codex

```text
Read pfas-misstransformer-masking/CODEX_HOME_TASK_INSTRUCTIONS.md and continue the PFAS descriptor-masking benchmark from there. Use the existing repository conventions, avoid committing large raw data files, and push results back to the existing GitHub branch or a new codex/* branch.
```

## Goal

Continue the artificial masking validation for descriptor imputation. The next
useful task is to create a third non-overlapping descriptor panel, run the same
50% blank benchmark, and compare the new panel against the existing 50-descriptor
Panel1+Panel2 result.

## Repository Context

GitHub repository:

```text
https://github.com/ChanHyeokJeong/PFAS_QSPR
```

Current package folder:

```text
pfas-misstransformer-masking/
```

Important files:

```text
pfas-misstransformer-masking/masking_benchmark_complete_descriptors.py
pfas-misstransformer-masking/README.md
pfas-misstransformer-masking/HOME_PC_HANDOFF_PFAS_MISSTRANSFORMER_2026-08-01.md
pfas-misstransformer-masking/targets/expanded_descriptor_targets_25_panel1.csv
pfas-misstransformer-masking/targets/expanded_descriptor_targets_25_panel2.csv
pfas-misstransformer-masking/results/expanded50_combined/benchmark_summary_final_all_methods.csv
```

## Raw Data Required Locally

The LogP raw data files required for the current Panel 3 task are included in
this branch under:

```text
pfas-misstransformer-masking/data/raw/
```

Included files:

```text
LogP_descriptor_original.xlsx
PFAS_embedding_numbers.xlsx
masking_input_cache_logp_endpoint_col4.npz
```

The benchmark script can now read LogP raw files from `data/raw/` automatically,
so no manual copy is needed after cloning/pulling this branch.

The BCF files are not needed for Panel 3. Do not commit additional `.xlsx`,
`.npz`, `.zip`, generated prediction folders, or cache files unless the user
explicitly asks for a data-release commit.

## Environment Setup

From the repository root:

```powershell
cd .\pfas-misstransformer-masking
conda create -n pfas_miss python=3.11 -y
conda activate pfas_miss
pip install -r requirements.txt
```

If an environment already exists, activate it and verify imports:

```powershell
python -c "import numpy, pandas, sklearn, torch, openpyxl; print('ok')"
python -m compileall .
```

## Smoke Test

Run this before launching long jobs:

```powershell
python .\masking_benchmark_complete_descriptors.py `
  --dataset logp `
  --target-cols ALogP apol `
  --max-target-cols 2 `
  --mask-fracs 0.1 `
  --seeds 42 `
  --epochs 1 `
  --methods mean `
  --out-dir .\masking_benchmark_home_smoke
```

Expected output:

```text
masking_benchmark_home_smoke/benchmark_results.csv
masking_benchmark_home_smoke/benchmark_summary.csv
```

## Current Completed Work

Completed panels:

- Panel 1: `targets/expanded_descriptor_targets_25_panel1.csv`
- Panel 2: `targets/expanded_descriptor_targets_25_panel2.csv`

Combined result:

```text
results/expanded50_combined/benchmark_summary_final_all_methods.csv
```

Current 50-descriptor summary:

```text
method            n_desc  mean_R2   median_R2  median_RMSE
MICE              50      0.9793    0.9975     0.0541
MissForest        50      0.9765    0.9895     0.5831
MissTransformer   50      0.9561    0.9843     0.6684
kNN               50      0.9353    0.9587     0.9197
mean              50     -0.0009   -0.0004     5.0342
median            50     -0.0531   -0.0287     5.1041
```

## Next Task: Panel 3

1. Inspect descriptor names from `LogP_descriptor_original.xlsx` or the cache.
2. Exclude descriptors already used in Panel 1 and Panel 2.
3. Pick about 20-25 additional descriptors with enough finite observations.
4. Prefer diversity across descriptor families rather than choosing many near
   duplicates from one family.
5. Save the list:

```text
targets/expanded_descriptor_targets_25_panel3.csv
```

When choosing descriptors, avoid the already-used descriptors below.

Panel 1:

```text
nAcid ALogP ALogp2 AMR apol nAtom nHeavyAtom nF nO XLogP TopoPSA MLFER_L ATS4m AATS4m ATSC4m AATSC4m MATS4m GATS4m ETA_Alpha ETA_Epsilon_3 MDEC-23 MWC5 SpAbs_Dzv VE1_Dzv JGI4
```

Panel 2:

```text
naAromAtom nAromBond nH nC nX MLFER_A MLFER_BH WTPT-3 Kier1 Kier3 ATS7e AATS7e ATSC7e AATSC7e MATS7e GATS7e ETA_Beta ETA_Shape_X MDEC-34 MPC7 piPC7 SRW8 SpMax_Dze SpDiam_Dzp VR1_Dze
```

## Panel 3 Run Template

After choosing panel 3 descriptors:

```powershell
$targets = Get-Content .\targets\expanded_descriptor_targets_25_panel3.csv

python .\masking_benchmark_complete_descriptors.py `
  --dataset logp `
  --target-cols $targets `
  --max-target-cols $targets.Count `
  --mask-fracs 0.5 `
  --seeds 42 `
  --epochs 10 `
  --methods miss_transformer `
  --out-dir .\masking_benchmark_logp_50pct_expanded25_panel3_misstransformer_epoch10 `
  --save-predictions
```

Baselines:

```powershell
$targets = Get-Content .\targets\expanded_descriptor_targets_25_panel3.csv

python .\masking_benchmark_complete_descriptors.py `
  --dataset logp `
  --target-cols $targets `
  --max-target-cols $targets.Count `
  --mask-fracs 0.5 `
  --seeds 42 `
  --epochs 1 `
  --methods mean median knn mice missforest `
  --baseline-max-features 200 `
  --iterative-max-iter 5 `
  --rf-trees 50 `
  --out-dir .\masking_benchmark_logp_50pct_expanded25_panel3_baselines `
  --save-predictions
```

Important: always set `--max-target-cols $targets.Count`. The script default is
10 and will otherwise truncate the target list.

## Combine Panel 3 Results

Create:

```text
results/expanded25_panel3/
results/expanded75_combined/
```

Panel 3 should contain:

```text
benchmark_results_final_all_methods.csv
benchmark_summary_final_all_methods.csv
misstransformer_hardest_descriptors.csv
benchmark_r2_by_descriptor.csv
benchmark_rmse_by_descriptor.csv
```

The 75-descriptor combined folder should contain:

```text
benchmark_results_final_all_methods.csv
benchmark_summary_final_all_methods.csv
panel_comparison_summary.csv
method_delta_panel3_vs_prior.csv
misstransformer_hardest_descriptors_75.csv
```

Use the existing `scripts/combine_panel_results.py` as a starting point, but
extend it if needed so it can combine more than two panels.

## Reporting Requirements

When finished, summarize in Korean:

- Which descriptors were selected for Panel 3.
- Whether smoke test passed.
- Whether MissTransformer and baselines completed.
- Panel 3 summary table: method, mean R2, median R2, median RMSE.
- Combined 75-descriptor summary table.
- Hardest MissTransformer descriptors.
- Exact output file paths.
- Any warnings, especially MICE convergence warnings.

## Git Instructions

Work on a `codex/*` branch. Commit only lightweight files:

```text
README or instruction updates
targets/*.csv
results/**/*.csv
scripts/*.py
```

Do not commit:

```text
new *.xlsx files other than the already tracked LogP raw inputs
new *.npz files other than the already tracked LogP cache
*.zip
masking_benchmark_*/
__pycache__/
```

Suggested branch:

```text
codex/pfas-panel3-masking
```

Suggested commit message:

```text
Add panel 3 PFAS masking benchmark results
```
