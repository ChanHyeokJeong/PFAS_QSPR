# PFAS MissTransformer Masking Benchmark

This repository packages the artificial descriptor-masking experiments used to
evaluate MissTransformer-style imputation for PFAS QSPR descriptor matrices.

The benchmark hides known finite descriptor values, imputes them, and scores
only the artificially masked cells. It is intended for reviewer-response
validation rather than for generating the final descriptor-complete matrix.

## Contents

```text
masking_benchmark_complete_descriptors.py    Main benchmark script
targets/                                     Descriptor panels used in tests
results/                                     Final CSV summaries and comparisons
scripts/combine_panel_results.py             Utility for combining panel outputs
HOME_PC_HANDOFF_PFAS_MISSTRANSFORMER_2026-08-01.md
data/README.md                               Raw data placement notes
```

Large raw inputs are not committed here because the original project contains
large Excel/cache/archive files, including a zip file above GitHub's regular
100 MB file limit. Place the raw files locally as described in `data/README.md`.

## Current Result Summary

Latest 50-descriptor LogP masking benchmark:

| Method | Descriptors | Mean R2 | Median R2 | Median RMSE |
|---|---:|---:|---:|---:|
| MICE | 50 | 0.9793 | 0.9975 | 0.0541 |
| MissForest | 50 | 0.9765 | 0.9895 | 0.5831 |
| MissTransformer | 50 | 0.9561 | 0.9843 | 0.6684 |
| kNN | 50 | 0.9353 | 0.9587 | 0.9197 |
| mean | 50 | -0.0009 | -0.0004 | 5.0342 |
| median | 50 | -0.0531 | -0.0287 | 5.1041 |

Key interpretation:

- MissTransformer is a real TransformerEncoder-based imputation model and is
  competitive across expanded descriptor panels.
- MICE and MissForest are stronger on average in this artificial masking setup.
- Performance depends strongly on descriptor family.
- `VR1_Dze` is difficult for all models, not only MissTransformer.

## Setup

```powershell
conda create -n pfas_miss python=3.11 -y
conda activate pfas_miss
pip install -r requirements.txt
```

If CUDA-enabled PyTorch is installed, run with `--device auto`; otherwise CPU is
fine but slower.

## Smoke Test

After placing the raw data files beside the script, run:

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

Expected output files:

```text
masking_benchmark_home_smoke/benchmark_results.csv
masking_benchmark_home_smoke/benchmark_summary.csv
```

## Reproduce Panel 2

```powershell
$targets = @(
  "naAromAtom","nAromBond","nH","nC","nX","MLFER_A","MLFER_BH","WTPT-3",
  "Kier1","Kier3","ATS7e","AATS7e","ATSC7e","AATSC7e","MATS7e","GATS7e",
  "ETA_Beta","ETA_Shape_X","MDEC-34","MPC7","piPC7","SRW8",
  "SpMax_Dze","SpDiam_Dzp","VR1_Dze"
)

python .\masking_benchmark_complete_descriptors.py `
  --dataset logp `
  --target-cols $targets `
  --max-target-cols $targets.Count `
  --mask-fracs 0.5 `
  --seeds 42 `
  --epochs 10 `
  --methods miss_transformer `
  --out-dir .\masking_benchmark_logp_50pct_expanded25_panel2_misstransformer_epoch10 `
  --save-predictions
```

Baselines:

```powershell
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
  --out-dir .\masking_benchmark_logp_50pct_expanded25_panel2_baselines `
  --save-predictions
```

Important: when using `--target-cols`, always set `--max-target-cols` to the
number of requested descriptors. The script default is 10.

## Result Files

Main combined 50-descriptor results:

```text
results/expanded50_combined/benchmark_summary_final_all_methods.csv
results/expanded50_combined/panel_comparison_summary.csv
results/expanded50_combined/method_delta_panel2_minus_panel1.csv
results/expanded50_combined/misstransformer_hardest_descriptors_50.csv
```

