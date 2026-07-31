# PFAS MissTransformer masking benchmark handoff

Date: 2026-08-01

This note is for continuing the PFAS descriptor-imputation masking benchmark on another Windows PC.

## 1. Project Folder

Current working folder:

```text
Z:\backup\Chanhyeok Jeong\Manuscript\4. PFAS QSPR (JHM)\Code & Data\Misstransformer
```

For the home PC, copy or sync the whole `Misstransformer` folder if possible. The safest option is to preserve the same folder contents and then run commands from inside that folder.

Minimum files needed to continue experiments:

```text
masking_benchmark_complete_descriptors.py
LogP_descriptor_original.xlsx
PFAS_embedding_numbers.xlsx
BCF_descriptor.xlsx
BCF_PFAS_embedding_numbers.xlsx
masking_input_cache_logp_endpoint_col4.npz
expanded_descriptor_targets_25.csv
expanded_descriptor_targets_25_panel2.csv
```

The cache file is optional, but keeping it avoids rebuilding the LogP descriptor/token input cache.

## 2. Python Environment

The current PC used Anaconda Python and these core packages:

```text
numpy
pandas
scikit-learn
torch
openpyxl
```

On the home PC, one simple setup is:

```powershell
conda create -n pfas_miss python=3.11 -y
conda activate pfas_miss
pip install numpy pandas scikit-learn torch openpyxl
```

If PyTorch CUDA is available, `--device auto` can use it. Otherwise CPU works, but MissTransformer and especially MissForest runs can take a long time.

## 3. Current Completed Status

Artificial masking benchmark logic:

- Script: `masking_benchmark_complete_descriptors.py`
- Task: hide known finite descriptor values, impute them, and score only the artificially masked cells.
- Dataset used in latest experiments: `logp`
- Mask fraction: `0.5`
- Seed: `42`
- MissTransformer epochs: `10`
- Baseline settings: `--baseline-max-features 200 --iterative-max-iter 5 --rf-trees 50 --save-predictions`

Completed result folders:

```text
masking_benchmark_logp_50pct_expanded25_final_all_methods
masking_benchmark_logp_50pct_expanded25_panel2_final_all_methods
masking_benchmark_logp_50pct_expanded50_panel1_panel2_final_all_methods
```

Main combined result file:

```text
masking_benchmark_logp_50pct_expanded50_panel1_panel2_final_all_methods\benchmark_summary_final_all_methods.csv
```

Latest 50-descriptor summary:

```text
method            n_desc  mean_R2   median_R2  median_RMSE
MICE              50      0.9793    0.9975     0.0541
MissForest        50      0.9765    0.9895     0.5831
MissTransformer   50      0.9561    0.9843     0.6684
kNN               50      0.9353    0.9587     0.9197
mean              50     -0.0009   -0.0004     5.0342
median            50     -0.0531   -0.0287     5.1041
```

Interpretation so far:

- MissTransformer is real TransformerEncoder-based imputation and is generally strong.
- MICE and MissForest are still stronger on average for these artificial complete-descriptor masking tests.
- The expanded descriptor panels reveal harder descriptors that were hidden in the 5-descriptor pilot.
- `VR1_Dze` is difficult for all models, not only MissTransformer.

Hardest MissTransformer descriptors across the 50-descriptor experiment:

```text
panel2  VR1_Dze    R2 0.7098
panel1  AATSC4m    R2 0.7154
panel1  GATS4m     R2 0.8131
panel2  MLFER_A    R2 0.8878
panel2  ATSC7e     R2 0.9010
panel2  MDEC-34    R2 0.9080
```

## 4. Already Used Target Panels

Panel 1 target descriptors:

```text
nAcid ALogP ALogp2 AMR apol nAtom nHeavyAtom nF nO XLogP TopoPSA MLFER_L ATS4m AATS4m ATSC4m AATSC4m MATS4m GATS4m ETA_Alpha ETA_Epsilon_3 MDEC-23 MWC5 SpAbs_Dzv VE1_Dzv JGI4
```

Panel 2 target descriptors:

```text
naAromAtom nAromBond nH nC nX MLFER_A MLFER_BH WTPT-3 Kier1 Kier3 ATS7e AATS7e ATSC7e AATSC7e MATS7e GATS7e ETA_Beta ETA_Shape_X MDEC-34 MPC7 piPC7 SRW8 SpMax_Dze SpDiam_Dzp VR1_Dze
```

When specifying targets, always include `--max-target-cols` equal to the number of requested descriptors. The default is 10, so forgetting this option silently truncates a long target list.

## 5. Smoke Test on Home PC

Run this first to verify the environment:

```powershell
cd "PATH_TO_HOME_COPY_OF\Misstransformer"
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

Expected: a small output folder with `benchmark_results.csv` and `benchmark_summary.csv`.

## 6. Re-run Panel 2 Commands If Needed

MissTransformer:

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
  --epochs 1 `
  --methods mean median knn mice missforest `
  --baseline-max-features 200 `
  --iterative-max-iter 5 `
  --rf-trees 50 `
  --out-dir .\masking_benchmark_logp_50pct_expanded25_panel2_baselines `
  --save-predictions
```

## 7. Suggested Next Work

Continue by selecting a third non-overlapping descriptor panel and running the same 50% blank benchmark:

1. Choose 20-25 descriptors not already in Panel 1 or Panel 2.
2. Save the list as `expanded_descriptor_targets_25_panel3.csv`.
3. Run MissTransformer with `--epochs 10`.
4. Run baselines with `--baseline-max-features 200 --iterative-max-iter 5 --rf-trees 50`.
5. Combine the result summaries with the existing 50-descriptor result.

For model comparison in the manuscript response, emphasize:

- This is an artificial masking validation on originally observed descriptor values.
- Metrics are computed only on artificially masked cells.
- Expanded descriptor coverage shows the imputation task is descriptor-family dependent.
- MICE/MissForest are strong classical baselines; MissTransformer is competitive but not uniformly best.

