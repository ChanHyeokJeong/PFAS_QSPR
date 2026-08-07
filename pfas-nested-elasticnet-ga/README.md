# PFAS Nested Elastic Net-GA

This package reruns the Elastic Net-GA descriptor selection so that GA hyperparameter optimization is performed only inside the outer training set.

## Design

- Outer split: 60% train, 20% validation, 20% test, `random_state = 42`.
- Nested GA split: the outer training set is split again into inner train and inner validation subsets.
- GA optimization: candidate Elastic Net models are fitted on inner train and scored on inner validation.
- Final descriptor selection: the optimized Elastic Net model is refitted on the outer training set only.
- Outer validation and outer test are not used during GA optimization or descriptor selection.

## Inputs

The package includes the eight Elastic Net input workbooks used for the rerun:

- `data/non_imputed/*_PLS_Filter_Loading.xlsx`
- `data/imputed/*_PLS_Filter_Loading.xlsx`

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python .\run_nested_elasticnet_ga.py
```

Run one endpoint if needed:

```powershell
python .\run_nested_elasticnet_ga.py --datasets Non-imputed --endpoints LogP
```

Outputs are written to `results/`, including:

- `nested_outer_train_ga_summary.csv`
- `nested_outer_train_ga_summary.xlsx`
- one detailed workbook per dataset/endpoint
- one selected-data workbook per dataset/endpoint

## Reviewer Wording

Use wording like this only after this nested rerun is used:

> For each endpoint-specific dataset, an outer 60/20/20 train/validation/test split was first generated with `random_state = 42`. The Elastic Net-GA optimization was performed only within the outer training subset, which was further divided into inner training and validation subsets for GA fitness evaluation. After selecting the optimized alpha and L1 ratio, the final Elastic Net descriptor-selection model was refitted using the outer training subset only, while the outer validation and test subsets remained held out from feature selection.

