from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COL = "Value.MeanValue"

OUTER_TRAIN_RATIO = 0.6
OUTER_VAL_RATIO = 0.2
OUTER_TEST_RATIO = 0.2
INNER_VAL_FRACTION_OF_OUTER_TRAIN = 0.2

POP_SIZE = 28
N_GEN = 25
TOURNAMENT_K = 3
CROSSOVER_PROB = 0.85
MUTATION_PROB = 0.25

ALPHA_MIN = 1e-4
ALPHA_MAX = 1e2
L1_MIN = 0.0
L1_MAX = 1.0

MAX_ITER = 15000
TOL = 1e-3
SELECTION = "random"

COEF_EPS = 1e-8
PATIENCE = 10
MIN_DELTA = 1e-4


@dataclass
class DatasetSpec:
    dataset: str
    endpoint: str
    path: Path


@dataclass
class SummaryRow:
    dataset: str
    endpoint: str
    source_file: str
    rows: int
    outer_train_n: int
    outer_validation_n: int
    outer_test_n: int
    inner_train_n: int
    inner_validation_n: int
    best_alpha: float
    best_l1_ratio: float
    best_fitness: float
    best_inner_validation_r2: float
    stopped_generation: int
    final_no_improve_count: int
    n_features: int
    n_selected_nonzero: int
    selected_fraction: float
    outer_validation_r2: float
    outer_validation_rmse: float
    outer_validation_mae: float
    outer_test_r2: float
    outer_test_rmse: float
    outer_test_mae: float
    output_workbook: str
    selected_data_file: str
    elapsed_seconds: float


def default_specs() -> list[DatasetSpec]:
    base = Path(__file__).resolve().parent / "data"
    non = base / "non_imputed"
    imp = base / "imputed"
    return [
        DatasetSpec(
            "Non-imputed",
            "LC50",
            non / "LC50_descriptor_missingremoval_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Non-imputed",
            "LogBCF",
            non / "LogBCF_descriptor_missingremoval_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Non-imputed",
            "LogKOA",
            non / "LogKOA_descriptor_missingremoval_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Non-imputed",
            "LogP",
            non / "LogP_descriptor_missingremoval_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Imputed",
            "LC50",
            imp / "LC50_imputed_finiteRowCleaned_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Imputed",
            "LogBCF",
            imp / "LogBCF_imputed_finiteRowCleaned_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Imputed",
            "LogKOA",
            imp / "LogKOA_imputed_finiteRowCleaned_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
        DatasetSpec(
            "Imputed",
            "LogP",
            imp / "LogP_imputed_finiteRowCleaned_2sigma_cleaned_PLS_Filter_Loading.xlsx",
        ),
    ]


def find_target_column(frame: pd.DataFrame) -> str:
    if TARGET_COL in frame.columns:
        return TARGET_COL
    candidates = [col for col in frame.columns if "MeanValue" in str(col)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot identify target column. Candidates: {candidates}")


def load_dataset(path: Path):
    xls = pd.ExcelFile(path)
    sheet = "FilteredData" if "FilteredData" in xls.sheet_names else xls.sheet_names[0]
    frame = pd.read_excel(path, sheet_name=sheet)
    target_col = find_target_column(frame)

    y = pd.to_numeric(frame[target_col], errors="coerce")
    valid = y.notna() & np.isfinite(y.to_numpy())
    frame = frame.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True).astype(float)

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col != target_col]
    if not feature_columns:
        raise ValueError(f"No numeric descriptors found in {path}")

    X = frame[feature_columns].astype(float)
    return frame, X, y, feature_columns, target_col, sheet


def split_outer_indices(n_rows: int):
    indices = np.arange(n_rows)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - OUTER_TRAIN_RATIO),
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    return train_idx, val_idx, test_idx


def split_inner_indices(outer_train_idx: np.ndarray):
    inner_train_idx, inner_val_idx = train_test_split(
        outer_train_idx,
        test_size=INNER_VAL_FRACTION_OF_OUTER_TRAIN,
        random_state=RANDOM_STATE,
    )
    return inner_train_idx, inner_val_idx


def fit_preprocessor(X_train: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, imputer, scaler


def transform_preprocessor(X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler):
    return scaler.transform(imputer.transform(X))


def build_enet(alpha: float, l1_ratio: float) -> ElasticNet:
    return ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=MAX_ITER,
        tol=TOL,
        selection=SELECTION,
        random_state=RANDOM_STATE,
    )


def calc_metrics(y_true, y_pred) -> tuple[float, float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
    )


def sample_individual(rng: np.random.Generator) -> np.ndarray:
    log_alpha = rng.uniform(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX))
    alpha = 10 ** log_alpha
    l1_ratio = rng.uniform(L1_MIN, L1_MAX)
    return np.array([alpha, l1_ratio], dtype=float)


def clip_individual(individual: np.ndarray) -> np.ndarray:
    individual[0] = float(np.clip(individual[0], ALPHA_MIN, ALPHA_MAX))
    individual[1] = float(np.clip(individual[1], L1_MIN, L1_MAX))
    return individual


def tournament_select(
    population: list[np.ndarray],
    fitness: np.ndarray,
    rng: np.random.Generator,
    k: int = TOURNAMENT_K,
) -> np.ndarray:
    candidates = rng.integers(0, len(population), size=k)
    best = candidates[0]
    for candidate in candidates[1:]:
        if fitness[candidate] > fitness[best]:
            best = candidate
    return population[best].copy()


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    weight = rng.uniform(0.0, 1.0)
    c1 = weight * p1 + (1.0 - weight) * p2
    c2 = (1.0 - weight) * p1 + weight * p2
    return clip_individual(c1), clip_individual(c2)


def mutate(individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.uniform() < 0.5:
        log_alpha = np.log10(individual[0])
        log_alpha += rng.normal(0.0, 0.30)
        individual[0] = 10 ** log_alpha
    else:
        individual[1] += rng.normal(0.0, 0.10)
    return clip_individual(individual)


def evaluate_individual(
    individual: np.ndarray,
    X_inner_train,
    y_inner_train,
    X_inner_val,
    y_inner_val,
):
    alpha, l1_ratio = individual
    model = build_enet(alpha, l1_ratio)
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            model.fit(X_inner_train, y_inner_train)
            pred = model.predict(X_inner_val)
            val_r2, val_rmse, val_mae = calc_metrics(y_inner_val, pred)
            convergence_warning = any(
                isinstance(warning.message, ConvergenceWarning)
                for warning in warning_list
            )
            fitness = val_r2 - 0.50 if convergence_warning else val_r2 - 1e-6 * val_rmse
            return float(fitness), val_r2, val_rmse, val_mae, bool(convergence_warning)
        except Exception:
            return -1e9, -1e9, 1e9, 1e9, True


def ga_optimize(X_inner_train, y_inner_train, X_inner_val, y_inner_val):
    rng = np.random.default_rng(RANDOM_STATE)
    population = [sample_individual(rng) for _ in range(POP_SIZE)]
    fitness = np.zeros(POP_SIZE, dtype=float)
    val_r2 = np.zeros(POP_SIZE, dtype=float)
    val_rmse = np.zeros(POP_SIZE, dtype=float)
    val_mae = np.zeros(POP_SIZE, dtype=float)
    warnings_seen = np.zeros(POP_SIZE, dtype=bool)

    best_individual = None
    best_fitness = -1e18
    best_inner_r2 = np.nan
    no_improve = 0
    history = []

    for generation in range(1, N_GEN + 1):
        for i in range(POP_SIZE):
            (
                fitness[i],
                val_r2[i],
                val_rmse[i],
                val_mae[i],
                warnings_seen[i],
            ) = evaluate_individual(
                population[i],
                X_inner_train,
                y_inner_train,
                X_inner_val,
                y_inner_val,
            )

        generation_best_index = int(np.argmax(fitness))
        generation_best_fitness = float(fitness[generation_best_index])

        if generation_best_fitness > best_fitness + MIN_DELTA:
            best_fitness = generation_best_fitness
            best_individual = population[generation_best_index].copy()
            best_inner_r2 = float(val_r2[generation_best_index])
            no_improve = 0
        else:
            no_improve += 1

        history.append(
            {
                "gen": generation,
                "best_alpha": float(best_individual[0]),
                "best_l1_ratio": float(best_individual[1]),
                "best_fitness": float(best_fitness),
                "best_inner_validation_r2": float(best_inner_r2),
                "gen_best_alpha": float(population[generation_best_index][0]),
                "gen_best_l1_ratio": float(population[generation_best_index][1]),
                "gen_best_fitness": generation_best_fitness,
                "gen_best_inner_validation_r2": float(val_r2[generation_best_index]),
                "gen_best_inner_validation_rmse": float(val_rmse[generation_best_index]),
                "gen_best_inner_validation_mae": float(val_mae[generation_best_index]),
                "gen_best_convergence_warning": bool(warnings_seen[generation_best_index]),
                "no_improve_count": int(no_improve),
            }
        )

        if no_improve >= PATIENCE:
            break

        new_population = [best_individual.copy()]
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population, fitness, rng)
            p2 = tournament_select(population, fitness, rng)

            if rng.uniform() < CROSSOVER_PROB:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2

            if rng.uniform() < MUTATION_PROB:
                c1 = mutate(c1, rng)
            if rng.uniform() < MUTATION_PROB:
                c2 = mutate(c2, rng)

            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)

        population = new_population

    if best_individual is None:
        raise RuntimeError("GA did not produce a valid individual")

    return best_individual, pd.DataFrame(history)


def predictions_frame(split_name: str, indices: np.ndarray, y_true, y_pred):
    return pd.DataFrame(
        {
            "Split": split_name,
            "RowIndex": indices,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "residual": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        }
    )


def run_one(spec: DatasetSpec, output_dir: Path) -> SummaryRow:
    started = time.perf_counter()
    frame, X, y, feature_columns, target_col, sheet = load_dataset(spec.path)

    outer_train_idx, outer_val_idx, outer_test_idx = split_outer_indices(len(frame))
    inner_train_idx, inner_val_idx = split_inner_indices(outer_train_idx)

    X_inner_train_scaled, inner_imputer, inner_scaler = fit_preprocessor(
        X.iloc[inner_train_idx]
    )
    X_inner_val_scaled = transform_preprocessor(
        X.iloc[inner_val_idx], inner_imputer, inner_scaler
    )

    best_individual, history = ga_optimize(
        X_inner_train_scaled,
        y.iloc[inner_train_idx].to_numpy(),
        X_inner_val_scaled,
        y.iloc[inner_val_idx].to_numpy(),
    )
    best_alpha = float(best_individual[0])
    best_l1_ratio = float(best_individual[1])

    X_outer_train_scaled, outer_imputer, outer_scaler = fit_preprocessor(
        X.iloc[outer_train_idx]
    )
    X_outer_val_scaled = transform_preprocessor(
        X.iloc[outer_val_idx], outer_imputer, outer_scaler
    )
    X_outer_test_scaled = transform_preprocessor(
        X.iloc[outer_test_idx], outer_imputer, outer_scaler
    )

    final_model = build_enet(best_alpha, best_l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(X_outer_train_scaled, y.iloc[outer_train_idx].to_numpy())

    val_pred = final_model.predict(X_outer_val_scaled)
    test_pred = final_model.predict(X_outer_test_scaled)
    val_r2, val_rmse, val_mae = calc_metrics(y.iloc[outer_val_idx].to_numpy(), val_pred)
    test_r2, test_rmse, test_mae = calc_metrics(
        y.iloc[outer_test_idx].to_numpy(), test_pred
    )

    coef = np.asarray(final_model.coef_).ravel()
    coef_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Coef": coef,
            "AbsCoef": np.abs(coef),
            "Selected": np.abs(coef) > COEF_EPS,
        }
    ).sort_values(["Selected", "AbsCoef"], ascending=[False, False])

    selected_features = coef_df.loc[coef_df["Selected"], "Feature"].tolist()
    n_selected = int(len(selected_features))
    safe_dataset = spec.dataset.replace("-", "_").replace(" ", "_")
    stem = f"{safe_dataset}_{spec.endpoint}_NestedOuterTrainGA_ElasticNet"
    workbook_path = output_dir / f"{stem}.xlsx"
    selected_data_path = output_dir / f"{stem}_SelectedData.xlsx"

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    meta_columns = [col for col in frame.columns if col not in numeric_columns]
    selected_frame = frame[meta_columns + [target_col] + selected_features].copy()

    split_df = pd.concat(
        [
            pd.DataFrame({"Split": "OuterTrain", "RowIndex": outer_train_idx}),
            pd.DataFrame({"Split": "OuterValidation", "RowIndex": outer_val_idx}),
            pd.DataFrame({"Split": "OuterTest", "RowIndex": outer_test_idx}),
            pd.DataFrame({"Split": "InnerTrainWithinOuterTrain", "RowIndex": inner_train_idx}),
            pd.DataFrame({"Split": "InnerValidationWithinOuterTrain", "RowIndex": inner_val_idx}),
        ],
        ignore_index=True,
    )

    best_df = pd.DataFrame(
        [
            {
                "Dataset": spec.dataset,
                "Endpoint": spec.endpoint,
                "SourceFile": str(spec.path),
                "SheetUsed": sheet,
                "TargetCol": target_col,
                "Best_alpha": best_alpha,
                "Best_l1_ratio": best_l1_ratio,
                "N_features": len(feature_columns),
                "N_selected_nonzero": n_selected,
                "Selected_fraction": n_selected / len(feature_columns),
                "Outer_train_ratio": OUTER_TRAIN_RATIO,
                "Outer_validation_ratio": OUTER_VAL_RATIO,
                "Outer_test_ratio": OUTER_TEST_RATIO,
                "Inner_validation_fraction_of_outer_train": INNER_VAL_FRACTION_OF_OUTER_TRAIN,
                "POP_SIZE": POP_SIZE,
                "N_GEN": N_GEN,
                "PATIENCE": PATIENCE,
                "MIN_DELTA": MIN_DELTA,
                "ALPHA_MIN": ALPHA_MIN,
                "ALPHA_MAX": ALPHA_MAX,
                "L1_MIN": L1_MIN,
                "L1_MAX": L1_MAX,
            }
        ]
    )
    perf_df = pd.DataFrame(
        [
            ["OuterValidation", val_r2, val_rmse, val_mae, len(outer_val_idx)],
            ["OuterTest", test_r2, test_rmse, test_mae, len(outer_test_idx)],
        ],
        columns=["Split", "R2", "RMSE", "MAE", "N"],
    )
    selected_features_df = pd.DataFrame({"Feature": selected_features})

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        best_df.to_excel(writer, sheet_name="BestParams", index=False)
        perf_df.to_excel(writer, sheet_name="Performance", index=False)
        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        selected_features_df.to_excel(writer, sheet_name="SelectedFeatures", index=False)
        history.to_excel(writer, sheet_name="GA_History", index=False)
        split_df.to_excel(writer, sheet_name="Split_Indices", index=False)
        predictions_frame(
            "OuterValidation",
            outer_val_idx,
            y.iloc[outer_val_idx].to_numpy(),
            val_pred,
        ).to_excel(writer, sheet_name="Predictions_OuterVal", index=False)
        predictions_frame(
            "OuterTest",
            outer_test_idx,
            y.iloc[outer_test_idx].to_numpy(),
            test_pred,
        ).to_excel(writer, sheet_name="Predictions_OuterTest", index=False)

    selected_frame.to_excel(selected_data_path, index=False)

    last_history = history.iloc[-1]
    best_history = history.loc[history["best_fitness"].idxmax()]
    elapsed = time.perf_counter() - started

    return SummaryRow(
        dataset=spec.dataset,
        endpoint=spec.endpoint,
        source_file=str(spec.path),
        rows=len(frame),
        outer_train_n=len(outer_train_idx),
        outer_validation_n=len(outer_val_idx),
        outer_test_n=len(outer_test_idx),
        inner_train_n=len(inner_train_idx),
        inner_validation_n=len(inner_val_idx),
        best_alpha=best_alpha,
        best_l1_ratio=best_l1_ratio,
        best_fitness=float(best_history["best_fitness"]),
        best_inner_validation_r2=float(best_history["best_inner_validation_r2"]),
        stopped_generation=int(last_history["gen"]),
        final_no_improve_count=int(last_history["no_improve_count"]),
        n_features=len(feature_columns),
        n_selected_nonzero=n_selected,
        selected_fraction=n_selected / len(feature_columns),
        outer_validation_r2=val_r2,
        outer_validation_rmse=val_rmse,
        outer_validation_mae=val_mae,
        outer_test_r2=test_r2,
        outer_test_rmse=test_rmse,
        outer_test_mae=test_mae,
        output_workbook=str(workbook_path),
        selected_data_file=str(selected_data_path),
        elapsed_seconds=elapsed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nested Elastic Net-GA descriptor selection inside the outer training split."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["Non-imputed", "Imputed"],
        default=["Non-imputed", "Imputed"],
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=["LC50", "LogBCF", "LogKOA", "LogP"],
        default=["LC50", "LogBCF", "LogKOA", "LogP"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        spec
        for spec in default_specs()
        if spec.dataset in args.datasets and spec.endpoint in args.endpoints
    ]
    rows = []
    for spec in specs:
        print(f"[RUN] {spec.dataset} / {spec.endpoint}: {spec.path.name}", flush=True)
        row = run_one(spec, args.output_dir)
        rows.append(row)
        print(
            "[DONE] "
            f"{spec.dataset} / {spec.endpoint}: "
            f"alpha={row.best_alpha:.6g}, "
            f"l1={row.best_l1_ratio:.6f}, "
            f"selected={row.n_selected_nonzero}/{row.n_features}, "
            f"test_R2={row.outer_test_r2:.4f}, "
            f"gen={row.stopped_generation}",
            flush=True,
        )

    summary = pd.DataFrame([asdict(row) for row in rows])
    csv_path = args.output_dir / "nested_outer_train_ga_summary.csv"
    xlsx_path = args.output_dir / "nested_outer_train_ga_summary.xlsx"
    summary.to_csv(csv_path, index=False)
    summary.to_excel(xlsx_path, index=False)
    print(f"[SAVED] {csv_path}", flush=True)
    print(f"[SAVED] {xlsx_path}", flush=True)


if __name__ == "__main__":
    main()
