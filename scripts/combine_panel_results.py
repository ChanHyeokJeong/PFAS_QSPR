from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key_vals, group in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals))
        row["n_runs"] = int(len(group))
        row["n_descriptors"] = int(group["descriptor"].nunique())
        for metric in ["mae", "rmse", "r2", "pearson_r"]:
            series = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = series.mean()
            row[f"{metric}_sd"] = series.std()
            row[f"{metric}_median"] = series.median()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel1", required=True, type=Path)
    parser.add_argument("--panel2", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel1 = pd.read_csv(args.panel1)
    panel2 = pd.read_csv(args.panel2)
    panel1.insert(0, "panel", "panel1")
    panel2.insert(0, "panel", "panel2")

    combined = pd.concat([panel1, panel2], ignore_index=True, sort=False)
    combined.to_csv(args.out_dir / "benchmark_results_final_all_methods.csv", index=False)
    summarize(combined, ["dataset", "method", "mask_frac"]).to_csv(
        args.out_dir / "benchmark_summary_final_all_methods.csv",
        index=False,
    )
    summarize(combined, ["panel", "dataset", "method", "mask_frac"]).to_csv(
        args.out_dir / "panel_comparison_summary.csv",
        index=False,
    )

    hardest = combined[combined["method"].eq("miss_transformer")].sort_values(
        ["r2", "rmse"],
        ascending=[True, False],
    )
    hardest.to_csv(args.out_dir / "misstransformer_hardest_descriptors.csv", index=False)


if __name__ == "__main__":
    main()
