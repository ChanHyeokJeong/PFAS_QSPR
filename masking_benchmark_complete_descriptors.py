# -*- coding: utf-8 -*-
"""
Artificial masking benchmark for MissTransformer.

This script hides known finite descriptor values, imputes them, and scores only
the artificially masked cells. It is intended for reviewer-response validation,
not for producing the final descriptor-complete matrix.

Example smoke run:
    python masking_benchmark_complete_descriptors.py --dataset logp --max-target-cols 3 --epochs 3 --methods miss_transformer mean median

Example fuller run:
    python masking_benchmark_complete_descriptors.py --dataset logp --mask-fracs 0.1 0.2 0.3 --seeds 42 43 44 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASETS = {
    "logp": {
        "descriptor_xlsx": "LogP_descriptor_original.xlsx",
        "token_xlsx": "PFAS_embedding_numbers.xlsx",
        "out_dir": "masking_benchmark_logp",
    },
    "bcf": {
        "descriptor_xlsx": "BCF_descriptor.xlsx",
        "token_xlsx": "BCF_PFAS_embedding_numbers.xlsx",
        "out_dir": "masking_benchmark_bcf",
    },
}

PAD_ID = 0


@dataclass
class MetricRow:
    dataset: str
    descriptor: str
    method: str
    seed: int
    mask_frac: float
    n_observed: int
    n_masked: int
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    seconds: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite_array(x: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def safe_metric_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 and np.std(y_true) > 0 else np.nan
    pearson_r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else np.nan
    )
    return {
        "mae": safe_metric_float(mae),
        "rmse": safe_metric_float(rmse),
        "r2": safe_metric_float(r2),
        "pearson_r": safe_metric_float(pearson_r),
    }


def find_metadata(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros((len(df), 0), dtype=np.float32)

    candidates = ["Value.MeanValue", "logP", "LogP", "value"]
    md_col = next((c for c in candidates if c in df.columns), None)
    if md_col is None:
        raise ValueError(f"Metadata column not found. Tried: {candidates}")

    md = pd.to_numeric(df[md_col], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
    md[~np.isfinite(md)] = np.nan
    mean = np.nanmean(md, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    inds = np.where(np.isnan(md))
    md[inds] = np.take(mean, inds[1])
    return md.astype(np.float32)


def load_excel_inputs(
    base_dir: Path,
    dataset: str,
    descriptor_start_col: int,
    metadata_mode: str,
    use_cache: bool,
    rebuild_cache: bool,
) -> tuple[pd.DataFrame | None, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    spec = DATASETS[dataset]
    desc_path = base_dir / spec["descriptor_xlsx"]
    tok_path = base_dir / spec["token_xlsx"]
    if not desc_path.exists():
        raise FileNotFoundError(desc_path)
    if not tok_path.exists():
        raise FileNotFoundError(tok_path)

    cache_path = base_dir / f"masking_input_cache_{dataset}_{metadata_mode}_col{descriptor_start_col}.npz"
    if use_cache and cache_path.exists() and not rebuild_cache:
        print(f"Loading cached arrays: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        descriptor_names = cached["descriptor_names"].tolist()
        return (
            None,
            cached["descriptors_np"].astype(np.float32),
            cached["token_ids"].astype(np.int64),
            cached["metadata"].astype(np.float32),
            descriptor_names,
        )

    print(f"Loading descriptors: {desc_path}")
    raw = pd.read_excel(desc_path)
    raw.columns = raw.columns.astype(str).str.strip()

    descriptors_df = raw.iloc[:, descriptor_start_col:].apply(pd.to_numeric, errors="coerce")
    descriptors_np = finite_array(descriptors_df)
    descriptor_names = list(descriptors_df.columns)

    print(f"Loading token ids: {tok_path}")
    token_df = pd.read_excel(tok_path, index_col=0)
    token_ids = token_df.iloc[:, 1:].fillna(0).astype(int).to_numpy(dtype=np.int64)

    if len(token_ids) != len(descriptors_np):
        raise ValueError(f"Row mismatch: token_ids={len(token_ids)} vs descriptors={len(descriptors_np)}")

    metadata = find_metadata(raw, metadata_mode)
    print(
        f"Loaded rows={len(descriptors_np)}, descriptors={len(descriptor_names)}, "
        f"seq_len={token_ids.shape[1]}, metadata_dim={metadata.shape[1]}"
    )
    if use_cache:
        print(f"Saving cached arrays: {cache_path}")
        np.savez_compressed(
            cache_path,
            descriptors_np=descriptors_np.astype(np.float32),
            token_ids=token_ids.astype(np.int64),
            metadata=metadata.astype(np.float32),
            descriptor_names=np.array(descriptor_names, dtype=object),
        )
    return raw, descriptors_np, token_ids, metadata, descriptor_names


def choose_targets(
    descriptor_names: list[str],
    descriptors_np: np.ndarray,
    requested: list[str] | None,
    max_target_cols: int | None,
    min_observed: int,
) -> list[int]:
    name_to_idx = {name: i for i, name in enumerate(descriptor_names)}
    if requested:
        missing = [name for name in requested if name not in name_to_idx]
        if missing:
            raise ValueError(f"Requested target descriptors not found: {missing}")
        candidates = [name_to_idx[name] for name in requested]
    else:
        candidates = list(range(len(descriptor_names)))

    usable = []
    for j in candidates:
        n = int(np.isfinite(descriptors_np[:, j]).sum())
        if n >= min_observed:
            usable.append(j)

    if max_target_cols is not None:
        usable = usable[:max_target_cols]
    return usable


class PFASImputeDataset(Dataset):
    def __init__(self, tokens, md, x_desc, x_mask, y):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.md = torch.tensor(md, dtype=torch.float32)
        self.x_desc = torch.tensor(x_desc, dtype=torch.float32)
        self.x_mask = torch.tensor(x_mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.md[idx], self.x_desc[idx], self.x_mask[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerFusionImputer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_desc: int,
        md_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        fusion: str = "concat",
    ):
        super().__init__()
        self.fusion = fusion
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.desc_encoder = nn.Sequential(
            nn.Linear(n_desc * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.use_md = md_dim > 0
        if self.use_md:
            self.md_encoder = nn.Sequential(
                nn.Linear(md_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        in_dim = d_model + d_model + (d_model if self.use_md else 0) if fusion == "concat" else d_model
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode_smiles(self, token_ids):
        x = self.embedding(token_ids)
        x = self.pos_encoder(x)
        pad_mask = token_ids == PAD_ID
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, token_ids, md, x_desc, x_mask):
        s = self.encode_smiles(token_ids)
        d = self.desc_encoder(torch.cat([x_desc, x_mask], dim=1))
        if self.use_md:
            m = self.md_encoder(md)
            z = torch.cat([s, d, m], dim=1) if self.fusion == "concat" else s + d + m
        else:
            z = torch.cat([s, d], dim=1) if self.fusion == "concat" else s + d
        return self.head(z).squeeze(-1)


def descriptor_z_and_mask(x_masked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_masked, axis=0)
    std = np.nanstd(x_masked, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[(~np.isfinite(std)) | (std == 0)] = 1.0
    z = (x_masked - mean) / std
    mask = np.isfinite(z).astype(np.float32)
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32), mask.astype(np.float32)


def train_transformer_for_mask(
    x_original: np.ndarray,
    x_masked: np.ndarray,
    token_ids: np.ndarray,
    metadata: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_j: int,
    args,
    device: torch.device,
) -> np.ndarray:
    desc_z, desc_mask = descriptor_z_and_mask(x_masked)

    y_train = x_original[train_idx, target_j].astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if not math.isfinite(y_std) or y_std == 0:
        y_std = 1.0
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32)

    x_desc_train = desc_z[train_idx].copy()
    x_mask_train = desc_mask[train_idx].copy()
    x_desc_train[:, target_j] = 0.0
    x_mask_train[:, target_j] = 0.0

    dataset = PFASImputeDataset(
        token_ids[train_idx],
        metadata[train_idx],
        x_desc_train,
        x_mask_train,
        y_train_z,
    )
    n_val = max(1, int(round(len(dataset) * args.val_frac))) if len(dataset) >= 5 else 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed_for_split),
        )
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False) if val_ds else None

    vocab_size = int(token_ids.max()) + 2
    seq_len = token_ids.shape[1]
    model = TransformerFusionImputer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_desc=x_original.shape[1],
        md_dim=metadata.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        fusion="concat",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for tokens, md, x_desc, x_mask, y in train_loader:
            tokens = tokens.to(device)
            md = md.to(device)
            x_desc = x_desc.to(device)
            x_mask = x_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(tokens, md, x_desc, x_mask)
            loss = F.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite transformer loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for tokens, md, x_desc, x_mask, y in val_loader:
                    pred = model(tokens.to(device), md.to(device), x_desc.to(device), x_mask.to(device))
                    val_losses.append(float(F.mse_loss(pred, y.to(device)).detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(losses))
        else:
            val_loss = float(np.mean(losses))

        if val_loss < best_val - args.early_stop_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.early_stop and bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    x_desc_test = desc_z[test_idx].copy()
    x_mask_test = desc_mask[test_idx].copy()
    x_desc_test[:, target_j] = 0.0
    x_mask_test[:, target_j] = 0.0

    test_ds = PFASImputeDataset(
        token_ids[test_idx],
        metadata[test_idx],
        x_desc_test,
        x_mask_test,
        np.zeros(len(test_idx), dtype=np.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    preds = []
    model.eval()
    with torch.no_grad():
        for tokens, md, x_desc, x_mask, _ in test_loader:
            pred_z = model(tokens.to(device), md.to(device), x_desc.to(device), x_mask.to(device))
            preds.append(pred_z.detach().cpu().numpy())
    pred_z = np.concatenate(preds).astype(np.float32)
    return pred_z * y_std + y_mean


def baseline_predict(
    method: str,
    x_masked: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_j: int,
    args,
) -> np.ndarray:
    y_train = x_masked[train_idx, target_j]
    if method == "mean":
        return np.full(len(test_idx), np.nanmean(y_train), dtype=np.float32)
    if method == "median":
        return np.full(len(test_idx), np.nanmedian(y_train), dtype=np.float32)

    keep_cols = choose_baseline_feature_mask(
        x_train_all=x_masked[train_idx],
        target_j=target_j,
        max_features=args.baseline_max_features,
        min_pairs=args.baseline_min_corr_pairs,
    )
    keep_cols[target_j] = True
    local_target = int(np.where(np.flatnonzero(keep_cols) == target_j)[0][0])
    x_train = x_masked[train_idx][:, keep_cols]
    x_test = x_masked[test_idx][:, keep_cols]
    x_train, x_test, scale_mean, scale_std = standardize_for_imputer(x_train, x_test)

    if method == "knn":
        imputer = KNNImputer(n_neighbors=args.knn_neighbors, weights="distance")
    elif method == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=args.iterative_max_iter,
            random_state=args.current_seed,
            initial_strategy="median",
            skip_complete=False,
        )
    elif method == "missforest":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=args.rf_trees,
                random_state=args.current_seed,
                n_jobs=args.n_jobs,
                min_samples_leaf=2,
            ),
            max_iter=args.iterative_max_iter,
            random_state=args.current_seed,
            initial_strategy="median",
            skip_complete=False,
        )
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    imputer.fit(x_train)
    x_imp = imputer.transform(x_test)
    pred = x_imp[:, local_target] * scale_std[local_target] + scale_mean[local_target]
    return pred.astype(np.float32)


def standardize_for_imputer(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = x_train.astype(np.float64, copy=True)
    x_test = x_test.astype(np.float64, copy=True)
    x_train[~np.isfinite(x_train)] = np.nan
    x_test[~np.isfinite(x_test)] = np.nan

    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[(~np.isfinite(std)) | (std == 0)] = 1.0

    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std
    return x_train_z, x_test_z, mean, std


def choose_baseline_feature_mask(
    x_train_all: np.ndarray,
    target_j: int,
    max_features: int,
    min_pairs: int,
) -> np.ndarray:
    keep_available = np.isfinite(x_train_all).any(axis=0)
    keep_available[target_j] = True
    n_available = int(keep_available.sum())
    if max_features <= 0 or max_features >= n_available:
        return keep_available

    y = x_train_all[:, target_j]
    y_ok = np.isfinite(y)
    scores = np.full(x_train_all.shape[1], -np.inf, dtype=np.float64)
    for j in np.flatnonzero(keep_available):
        if j == target_j:
            continue
        x = x_train_all[:, j]
        ok = y_ok & np.isfinite(x)
        if int(ok.sum()) < min_pairs:
            continue
        x_ok = x[ok].astype(np.float64)
        y_sub = y[ok].astype(np.float64)
        if np.std(x_ok) == 0 or np.std(y_sub) == 0:
            continue
        r = np.corrcoef(x_ok, y_sub)[0, 1]
        if np.isfinite(r):
            scores[j] = abs(float(r))

    n_aux = max(0, max_features - 1)
    ranked = np.argsort(scores)[::-1]
    selected = [j for j in ranked if np.isfinite(scores[j])][:n_aux]
    keep = np.zeros(x_train_all.shape[1], dtype=bool)
    keep[selected] = True
    keep[target_j] = True
    return keep


def run_one_target(
    dataset: str,
    descriptor_name: str,
    target_j: int,
    x_original: np.ndarray,
    token_ids: np.ndarray,
    metadata: np.ndarray,
    seed: int,
    mask_frac: float,
    args,
    device: torch.device,
    out_dir: Path,
) -> list[MetricRow]:
    args.current_seed = seed
    seed_all(seed)
    rng = np.random.default_rng(seed)

    finite_idx = np.where(np.isfinite(x_original[:, target_j]))[0]
    if args.complete_rows_only:
        complete_rows = np.isfinite(x_original).all(axis=1)
        finite_idx = finite_idx[complete_rows[finite_idx]]
    n_observed = len(finite_idx)
    if n_observed < args.min_observed:
        print(f"Skip {descriptor_name}: only {n_observed} usable observed values.")
        return []

    n_masked = int(round(n_observed * mask_frac))
    n_masked = min(max(args.min_masked, n_masked), n_observed - args.min_train)
    if n_masked <= 0:
        print(f"Skip {descriptor_name}: not enough values after mask/train constraints.")
        return []

    test_idx = np.sort(rng.choice(finite_idx, size=n_masked, replace=False))
    train_idx = np.setdiff1d(finite_idx, test_idx, assume_unique=False)
    y_true = x_original[test_idx, target_j]

    x_masked = x_original.copy()
    x_masked[test_idx, target_j] = np.nan

    rows = []
    pred_records = {"row_index": test_idx, "y_true": y_true}
    for method in args.methods:
        print(f"  {descriptor_name} | seed={seed} | mask={mask_frac:.2f} | {method}", flush=True)
        t0 = time.time()
        try:
            if method == "miss_transformer":
                y_pred = train_transformer_for_mask(
                    x_original=x_original,
                    x_masked=x_masked,
                    token_ids=token_ids,
                    metadata=metadata,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    target_j=target_j,
                    args=args,
                    device=device,
                )
            else:
                y_pred = baseline_predict(method, x_masked, train_idx, test_idx, target_j, args)
            metrics = compute_metrics(y_true, y_pred)
            pred_records[f"pred_{method}"] = y_pred
        except Exception as exc:
            print(f"    FAILED {method}: {exc}")
            metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan}
        rows.append(
            MetricRow(
                dataset=dataset,
                descriptor=descriptor_name,
                method=method,
                seed=seed,
                mask_frac=mask_frac,
                n_observed=n_observed,
                n_masked=n_masked,
                seconds=time.time() - t0,
                **metrics,
            )
        )
        incremental_path = out_dir / "benchmark_results_incremental.csv"
        pd.DataFrame([asdict(rows[-1])]).to_csv(
            incremental_path,
            mode="a",
            header=not incremental_path.exists(),
            index=False,
        )

    if args.save_predictions:
        pred_dir = out_dir / "predictions" / f"seed_{seed}" / f"mask_{mask_frac:g}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_records).to_csv(pred_dir / f"{descriptor_name}.csv", index=False)

    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "r2", "pearson_r"]
    group_cols = ["dataset", "method", "mask_frac"]
    parts = []
    for keys, g in results.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        row["n_descriptors"] = g["descriptor"].nunique()
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sd"] = g[col].std()
            row[f"{col}_median"] = g[col].median()
        parts.append(row)
    return pd.DataFrame(parts).sort_values(group_cols).reset_index(drop=True)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="logp")
    parser.add_argument("--descriptor-start-col", type=int, default=4)
    parser.add_argument("--metadata-mode", choices=["endpoint", "none"], default="endpoint")
    parser.add_argument("--target-cols", nargs="*", default=None)
    parser.add_argument("--max-target-cols", type=int, default=10)
    parser.add_argument("--mask-fracs", nargs="+", type=float, default=[0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["miss_transformer", "mean", "median", "knn", "mice", "missforest"],
        default=["miss_transformer", "mean", "median", "knn", "mice", "missforest"],
    )
    parser.add_argument("--complete-rows-only", action="store_true")
    parser.add_argument("--min-observed", type=int, default=50)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--min-masked", type=int, default=5)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=1e-6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--iterative-max-iter", type=int, default=10)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--baseline-max-features",
        type=int,
        default=0,
        help="For kNN/MICE/MissForest, use target plus top correlated auxiliary descriptors. 0 means all.",
    )
    parser.add_argument("--baseline-min-corr-pairs", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.seed_for_split = 12345

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    out_dir = args.out_dir or (args.base_dir / DATASETS[args.dataset]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, x_original, token_ids, metadata, descriptor_names = load_excel_inputs(
        base_dir=args.base_dir,
        dataset=args.dataset,
        descriptor_start_col=args.descriptor_start_col,
        metadata_mode=args.metadata_mode,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_cache,
    )

    targets = choose_targets(
        descriptor_names=descriptor_names,
        descriptors_np=x_original,
        requested=args.target_cols,
        max_target_cols=args.max_target_cols,
        min_observed=args.min_observed,
    )
    if not targets:
        raise RuntimeError("No target descriptors selected. Lower --min-observed or specify --target-cols.")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        serializable = vars(args).copy()
        serializable["base_dir"] = str(serializable["base_dir"])
        serializable["out_dir"] = str(out_dir)
        json.dump(serializable, f, indent=2)

    print("Selected target descriptors:")
    for j in targets:
        print(f"  - {descriptor_names[j]} (observed={np.isfinite(x_original[:, j]).sum()})")

    all_rows: list[MetricRow] = []
    for mask_frac in args.mask_fracs:
        if not 0 < mask_frac < 1:
            raise ValueError(f"mask fraction must be between 0 and 1: {mask_frac}")
        for seed in args.seeds:
            for j in targets:
                rows = run_one_target(
                    dataset=args.dataset,
                    descriptor_name=descriptor_names[j],
                    target_j=j,
                    x_original=x_original,
                    token_ids=token_ids,
                    metadata=metadata,
                    seed=seed,
                    mask_frac=mask_frac,
                    args=args,
                    device=device,
                    out_dir=out_dir,
                )
                all_rows.extend(rows)
                pd.DataFrame([asdict(r) for r in all_rows]).to_csv(out_dir / "benchmark_results.csv", index=False)

    results = pd.DataFrame([asdict(r) for r in all_rows])
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    print("\nSaved:")
    print(" ", out_dir / "benchmark_results.csv")
    print(" ", out_dir / "benchmark_summary.csv")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
