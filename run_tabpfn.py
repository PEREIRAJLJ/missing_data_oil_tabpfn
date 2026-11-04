#!/usr/bin/env python3
"""Reconstructed TabPFN evaluation, NOT the recovered original experiment.

Rebuilds the published temporal features and evaluates five seeded 50k
subsamples on the fixed test set. The historical CSV is never used as input.
Exact equality with historical metrics is NOT guaranteed.
"""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET, GROUP, ORDER, TIME = "R_pressure", "curve_id", "t_index", "Q_time"

def features(train, test):
    a, b = train.copy(), test.copy()
    a["__split"], b["__split"] = "train", "test"
    a["__row_order"], b["__row_order"] = np.arange(len(a)), np.arange(len(b))
    full = pd.concat([a, b], ignore_index=True).sort_values([GROUP, ORDER], kind="mergesort")
    q = full[TIME].astype(float)
    full["Q_time_signed_log"] = np.sign(q) * np.log1p(np.abs(q))
    full["t_index_log"] = np.log1p(full[ORDER].clip(lower=0))
    full["delta_Q_time"] = full.groupby(GROUP, sort=False)[TIME].diff()
    dq = full["delta_Q_time"].astype(float)
    full["delta_Q_time_signed_log"] = np.sign(dq) * np.log1p(np.abs(dq))
    full["Q_time_from_start"] = full[TIME] - full.groupby(GROUP, sort=False)[TIME].transform("first")
    def restore(split):
        return (full[full["__split"] == split].sort_values("__row_order")
                .drop(columns=["__split", "__row_order"]).reset_index(drop=True))
    return restore("train"), restore("test")

def balanced_sample(groups, n, seed):
    """Randomized round-robin sampling across curves, without replacement."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(groups)
    unique, inv = np.unique(ids, return_inverse=True)
    pools = [rng.permutation(np.flatnonzero(inv == i)) for i in range(len(unique))]
    order = rng.permutation(len(unique))
    selected = []
    level = 0
    while len(selected) < min(n, len(ids)):
        added = False
        for i in order:
            if level < len(pools[i]):
                selected.append(int(pools[i][level]))
                added = True
                if len(selected) >= n: break
        if not added: break
        level += 1
    return np.asarray(selected, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--output", type=Path, default=Path("results_tabpfn_reconstructed.csv"))
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    ap.add_argument("--device", choices=["cpu","cuda","auto"], default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="Check preprocessing without loading TabPFN")
    args = ap.parse_args()
    if args.n_train < 1: ap.error("--n-train must be positive")
    train = pd.read_excel(args.data_dir / "train_timeseries.xlsx")
    test = pd.read_excel(args.data_dir / "test_timeseries.xlsx")
    for df in (train, test):
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    for df in (train, test):
        if not {TARGET,GROUP,ORDER,TIME}.issubset(df.columns):
            raise ValueError("Required target/group/time columns are missing")
    tmax = train.groupby(GROUP)[ORDER].max()
    tmin = test.groupby(GROUP)[ORDER].min()
    shared = tmax.index.intersection(tmin.index)
    if len(shared) == 0 or not (tmin.loc[shared] > tmax.loc[shared]).all():
        raise ValueError("Temporal train/test split is invalid")
    train, test = features(train, test)
    cols = [c for c in train if c not in {TARGET,GROUP}]
    X_train, X_test = train[cols].copy(), test[cols].copy()
    # TabPFN handles missing numerical entries; avoid target leakage.
    X_train = X_train.replace([np.inf,-np.inf], np.nan)
    X_test = X_test.replace([np.inf,-np.inf], np.nan)
    y_train = train[TARGET].to_numpy(float)
    y_test = test[TARGET].to_numpy(float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Nonfinite targets; inspect the supplied datasets")
    print(f"Train={len(train)}, test={len(test)}, curves={len(shared)}, features={len(cols)}")
    print("Feature names:",cols)
    if args.dry_run:
        for seed in args.seeds:
            idx=balanced_sample(train[GROUP],args.n_train,seed)
            print(f"seed={seed}: {len(idx)} rows from {train.iloc[idx][GROUP].nunique()} curves")
        print("Dry run passed. No model was trained.")
        return
    from tabpfn import TabPFNRegressor
    rows=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for rep,seed in enumerate(args.seeds,1):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        except ImportError: pass
        idx=balanced_sample(train[GROUP],args.n_train,seed)
        model=TabPFNRegressor(device=args.device, random_state=seed)
        model.fit(X_train.iloc[idx],y_train[idx])
        pred=np.asarray(model.predict(X_test)).ravel()
        curve_rmse=(pd.DataFrame({"curve_id":test[GROUP],"squared":(y_test-pred)**2})
                    .groupby("curve_id")["squared"].mean().pow(.5))
        row=dict(modelo="TabPFN-3",repeticao=rep,seed=seed,n_train=len(idx),
                 RMSE=float(np.sqrt(mean_squared_error(y_test,pred))),
                 MAE=float(mean_absolute_error(y_test,pred)),
                 R2=float(r2_score(y_test,pred)),
                 RMSE_medio_por_curva=float(curve_rmse.mean()),
                 RMSE_mediano_por_curva=float(curve_rmse.median()))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output,index=False)
        print(json.dumps(row))
        del model
    print(f"Saved {args.output}")

if __name__=="__main__":
    main()
