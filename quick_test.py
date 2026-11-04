"""Lightweight repository check; does not train any models."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
required = [
    "train_timeseries.xlsx", "test_timeseries.xlsx",
    "resultados_tabpfn_5_repeticoes.csv",
    "machine_learning_final_reproducao.ipynb",
    "pocopetroleo_exploratory.ipynb",
]
for filename in required:
    assert (ROOT / filename).is_file(), f"Missing file: {filename}"

cols = ["curve_id", "t_index", "Q_time", "R_pressure"]
train = pd.read_excel(ROOT / "train_timeseries.xlsx", usecols=cols)
test = pd.read_excel(ROOT / "test_timeseries.xlsx", usecols=cols)
assert not train.empty and not test.empty
assert train["R_pressure"].notna().any() and test["R_pressure"].notna().any()
train_last = train.groupby("curve_id")["t_index"].max()
test_first = test.groupby("curve_id")["t_index"].min()
shared = train_last.index.intersection(test_first.index)
assert len(shared) > 0, "No shared curves between train and test"
assert (test_first.loc[shared] > train_last.loc[shared]).all(), "Temporal overlap detected"

metrics = pd.read_csv(ROOT / "resultados_tabpfn_5_repeticoes.csv")
assert sorted(metrics["seed"].astype(int).tolist()) == [42, 43, 44, 45, 46]
for col in ["RMSE", "MAE", "R2"]:
    assert pd.to_numeric(metrics[col], errors="coerce").notna().all()

for filename in required[-2:]:
    notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in notebook["cells"])

print(f"PASS: train={len(train):,}, test={len(test):,}, shared curves={len(shared):,}")
print("PASS: five recorded TabPFN runs and two valid notebooks")
print("NOTE: This is an input-integrity test, not a full experimental reproduction.")
