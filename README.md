# When Well Data Are Missing: Can TabPFN-3 Outperform Classical Machine Learning?

This repository provides the datasets, Python notebooks, evaluation scripts, and recorded results associated with the manuscript submitted to *Computers & Geosciences*. The materials support independent inspection and computational verification of the machine-learning workflow.

## Repository files

| File | Description |
|---|---|
| `machine_learning_final_reproducao.ipynb` | Main experimental notebook: feature processing, temporal evaluation, optimization, classical models, results, and figures |
| `run_tabpfn.py` | TabPFN regression experiment with five random seeds and evaluation metrics |
| `pocopetroleo_exploratory.ipynb` | Additional TabPFN analysis and examples |
| `train_timeseries.xlsx` | Training observations |
| `test_timeseries.xlsx` | Temporal test observations |
| `resultados_tabpfn_5_repeticoes.csv` | TabPFN metrics recorded for the five repetitions |
| `quick_test.py` | Fast checks of input files and notebook structure |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

## Requirements

Python 3.11 is recommended. TabPFN evaluation can require a compatible GPU, substantial RAM/VRAM, model download, and acceptance of upstream model access terms.

Install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verification

Run the input and structure checks:

```bash
python quick_test.py
```

Check TabPFN preprocessing and sample selection without fitting the model:

```bash
python run_tabpfn.py --dry-run
```

Run the TabPFN evaluation with the five seeds (42–46), using 50,000 training observations per run:

```bash
python run_tabpfn.py --device cuda --n-train 50000 --output results_tabpfn_verification.csv
```

Run the main experiment by opening `machine_learning_final_reproducao.ipynb` from the repository root in Jupyter. The exploratory notebook `pocopetroleo_exploratory.ipynb` contains Colab-specific paths that may require adaptation for a local environment.

**Computational note:** Full verification requires executing the scripts and notebooks. Model fitting, hyperparameter optimization, and five TabPFN runs can be time-consuming and computationally demanding; execution time depends on hardware and model access. The supplied CSV contains the recorded TabPFN results for reference. New execution results should be compared with those reported in the manuscript; exact numerical equality is not guaranteed across software versions, model checkpoints, or hardware.

## Data use

The Excel datasets are included for scientific verification. Users are responsible for respecting any applicable third-party data rights and access restrictions.

## Reference

*When Well Data Are Missing: Can TabPFN-3 Outperform Classical Machine Learning?* Manuscript submitted to *Computers & Geosciences*.
