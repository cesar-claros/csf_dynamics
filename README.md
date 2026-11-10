# CSF dynamics

This project asks whether quantities computed from ID data alone can tell where unseen inputs land relative to the ID classes, well enough to choose a confidence scoring function (CSF) without OOD data. It starts with a pilot: ResNet-18 models trained from scratch on Imagenette, with checkpoints over the epochs.

The protocol, with the gates and the decision rule, is in `../documentation/imagenette_dynamics_protocol_2026-11-10.md`. The paper code is a read-only submodule at `external/ood_systematic`, pinned to commit `563fb63`. This project imports its CSF and NC classes and does not copy or edit them.

## Contents

| Path | Purpose |
|---|---|
| `dynamics/config.py` | Fixed values of the protocol: checkpoints, recipes, stages. |
| `dynamics/data.py` | ID splits, external sets, and the ID-only stand-in sets. |
| `dynamics/model.py` | ResNet-18 and the pooled features of its four stages. |
| `dynamics/train.py` | Trains one run and keeps its checkpoints (HPC). |
| `dynamics/paper_api.py` | Calls the paper's 21 CSFs and 8 NC metrics on our features. |
| `dynamics/scores.py` | All 23 CSFs: the paper's 21, kNN distance, and feature norm. Also the CSF groups of amendment A1. |
| `dynamics/metrics.py` | AUROC, FPR@95, AURC, and AUGRC. |
| `dynamics/descriptors.py` | ID-only descriptors and placement descriptors. |
| `dynamics/evaluate.py` | Evaluates every checkpoint of a run (HPC). |
| `dynamics/run_stage.py` | Trains and evaluates the runs of a stage (HPC). |
| `dynamics/analyze_e.py` | Stage E gates, rosters, convergence, design-regime check, and analyses E1 to E7 (local). |
| `tests/` | Unit tests and an end-to-end test on tiny images. No GPU needed. |
| `requirements-hpc.txt` | The two packages that the x9 container lacks, with exact versions. |

## Local setup

```bash
git clone --recurse-submodules <repository-url>
```

```bash
uv sync
```

```bash
uv run pytest
```

## Data on the HPC

Imagenette and Imagewoof, 160 px versions, from the fastai repository. Textures (DTD) is already on the HPC.

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz && tar xzf imagenette2-160.tgz
```

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz && tar xzf imagewoof2-160.tgz
```

`--textures-dir` takes any folder. The code reads every image file below it.

## HPC runs

Run these from `code/` inside the x9 container (`x9_imagenet.sif`: Python 3.12, torch 2.6.0+cu124, torchvision 0.21.0+cu124; definition in `Systematic_OOD/container/x9/`). The runs do not need `fd_shifts` or its `.env` file.

The x9 container has no `torch-pca` and no `torchmetrics`, and the paper's CSF classes import both. Install them once into a folder outside the container. `--no-deps` keeps pip away from the container's torch:

```bash
export DYN_ENV=/path/to/csf_dynamics_env
```

```bash
python -m pip install --no-deps --target "$DYN_ENV" -r requirements-hpc.txt
```

Set the paths in every new shell:

```bash
export PYTHONPATH="$DYN_ENV" DYN_ROOT=/path/to/csf_dynamics_runs IMAGENETTE_DIR=/path/to/imagenette2-160 IMAGEWOOF_DIR=/path/to/imagewoof2-160 TEXTURES_DIR=/path/to/dtd
```

Check the environment. This prints the versions and then `ok`:

```bash
python -c "import torch, torchvision, numpy; from dynamics import paper_api; paper_api.install(); import src.csfs; print(torch.__version__, torchvision.__version__, numpy.__version__, torch.cuda.is_available()); print('ok')"
```

First run, to measure the time and check gates G1 and G2 before the other runs start:

```bash
nohup python -m dynamics.run_stage --only A_seed0 --out-root "$DYN_ROOT" --imagenette-dir "$IMAGENETTE_DIR" --imagewoof-dir "$IMAGEWOOF_DIR" --textures-dir "$TEXTURES_DIR" > dyn_first.log 2>&1 &
```

Stage E, 10 main runs:

```bash
nohup python -m dynamics.run_stage --stage e_main --out-root "$DYN_ROOT" --imagenette-dir "$IMAGENETTE_DIR" --imagewoof-dir "$IMAGEWOOF_DIR" --textures-dir "$TEXTURES_DIR" > dyn_e_main.log 2>&1 &
```

Stage E, 20 held-out-class runs:

```bash
nohup python -m dynamics.run_stage --stage e_heldout --out-root "$DYN_ROOT" --imagenette-dir "$IMAGENETTE_DIR" --imagewoof-dir "$IMAGEWOOF_DIR" --textures-dir "$TEXTURES_DIR" > dyn_e_heldout.log 2>&1 &
```

Gate G3, a second extraction of 3 checkpoints of each seed-0 main run:

```bash
nohup python -m dynamics.run_stage --stage e_main --recheck --out-root "$DYN_ROOT" --imagenette-dir "$IMAGENETTE_DIR" --imagewoof-dir "$IMAGEWOOF_DIR" --textures-dir "$TEXTURES_DIR" > dyn_recheck.log 2>&1 &
```

Run one command at a time on a GPU. A command can restart: it skips complete runs, resumes a stopped training from its last epoch, and skips evaluated checkpoints. Each run folder `$DYN_ROOT/runs/<run>/` holds `checkpoints/` (about 1 GB), `results/` (one JSON file per checkpoint), `train_log.csv`, `meta.json`, and `manifest.json`.

Pack the results without the checkpoints:

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_e.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs
```

## Local analysis

Extract the archive into `dyn_results/`, so that `dyn_results/runs/` exists. Then:

```bash
uv run python -m dynamics.analyze_e --results-root dyn_results
```

The report goes to `dynamics/outputs/stage_e_report.md`, and the tables go to `dynamics/outputs/stage_e/`.

## Smoke run

A short run with small images checks the installation. Use a separate output folder. The runner marks such a run as not a protocol run.

```bash
python -m dynamics.run_stage --only A_seed0 --epochs 2 --image-size 64 --limit 400 --out-root /tmp/dyn_smoke --imagenette-dir "$IMAGENETTE_DIR" --imagewoof-dir "$IMAGEWOOF_DIR" --textures-dir "$TEXTURES_DIR"
```

## Fixed choices of the build

These were fixed on 2026-11-10, before any run. The protocol records them.

- **No tuning of CSF hyperparameters.** The paper pipeline tunes some CSFs on validation data. Here they are fixed: GEN and REN use gamma 0.1 and all classes, NNGuide uses 10% of the training features and 10 neighbors, NCI uses alpha 0.003, PCA RecError keeps 90% of the variance, KPCA RecError uses 200 landmarks and an RBF kernel with gamma 0.2, and MLS, Energy, and GradNorm use temperature 1.
- **Precision.** The logit-only CSFs run in float64, so that a saturated softmax does not create ties. The feature CSFs run in float32, as in the paper pipeline. The descriptors run in float64.
- **Two rosters (amendment A1).** 10 CSFs need no fit, and 13 fit statistics of the training features. The analysis reports the no-fit roster and the full roster. Before a run converges, the fitted CSFs are outside their design regime, and the no-fit roster is the reference. A run has converged at the first checkpoint whose validation accuracy is within 1 point of its final value.
- **Epoch 0** is the random start. The analysis reports it and keeps it out of the roster rule and the regret tables.
- **A resumed training run** has the same data order as an uninterrupted run, but not the same augmentation draws.
- **Environment.** The HPC runs use the x9 container, not the paper container (torch 1.13.1, Python 3.10). The tests pass with torch 2.6.0 on Python 3.12, with numpy 1.26 and with numpy 2.5.
