# CSF dynamics

This project asks whether quantities computed from ID data alone can tell where unseen inputs land relative to the ID classes, well enough to choose a confidence scoring function (CSF) without OOD data. It starts with a pilot: ResNet-18 models trained from scratch on Imagenette, with checkpoints over the epochs.

The protocol, with the gates and the decision rule, is in `../documentation/imagenette_dynamics_protocol_2026-11-10.md`. Stage D has its own protocol, `../documentation/time_and_collapse_protocol_2026-11-12.md`. The paper code is a read-only submodule at `external/ood_systematic`, pinned to commit `563fb63`. This project imports its CSF and NC classes and does not copy or edit them.

## Contents

| Path | Purpose |
|---|---|
| `dynamics/config.py` | Fixed values of the protocols: checkpoints, recipes (A, B, C1 to C3, D1 to D5), stages. |
| `dynamics/data.py` | ID splits, external sets, and the ID-only stand-in sets. |
| `dynamics/model.py` | ResNet-18 and the pooled features of its four stages. |
| `dynamics/train.py` | Trains one run and keeps its checkpoints (HPC). |
| `dynamics/paper_api.py` | Calls the paper's 21 CSFs and 8 NC metrics on our features. |
| `dynamics/scores.py` | All 23 CSFs: the paper's 21, kNN distance, and feature norm. Also the CSF groups of amendment A1. |
| `dynamics/metrics.py` | AUROC, FPR@95, AURC, and AUGRC. |
| `dynamics/descriptors.py` | ID-only descriptors (with CDNV and the other collapse metrics without an inversion), placement descriptors, and the three metric conventions of Stage D. |
| `dynamics/evaluate.py` | Evaluates every checkpoint of a run (HPC). |
| `dynamics/run_stage.py` | Trains and evaluates the runs of a stage (HPC). |
| `dynamics/analyze_e.py` | Stage E gates, rosters, convergence, design-regime check, and analyses E1 to E7 (local). |
| `dynamics/analyze_step0.py` | Step 0: the two rules of amendment A4 on more OOD sets and an earlier layer (local). |
| `dynamics/analyze_checks.py` | Four checks on the Stage E and step 0 results, without new runs (local). |
| `dynamics/analyze_refcheck.py` | Reference check: the decline of each fitted score under three reference sets, and which descriptors index the scores through a reset (local). |
| `dynamics/analyze_selection.py` | Checkpoint selection with ID-only rules against a trivial rule (local). |
| `dynamics/analyze_multimetric.py` | Score selection from several ID-only metrics together, with one run or one recipe family left out (local). |
| `dynamics/rules.py` | Fits the selection rules of Stage C on the earlier runs and writes them to `dynamics/frozen/stage_c_rules.json` (local). |
| `dynamics/analyze_c.py` | Stage C: applies the frozen rules to the new runs and reads the result with the decision rule of amendment A5 (local). |
| `dynamics/interpolation.py` | Lists the training accuracy at each checkpoint of a run, for gate I of Stage D (HPC). |
| `dynamics/analyze_d.py` | Stage D: gates I, P, and M, and the rules for questions Q1 to Q4 (local). |
| `tests/` | Unit tests and an end-to-end test on tiny images. No GPU needed. |
| `requirements-hpc.txt` | The packages that an x9 image built before 2026-11-10 lacks, with exact versions. |

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

Run these from the repository folder inside the x9 container (`x9_imagenet.sif`: Python 3.12, torch 2.6.0+cu124, torchvision 0.21.0+cu124; definition in `Systematic_OOD/container/x9/`). The runs do not need `fd_shifts` or its `.env` file.

The paper's CSF classes import `torch-pca` and `torchmetrics`. An image built from the x9 definition on or after 2026-11-10 has both. An older x9 image lacks them. For an older image, install them once into a folder outside the container and add that folder to `PYTHONPATH`. `--no-deps` keeps pip away from the container's torch:

```bash
python -m pip install --no-deps --target /path/to/csf_dynamics_env -r requirements-hpc.txt
```

Set the paths in every new shell. Keep the output folder outside the repository folder:

```bash
export DYN_ROOT=/path/to/csf_dynamics_runs IMAGENETTE_DIR=/path/to/imagenette2-160 IMAGEWOOF_DIR=/path/to/imagewoof2-160 TEXTURES_DIR=/path/to/dtd
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

Run one command at a time on a GPU. A command can restart: it skips complete runs, resumes a stopped training from its last epoch, and skips evaluated checkpoints. Each run folder `$DYN_ROOT/runs/<run>/` holds `checkpoints/` (about 1 GB), `results_f64/` (one JSON file per checkpoint), `train_log.csv`, `meta.json`, and `manifest.json`.

To evaluate existing runs again without training, add `--skip-train`. This is how the Stage E runs get their float64 results:

```bash
nohup bash -c "python -m dynamics.run_stage --stage e_main --skip-train --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR && python -m dynamics.run_stage --stage e_main --recheck --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR && python -m dynamics.run_stage --stage e_heldout --skip-train --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR" > dyn_stage_e_f64.log 2>&1 &
```

### Step 0: more OOD sets and an earlier layer (amendment A4)

This evaluates the saved checkpoints of the 10 main runs again, with seven more OOD sets and with the head-free CSFs at stages 1 to 3. It trains nothing. The results go to `results_f64_step0/`. Use exactly these set names, because the analysis reads them:

```bash
export OOD=/path/to/openood/folder
```

```bash
nohup python -m dynamics.run_stage --stage e_main --skip-train --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_step0.log 2>&1 &
```

Each extra set contributes at most 5,000 images (`--extra-limit`). Local analysis:

```bash
uv run python -m dynamics.analyze_step0 --results-root dyn_results
```

Pack the results without the checkpoints:

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_e.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs
```

### Stage D: time, collapse, and conventions

Stage D trains 25 new runs: 5 arms (D1 to D5) with 5 seeds each. Each run has 500 epochs and 38 checkpoints. The learning rate falls to 0.01 by epoch 100 and then stays constant. Each run uses the step 0 evaluation, so set `OOD` as above and use exactly the same set names.

First, 2 runs (`D1_seed0` and `D2_seed0`). Read only their training logs and gate I, to find a schedule problem before the other 23 runs:

```bash
nohup python -m dynamics.run_stage --stage d_first --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_d_first.log 2>&1 &
```

Check gate I for these runs. This prints the first checkpoint with 100% accuracy on the training set without augmentation (`t0`), and the accuracy at each checkpoint from epoch 60. The training log does not show this value for D2, because its accuracy column uses augmented images:

```bash
python -m dynamics.interpolation --out-root $DYN_ROOT --runs D1_seed0 D2_seed0
```

Gate I needs `t0` of 200 or less for `D2_seed0`. If the value is larger or the run never reaches 100%, stop and report the listing. The protocol then lowers the floor of the learning rate by an amendment.

Then the full stage. It skips the 2 complete runs:

```bash
nohup python -m dynamics.run_stage --stage d --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_d.log 2>&1 &
```

Gate G3, a second extraction of 3 checkpoints of the seed-0 run of each arm:

```bash
nohup python -m dynamics.run_stage --stage d --recheck --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_d_recheck.log 2>&1 &
```

Pack the Stage D results only:

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_d.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/D*
```

Local analysis. The report goes to `dynamics/outputs/stage_d_report.md`, and the tables go to `dynamics/outputs/stage_d/`:

```bash
uv run python -m dynamics.analyze_d --results-root dyn_results
```

A question whose gate fails gets the verdict "not decided". Each run needs about 1.7 GB for its 38 checkpoints.

### Stage C: frozen selection rules on new recipes (amendment A5)

Stage C trains 15 new runs of 200 epochs: three fresh seeds of recipes A and B, and three seeds of each new recipe (C1, C2, C3). The selection rules are already fixed in `dynamics/frozen/stage_c_rules.json`. Do not build that file again after the first Stage C run: the analysis accepts only the file with the recorded hash. Set `OOD` as for step 0.

First, one run with label smoothing, to measure the time and to check the log:

```bash
nohup python -m dynamics.run_stage --only C2_seed0 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_c_first.log 2>&1 &
```

Then the full stage. It skips the complete run (about 6.5 GPU-hours, estimated):

```bash
nohup python -m dynamics.run_stage --stage c --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_c.log 2>&1 &
```

Pack the Stage C runs. The fresh seeds of A and B have the numbers 5 to 7:

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_c.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/C?_seed? runs/A_seed[567] runs/B_seed[567]
```

Local analysis, after `tar xzf dyn_stage_c.tgz -C dyn_results`. It also needs the earlier `results_f64_refcheck` folders, which are local already. The report goes to `dynamics/outputs/stage_c_report.md`:

```bash
uv run python -m dynamics.analyze_c --results-root dyn_results
```

### Reference check on saved checkpoints

This evaluates saved checkpoints again and trains nothing. It fits every CSF on three reference sets (all training features, a training subset, and one half of the validation set) and tests each fit on the other half of the validation set. It also stores CDNV, the other collapse metrics without an inversion, and the weight norms. The results go to `results_f64_refcheck/`. Set `OOD` as for step 0.

The two first runs of Stage D (about 40 minutes for each run, estimated):

```bash
nohup python -m dynamics.run_stage --only D1_seed0 D2_seed0 --skip-train --suite refcheck --reference-check --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_refcheck.log 2>&1 &
```

Optional, the 10 main runs of the pilot, which give 5 seeds for each recipe (about 4 hours, estimated):

```bash
nohup python -m dynamics.run_stage --stage e_main --skip-train --suite refcheck --reference-check --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_refcheck_pilot.log 2>&1 &
```

Each log line of this evaluation ends with the number of CSF errors and of reference errors. Pack only the new folders:

```bash
cd "$DYN_ROOT" && tar czf dyn_refcheck.tgz runs/*/results_f64_refcheck
```

Local analysis, after `tar xzf dyn_refcheck.tgz -C dyn_results`. The report goes to `dynamics/outputs/refcheck_report.md`:

```bash
uv run python -m dynamics.analyze_refcheck --results-root dyn_results
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
- **Precision (amendment A3).** All CSFs and descriptors run in float64. The paper pipeline fits its feature CSFs in float32. On the Stage E runs, those float32 fits failed gate G3: a second extraction of identical features changed the AUROC of 7 CSFs. `--precision float32` still exists and writes to `results/`. The default writes to `results_f64/`.
- **Two rosters (amendments A1 and A2).** 10 CSFs need no fit, and 13 fit statistics of the training features. The analysis reports the no-fit roster and the full roster. On an unconverged checkpoint, the fitted CSFs are outside their design regime, and the no-fit roster is the reference. A checkpoint is converged if its own validation accuracy is at most 1 point below the final value of its run (amendment A2). So a checkpoint inside a training instability is not converged.
- **Stage D (2026-11-12).** The five arms differ from D1 in one setting each: D2 adds augmentation, D3 adds label smoothing 0.1, D4 lowers the weight decay to 5e-5, and D5 trains without mixed precision. All five use the same schedule. `meta.json` of each run records the recipe. Without `--epochs`, a run uses the length of its recipe (200 for A, B, and C; 500 for D).
- **Reference check (2026-11-12).** The validation set is split once into two class-stratified halves with seed 0. The training subset has the class counts of the first half and uses seed 1. The same split serves every checkpoint and every run with the same classes. From this date, every evaluation also stores the collapse metrics without an inversion (group `spread`) and the weight norms (group `weights`).
- **Epoch 0** is the random start. The analysis reports it and keeps it out of the roster rule and the regret tables.
- **A resumed training run** has the same data order as an uninterrupted run, but not the same augmentation draws.
- **Environment.** The HPC runs use the x9 container, not the paper container (torch 1.13.1, Python 3.10). The tests pass with torch 2.6.0 on Python 3.12, with numpy 1.26 and with numpy 2.5.
