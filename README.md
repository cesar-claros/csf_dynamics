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
| `dynamics/analyze_c2.py` | Second test of the frozen all-metrics rule on three more recipes, amendment A6 (local). |
| `dynamics/analyze_shortlist.py` | Shortlists of scores from the frozen rules against fixed and random shortlists (local). |
| `dynamics/build_subset.py` | Builds the 100-class ImageNet subset of Stage H from parquet files (HPC). |
| `dynamics/analyze_h.py` | Stage H: the gate after phase 1, the freeze of the rules, and the test on phase 2 (local). |
| `dynamics/analyze_valonly.py` | Post hoc check: a rule from the stored validation statistics against one from training statistics (local). |
| `dynamics/analyze_t.py` | Amendment A7: freezes the rules from the wider validation statistics and tests them on four new recipes (local). |
| `dynamics/analyze_u.py` | Amendment A8: tests the frozen all-tiers rule of amendment A7 on four more recipes, with a second fit as a secondary rule (local). |
| `dynamics/analyze_v.py` | Amendments A9 and A10: freezes and tests the rules for the correct-only target of the paper. Part 1 has 10 classes, part 2 has 100 classes, and part 3 has 10 classes without label smoothing (local). |
| `dynamics/analyze_y.py` | Amendment A11: checks if the fixed NCI score stays within 1 point at a weight decay of 5e-3 with 100 classes. It has no rule (local). |
| `dynamics/analyze_r.py` | Amendment A13: applies the frozen rule of amendment A12 part 1 to runs of another architecture (ResNet-50, DenseNet-121), with no new fit (local). |
| `dynamics/draw_recipes.py` | Amendments A14 and A15: draws the recipes, once, into `dynamics/frozen/a14_recipes.json` and `a15_recipes.json` (local). |
| `dynamics/build_sets_parquet.py` | Amendment A15: lists the image counts of the ImageNet classes (HPC), draws the 32 new ID class sets (local, once), and builds their folders from the parquet files, with gate D (HPC). |
| `dynamics/analyze_s.py` | Amendment A15: fits and freezes a rule on runs of several ID sets and a choice from the recipe alone, and tests them on new class sets with the class set as the unit (local). |
| `dynamics/build_class_sets.py` | Amendment A14: draws the four new ID class sets (local, once) and builds their folders from the 100-class folder, with gate D (HPC). |
| `dynamics/analyze_p.py` | Amendment A14: applies the frozen rule of amendment A12 part 1 to recipes drawn at random, on Imagenette and on new ID classes, with no new fit (local). |
| `dynamics/analyze_z.py` | Amendment A12: freezes and tests the rule from the statistics of each score, part 1 with 10 classes and part 2 with 100 classes, on the far sets and on two secondary targets (local). |
| `dynamics/analyze_loro.py` | Exploratory check, post hoc: rules from each set of ID-only metrics with one recipe left out, against choices with no metric (local). |
| `dynamics/analyze_factors.py` | Exploratory check, post hoc: how much each recipe setting changes the order of the scores, on matched pairs of recipes (local). |
| `dynamics/analyze_augrc.py` | Exploratory check, post hoc: the selection results with two other conventions for the misclassified ID images, the correct-only convention of the paper and the failure convention of the stored AUGRC (local). |
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

### Second test of the all-metrics rule (amendment A6)

Nine new runs of 200 epochs: three seeds of each recipe C4, C5, and C6. The rule under test is already in `dynamics/frozen/stage_c_rules.json`, and a second rule is in `dynamics/frozen/stage_c2_rules.json`. Do not build either file again. Set `OOD` as for step 0. About 4.2 GPU-hours, estimated:

```bash
nohup python -m dynamics.run_stage --stage c2 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_c2.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_c2.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/C[456]_seed?
```

Local analysis, after `tar xzf dyn_stage_c2.tgz -C dyn_results`. The report goes to `dynamics/outputs/stage_c2_report.md`:

```bash
uv run python -m dynamics.analyze_c2 --results-root dyn_results
```

### Amendment A7: a rule from validation statistics (10 classes)

The plan is `../documentation/validation_statistics_amendment_2026-11-17.md`.

**1. Light pass on the saved checkpoints of the 36 earlier 10-class runs.** It reads the Imagenette validation set only, and it writes `results_f64_valstats/` in each run folder. It reports a run whose checkpoints are missing and continues:

```bash
nohup python -m dynamics.run_stage --stage a7_train --valstats-only --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR > dyn_a7_valstats.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_a7_valstats.tgz runs/*/results_f64_valstats
```

**2. Freeze, local, once.** After `tar xzf dyn_a7_valstats.tgz -C dyn_results`. The hash that it prints goes into the amendment and into `RULES_T_SHA256` of `dynamics/analyze_t.py`, and both go into a public commit before step 3:

```bash
uv run python -m dynamics.analyze_t freeze --results-root dyn_results
```

**3. The 8 runs with the new recipes,** about 3.7 GPU-hours. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage t --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_t.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_t.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/T?_seed?
```

```bash
uv run python -m dynamics.analyze_t test --results-root dyn_results
```

### Amendment A8: the all-tiers rule on four more recipes (10 classes)

The plan is `../documentation/all_tiers_test_amendment_2026-11-17.md`. The rule under test is the all-tiers rule in `dynamics/frozen/stage_t_rules.json`, as amendment A7 froze it. The second fit in `dynamics/frozen/stage_u_rules.json` is a secondary rule. It is frozen, and its hash is in `RULES_U_SHA256` of `dynamics/analyze_u.py`.

**1. The 8 runs with the new recipes,** about 3.7 GPU-hours. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage u --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_u.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_u.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/U?_seed?
```

**2. The test, local.** After `tar xzf dyn_stage_u.tgz -C dyn_results`. It stops if one of the two rules files does not have its frozen hash:

```bash
uv run python -m dynamics.analyze_u test --results-root dyn_results
```

**3. Exploratory check of the convention for the misclassified ID images (post hoc).** The planned tests use the AUROC of all ID images against the unseen images. This check takes two other targets from the stored values of the runs of the five tests, and it writes no rules file:

```bash
uv run python -m dynamics.analyze_augrc --results-root dyn_results
```

### Amendment A9: selection with the correct-only target (10 and 100 classes)

The plan is `../documentation/correct_only_selection_amendment_2026-11-18.md`. The target is the AUROC of the correct ID validation images against the OOD images. From this build on, every evaluation stores the correct-only values (`auroc_correct`, `fpr95_correct`, `aurc_correct`, `augrc_correct`) next to the old values. The two rules files are frozen: `dynamics/frozen/stage_v_rules.json` (part 1) and `dynamics/frozen/stage_w_rules.json` (part 2). Their hashes are in `dynamics/analyze_v.py`.

**1. Part 1, 10 classes: the 12 runs of stage `v`,** about 5.6 GPU-hours. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage v --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_v.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_v.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/V?_seed?
```

**2. Part 2, 100 classes: the 8 runs of stage `w`,** about 13 GPU-hours. It needs the subset folder of Stage H:

```bash
nohup python -m dynamics.run_stage --stage w --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_w.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_w.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/W?_seed?
```

**3. The tests, local.** After `tar xzf dyn_stage_v.tgz -C dyn_results` and the same for part 2. Each test stops if a rules file does not have its frozen hash, or if the stored target and the target from the old values differ (gate M):

```bash
uv run python -m dynamics.analyze_v test --part 1 --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_v test --part 2 --results-root dyn_results
```

### Amendment A10: selection without label smoothing (10 classes)

The plan is `../documentation/no_label_smoothing_amendment_2026-11-18.md`. The scope is recipes without label smoothing, and the target is that of amendment A9. The rules file `dynamics/frozen/stage_x_rules.json` is frozen, and its hash is in `dynamics/analyze_v.py`.

**1. The 12 runs of stage `x`,** about 5.6 GPU-hours. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage x --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_x.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_x.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/X?_seed?
```

**2. The test, local.** After `tar xzf dyn_stage_x.tgz -C dyn_results`:

```bash
uv run python -m dynamics.analyze_v test --part 3 --results-root dyn_results
```

**3. Exploratory check of the recipe settings (post hoc).** It compares pairs of recipes that differ in one setting (augmentation, weight decay, label smoothing, or learning rate), on the 10-class runs with 200 epochs:

```bash
uv run python -m dynamics.analyze_factors --results-root dyn_results
```

### Amendment A11: the fixed score at high weight decay (100 classes)

The plan is `../documentation/fixed_score_high_weight_decay_amendment_2026-11-19.md`. The check has no rule. The fixed choices are frozen in `dynamics/frozen/stage_y_choices.json`, and the hash is in `dynamics/analyze_y.py`.

**1. The 6 runs of stage `y`,** about 9.6 GPU-hours. It needs the subset folder of Stage H:

```bash
nohup python -m dynamics.run_stage --stage y --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_y.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_y.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/Y?_seed?
```

**2. The check, local.** After `tar xzf dyn_stage_y.tgz -C dyn_results`:

```bash
uv run python -m dynamics.analyze_y test --results-root dyn_results
```

**3. Exploratory read with one recipe left out (post hoc).** The plan and the result are in `../documentation/metrics_leave_one_recipe_out_2026-11-20.md`. The default stages are the 100-class ones. For the 10-class runs, add `--stages c c2 t u v x --out dynamics/outputs/loro_ten_report.md`:

```bash
uv run python -m dynamics.analyze_loro --results-root dyn_results
```

### Amendment A12: a rule from the statistics of each score (10 and 100 classes)

The plan is `../documentation/score_statistics_rule_amendment_2026-11-20.md`. The two rules files are frozen: `dynamics/frozen/stage_z_rules.json` (part 1) and `dynamics/frozen/stage_k_rules.json` (part 2). Their hashes are in `dynamics/analyze_z.py`.

**1. Part 1, 10 classes: the 16 runs of stage `z`,** about 7.5 GPU-hours. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage z --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_z.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_z.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/Z?_seed?
```

**2. Part 2, 100 classes: the 8 runs of stage `k`,** about 12.8 GPU-hours. It needs the subset folder of Stage H:

```bash
nohup python -m dynamics.run_stage --stage k --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_k.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_k.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/K?_seed?
```

**3. The tests, local.** After `tar xzf dyn_stage_z.tgz -C dyn_results` and the same for part 2:

```bash
uv run python -m dynamics.analyze_z test --part 1 --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_z test --part 2 --results-root dyn_results
```

### Amendment A13: the frozen rule on other architectures (10 classes)

The plan is `../documentation/other_architecture_replication_amendment_2026-11-21.md`. The recipes R1 to R8 (ResNet-50) and N1 to N8 (DenseNet-121) are the recipes of stage `z` with another network. Nothing is fitted: the analysis reads the frozen file `dynamics/frozen/stage_z_rules.json`. Each run is evaluated at 8 checkpoints. A run saves its 23 checkpoints, so stage `r` needs about 36 GB of disk and stage `n` about 12 GB.

**1. The two first runs.** They give the time of a run and show if every score has a value on the larger features. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage r_first --eval-epochs 20 40 60 80 100 120 160 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_r_first.log 2>&1 &
```

```bash
nohup python -m dynamics.run_stage --stage n_first --eval-epochs 20 40 60 80 100 120 160 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_n_first.log 2>&1 &
```

**2. The two stages,** 16 runs each. A run that is complete is skipped, so the first runs are not repeated:

```bash
nohup python -m dynamics.run_stage --stage r --eval-epochs 20 40 60 80 100 120 160 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_r.log 2>&1 &
```

```bash
nohup python -m dynamics.run_stage --stage n --eval-epochs 20 40 60 80 100 120 160 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_n.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_r.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/R?_seed?
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_n.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/N?_seed?
```

**3. The analysis, local.** After `tar xzf dyn_stage_r.tgz -C dyn_results` and the same for stage `n`:

```bash
uv run python -m dynamics.analyze_r --part 1 --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_r --part 2 --results-root dyn_results
```

### Amendment A14: recipes drawn at random and new ID classes (10 classes)

The plan is `../documentation/random_recipes_new_classes_amendment_2026-11-22.md`. Nothing is fitted: the analysis reads the frozen file `dynamics/frozen/stage_z_rules.json`. The network is ResNet-18, and each recipe has one run.

- **Recipes.** P01 to P48 (part 1) and Q01 to Q48 (part 2) are in `dynamics/frozen/a14_recipes.json`. `dynamics/draw_recipes.py` drew them once.
- **Class sets.** The four sets of part 2 are in `dynamics/frozen/a14_class_sets.json`. Each has 10 ImageNet classes that are not Imagenette or Imagewoof classes, with 9,469 training and 3,925 validation images.
- **Checkpoints.** `--save-epochs 200` keeps only the final checkpoint, about 45 MB for a run. Without it a run keeps 23 checkpoints, about 1 GB.
- **Time.** About 7.5 minutes for a run, so about 6 GPU-hours for each part.
- **The ID folder is checked.** `run_stage` stops before any training if the classes of the ID folder are not those of the runs. So a stage of part 2 runs only with the folder of its class set.

**1. Build the four class sets, once.** It needs the 100-class folder of Stage H. The script copies about 53,600 small images (use `--link` for symbolic links). It prints the hash of the frozen file, and the image counts, the hash of the split, and gate D for each set:

```bash
export A14_SETS_DIR=/path/to/a14_class_sets
```

```bash
python -m dynamics.build_class_sets --source $IMAGENET100_DIR --out $A14_SETS_DIR
```

**2. The first run of part 2.** It shows that the new folders work and gives the accuracy on new classes. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage q_first --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $A14_SETS_DIR/set1 --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_q_first.log 2>&1 &
```

**3. Part 1: the 48 runs of stage `p`,** on Imagenette:

```bash
nohup python -m dynamics.run_stage --stage p --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_p.log 2>&1 &
```

**4. Part 2: the four stages `q1` to `q4`,** 12 runs each, each with the folder of its class set. The first run is not repeated:

```bash
nohup python -m dynamics.run_stage --stage q1 --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $A14_SETS_DIR/set1 --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_q1.log 2>&1 &
```

```bash
nohup python -m dynamics.run_stage --stage q2 --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $A14_SETS_DIR/set2 --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_q2.log 2>&1 &
```

```bash
nohup python -m dynamics.run_stage --stage q3 --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $A14_SETS_DIR/set3 --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_q3.log 2>&1 &
```

```bash
nohup python -m dynamics.run_stage --stage q4 --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $A14_SETS_DIR/set4 --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_q4.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_p.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/P??_seed0
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_q.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/Q??_seed0
```

**5. The analysis, local.** After `tar xzf dyn_stage_p.tgz -C dyn_results` and the same for stage `q`:

```bash
uv run python -m dynamics.analyze_p --part 1 --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_p --part 2 --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_p --together --results-root dyn_results
```

The analysis stops if a run of the stage has no results. A recipe is not replaced. If the training of a run failed (for example, a loss that is not finite), add `--count-missing-as-failed`: the recipe then counts as a failure of gate T.

### Amendment A15: a rule fitted on several ID sets against a choice from the recipe (10 classes)

The plan is `../documentation/several_id_sets_rule_amendment_2026-11-23.md`. The rule under test and the choice by cell are in the frozen file `dynamics/frozen/stage_s_rules.json`. They were fitted on the 186 earlier ResNet-18 runs. The test has 32 new class sets with 4 recipes each (S001 to S128, in `dynamics/frozen/a15_recipes.json`), and the class set is the unit. Each run takes about 7.5 minutes, so the stage takes about 16 GPU-hours.

**1. The image counts of the ImageNet classes, on the HPC.** The draw of the class sets needs them. The script reads the labels and the file names of the parquet files, not the images. It needs `pyarrow`, as the build of the subset of Stage H does. Set `PARQUET_DIR` as for Stage H:

```bash
python -m dynamics.build_sets_parquet --count --parquet-dir $PARQUET_DIR --out imagenet_class_counts.json
```

Put `imagenet_class_counts.json` in the local `code/` folder.

**2. The draw of the class sets, local, once.** It writes `dynamics/frozen/a15_class_sets.json`. Then set `CLASS_SETS_SHA256` in `dynamics/analyze_s.py` to the printed hash, and commit and push before any run:

```bash
uv run python -m dynamics.build_sets_parquet --draw --counts imagenet_class_counts.json
```

**3. Build the 32 folders, on the HPC,** after a pull. The script reads the images of 320 classes from the parquet files and writes about 430,000 small images (about 5 GB). It prints the hash of the frozen file, and the image counts, the hash of the split, and gate D for each set:

```bash
export A15_SETS_DIR=/path/to/a15_class_sets
```

```bash
nohup python -m dynamics.build_sets_parquet --parquet-dir $PARQUET_DIR --out $A15_SETS_DIR > dyn_s_build.log 2>&1 &
```

**4. The first run.** `--id-root` gives each run the folder of its class set (`set01` to `set32`). Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage s_first --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-root $A15_SETS_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_s_first.log 2>&1 &
```

**5. The 128 runs of stage `s`.** The first run is not repeated:

```bash
nohup python -m dynamics.run_stage --stage s --save-epochs 200 --eval-epochs 200 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-root $A15_SETS_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_s.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_s.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/S???_seed0
```

**6. The analysis, local.** After `tar xzf dyn_stage_s.tgz -C dyn_results`:

```bash
uv run python -m dynamics.analyze_s test --results-root dyn_results
```

The analysis stops if a run of the stage has no results. If the training of a run failed, add `--count-missing-as-failed`.

**The fit, for the record (local, done).** `freeze` wrote the frozen file once and does not write it again. `read` is the check of the fit: it leaves one class set of amendment A14 out at a time and gives the numbers of the read in the plan:

```bash
uv run python -m dynamics.analyze_s read --results-root dyn_results
```

### Stage H: 100 classes

Stage H repeats the selection tests with a 100-class subset of ImageNet-1k. The plan is `../documentation/imagenet100_selection_protocol_2026-11-14.md`.

**1. Build the subset, once.** It needs `pyarrow`. Check it, and install it into the overlay folder if it is missing:

```bash
python -c "import pyarrow; print(pyarrow.__version__)"
```

```bash
python -m pip install --no-deps --target $DYN_ENV pyarrow
```

Set the folder of the parquet files and a new folder for the subset, then build. The script prints the hash of the class list and the image counts, and it writes the list to `dynamics/frozen/imagenet100_classes.txt`:

```bash
export PARQUET_DIR=/work/cniel/sw/FD_Shifts/project/datasets/imagenet1k_raw/data IMAGENET100_DIR=/path/to/imagenet100-160
```

```bash
nohup python -m dynamics.build_subset --parquet-dir $PARQUET_DIR --imagenette-dir $IMAGENETTE_DIR --imagewoof-dir $IMAGEWOOF_DIR --out $IMAGENET100_DIR > dyn_h_build.log 2>&1 &
```

**2. First run,** to measure the time and the memory. Set `OOD` as for step 0:

```bash
nohup python -m dynamics.run_stage --stage h_first --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_h_first.log 2>&1 &
```

**3. Phase 1,** 10 runs. It skips the complete first run:

```bash
nohup python -m dynamics.run_stage --stage h1 --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_h1.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_h1.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/H[0-4]_seed?
```

**4. Gate and freeze, local.** Do this once, and before any phase 2 run. The hash that `freeze` prints goes into the protocol and into `RULES_H_SHA256` of `dynamics/analyze_h.py`:

```bash
uv run python -m dynamics.analyze_h gate --results-root dyn_results
```

```bash
uv run python -m dynamics.analyze_h freeze --results-root dyn_results
```

**5. Phase 2,** 6 runs, only if the gate finds room to select:

```bash
nohup python -m dynamics.run_stage --stage h2 --eval-epochs 10 20 30 40 50 60 80 100 --suite step0 --extra-set inaturalist=$OOD/inaturalist --extra-set places=$OOD/places --extra-set sun=$OOD/sun --extra-set openimage_o=$OOD/openimage_o --extra-set ssb_hard=$OOD/ssb_hard --extra-set ninco=$OOD/ninco --extra-set imagenet_o=$OOD/imagenet_o --out-root $DYN_ROOT --id-dir $IMAGENET100_DIR --imagewoof-dir $IMAGEWOOF_DIR --textures-dir $TEXTURES_DIR > dyn_h2.log 2>&1 &
```

```bash
cd "$DYN_ROOT" && tar czf dyn_stage_h2.tgz --exclude=checkpoints --exclude=features --exclude=last.pt runs/H[567]_seed?
```

```bash
uv run python -m dynamics.analyze_h test --results-root dyn_results
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
