# Stage E: exploration report

Stage E makes no claim. Every number here is descriptive.

## Runs

- `e_main`: 10 of 10 runs
- `e_heldout`: 20 of 20 runs
- Precision of the fitted CSFs: float64.
- Last epoch in the results: 200. Epoch 0 is the random start. It is outside the roster rule and the tables below.

## Gates

```
                                                         gate status                                                                                                                                     detail
            G1, regime A: 100% training accuracy by epoch 100   pass                                                                            A_seed0: 25; A_seed1: 30; A_seed2: 25; A_seed3: 25; A_seed4: 25
 G1, regime B: validation accuracy of at least 85% at the end   pass                                                             B_seed0: 0.898; B_seed1: 0.902; B_seed2: 0.898; B_seed3: 0.892; B_seed4: 0.896
G2, regime A: var_collapse falls after 100% training accuracy   pass A_seed0: 0.2115 to 0.01999; A_seed1: 0.1959 to 0.02551; A_seed2: 0.2095 to 0.01609; A_seed3: 0.2038 to 0.02675; A_seed4: 0.2048 to 0.01374
                       G3, determinism of a second extraction   pass                                                                                                  6 checkpoints, largest AUROC gap 0.00e+00
    G4, MSR, MLS, and Energy equal an independent computation   pass                                                                                                                 largest AUROC gap 0.00e+00
                  G5, Imagenette and Imagewoof share no class   pass                                        imagenette train 9469, imagenette val 3925, imagewoof train 9025, imagewoof val 3929, textures 5640
```

Largest relative gap between our class-mean NC metrics and the paper's, at the penultimate layer: 2.18e-06.

## Rosters and convergence

- Full roster, 23 CSFs: MSR, MLS, Energy, GEN, REN, PCE, GE, PE, GradNorm, pNML, CTM, fDBD, NNGuide, NeCo, Maha, MahaPP, NCI, ViM, Residual, PCA_RecError, KPCA_RecError, kNN, FeatureNorm.
- No-fit roster, 10 CSFs: MSR, MLS, Energy, GEN, REN, PCE, GE, PE, GradNorm, FeatureNorm.
- No CSF is removed by the roster rule.

A checkpoint is converged if its validation accuracy is at most 1 point below the final value of its run. On the other checkpoints, the fitted CSFs are outside their design regime, and the no-fit roster is the reference. A disturbed checkpoint is an unconverged one after the first converged one. Main runs:

```
    run  first_converged_epoch  converged_checkpoints disturbed_checkpoints
A_seed0                     25                     11                    60
A_seed1                     40                      8                80 100
A_seed2                     25                     11                    60
A_seed3                     40                      9                    80
A_seed4                     25                     11                    60
B_seed0                    140                      4                      
B_seed1                    160                      3                      
B_seed2                    160                      3                      
B_seed3                    140                      4                      
B_seed4                    160                      3                      
```

Also in `stage_e/convergence.csv`.

## E1: is there something to predict?

For each external set and roster: the CSF that is best on average over the main runs (B1), its regret in AUROC points, the regret of a random CSF, how many different CSFs win, and the mean gap to the best CSF of the full roster (`cost_vs_full`). `converged_runs` is the share of the runs that have converged at that epoch.

```
      set  epoch roster  runs  converged_runs           b1  b1_regret  random_regret  distinct_winners  cost_vs_full
imagewoof      1   full    10             0.0  FeatureNorm      0.052         11.303                 2         0.000
imagewoof      1 no_fit    10             0.0  FeatureNorm      0.000          4.838                 1         0.052
imagewoof     10   full    10             0.0          REN      2.848          7.231                 5         0.000
imagewoof     10 no_fit    10             0.0          REN      0.878          1.807                 5         1.970
imagewoof     50   full    10             0.5          NCI      1.326          5.279                 5         0.000
imagewoof     50 no_fit    10             0.5       Energy      0.335          1.120                 4         1.010
imagewoof    100   full    10             0.4          REN      0.321          3.996                 5         0.000
imagewoof    100 no_fit    10             0.4          REN      0.093          0.771                 4         0.228
imagewoof    200   full    10             1.0          NCI      0.725          3.374                 4         0.000
imagewoof    200 no_fit    10             1.0          GEN      0.290          1.712                 5         0.584
 textures      1   full    10             0.0       MahaPP      2.691         16.853                 4         0.000
 textures      1 no_fit    10             0.0  FeatureNorm      0.061          5.377                 2        17.979
 textures     10   full    10             0.0 PCA_RecError      0.000         14.861                 1         0.000
 textures     10 no_fit    10             0.0     GradNorm      0.000          7.505                 1         9.307
 textures     50   full    10             0.5 PCA_RecError      0.094         12.989                 3         0.000
 textures     50 no_fit    10             0.5     GradNorm      0.000          6.107                 1        11.370
 textures    100   full    10             0.4 PCA_RecError      0.615         11.795                 2         0.000
 textures    100 no_fit    10             0.4     GradNorm      0.000          4.856                 1        10.534
 textures    200   full    10             1.0         pNML      1.940         10.587                 4         0.000
 textures    200 no_fit    10             1.0           PE      1.676          2.902                 3        12.575
```

## E2: trajectories

Full tables: `stage_e/auroc_trajectories.csv`, `descriptor_trajectories.csv`, and `placement_trajectories.csv`. The five best CSFs at the last epoch, mean and spread over seeds:

```
      set recipe          csf        group   mean   std
    blend      A  FeatureNorm       no_fit 75.654 0.962
    blend      A     GradNorm       no_fit 74.872 0.819
    blend      A       Energy       no_fit 74.387 0.526
    blend      A          REN       no_fit 74.338 0.514
    blend      A          GEN       no_fit 74.288 0.498
    blend      B          kNN fit_relative 79.613 0.249
    blend      B          GEN       no_fit 79.253 0.481
    blend      B          REN       no_fit 79.204 0.500
    blend      B          CTM    fit_plain 79.153 0.306
    blend      B       Energy       no_fit 79.125 0.526
id_errors      A          MSR       no_fit 87.916 0.494
id_errors      A          kNN fit_relative 87.898 0.499
id_errors      A          PCE       no_fit 87.898 0.494
id_errors      A          NCI fit_absolute 87.830 0.461
id_errors      A           PE       no_fit 87.775 0.520
id_errors      B          kNN fit_relative 92.175 0.396
id_errors      B          CTM    fit_plain 92.109 0.352
id_errors      B          PCE       no_fit 92.084 0.502
id_errors      B          MSR       no_fit 92.067 0.512
id_errors      B           PE       no_fit 92.047 0.494
imagewoof      A  FeatureNorm       no_fit 73.522 1.136
imagewoof      A          NCI fit_absolute 73.426 0.437
imagewoof      A     GradNorm       no_fit 73.152 1.099
imagewoof      A       Energy       no_fit 73.143 0.825
imagewoof      A          REN       no_fit 73.138 0.807
imagewoof      B          kNN fit_relative 81.523 0.137
imagewoof      B      NNGuide fit_relative 81.105 0.786
imagewoof      B          GEN       no_fit 80.562 0.596
imagewoof      B          NCI fit_absolute 80.552 0.434
imagewoof      B           PE       no_fit 80.520 0.450
  shuffle      A          CTM    fit_plain 72.343 0.613
  shuffle      A         Maha    fit_plain 72.067 0.834
  shuffle      A          ViM fit_absolute 71.699 0.981
  shuffle      A           PE       no_fit 71.615 0.824
  shuffle      A          MLS       no_fit 71.605 0.901
  shuffle      B          CTM    fit_plain 77.268 0.709
  shuffle      B          kNN fit_relative 76.960 0.847
  shuffle      B         fDBD    fit_plain 74.984 1.205
  shuffle      B          MSR       no_fit 74.287 0.813
  shuffle      B          PCE       no_fit 74.267 0.835
 textures      A PCA_RecError fit_relative 81.170 5.909
 textures      A         pNML    fit_plain 81.079 1.497
 textures      A     Residual fit_absolute 80.741 3.699
 textures      A          ViM fit_absolute 79.921 1.040
 textures      A         Maha    fit_plain 79.499 2.199
 textures      B       MahaPP    fit_plain 96.134 0.294
 textures      B         pNML    fit_plain 96.132 0.297
 textures      B          ViM fit_absolute 95.569 0.249
 textures      B         NeCo fit_absolute 94.861 0.425
 textures      B         Maha    fit_plain 94.690 0.363
```

## Design regime of the fitted CSFs

Mean regret of each CSF group on the external sets, on the unconverged and on the converged checkpoints, and the difference (`excess`). A larger excess for the fitted groups than for `no_fit` means that the fitted CSFs lose more on unconverged checkpoints.

```
       group  unconverged  converged  excess  csfs
fit_absolute       16.444      7.466   8.979     4
   fit_plain       12.349      6.235   6.114     5
fit_relative        6.729      5.911   0.818     4
      no_fit       12.101      9.255   2.846    10
```

Per CSF: `stage_e/design_regime.csv`.

## E3: placement to ranking

Out-of-run R2 of the logit of each CSF's AUROC. All models read the set and the regime. A no-fit CSF uses every cell, and a fitted CSF uses only the converged cells. `clock` is a linear model of the clock group. `epoch_mean` is the strict clock: each run is predicted by the mean of the other runs of the same regime at the same epoch and set. Median over the CSFs of each group:

```
       group  base  clock  epoch_mean  placement  both
fit_absolute 0.732  0.798       0.984      0.944 0.944
   fit_plain 0.737  0.779       0.985      0.975 0.975
fit_relative 0.841  0.840       0.962      0.961 0.961
      no_fit 0.312  0.730       0.856      0.916 0.916
```

Per CSF: `stage_e/e3_link_two.csv`.

## E4: ID-only descriptors to placement

Out-of-run R2 of each placement descriptor of an external set, from a linear model of the clock group (`clock`), from the strict clock (`epoch_mean`, the mean of the other runs of the same regime at the same epoch), from the geometric ID-only descriptors, and from both. The geometric descriptors carry information beyond the time only if they beat `epoch_mean`:

```
      set         placement  cells  clock  epoch_mean  geometry  both
imagewoof   centroid_offset    220  0.857       0.902     0.901 0.903
imagewoof  class_cosine_gap    220  0.879       0.929     0.937 0.937
imagewoof      margin_ratio    220  0.447       0.367     0.449 0.437
imagewoof    neighbor_ratio    220  0.803       0.958     0.971 0.972
imagewoof        norm_ratio    220  0.883       0.912     0.896 0.896
imagewoof outside_share_gap    220  0.616       0.865     0.827 0.831
 textures   centroid_offset    220  0.671       0.856     0.784 0.788
 textures  class_cosine_gap    220  0.740       0.921     0.928 0.927
 textures      margin_ratio    220  0.800       0.872     0.898 0.899
 textures    neighbor_ratio    220  0.713       0.938     0.920 0.934
 textures        norm_ratio    220  0.955       0.965     0.967 0.967
 textures outside_share_gap    220  0.599       0.896     0.890 0.890
```

## E5: stand-ins

Spearman correlation between the CSF ranking on the stand-in and on the external set of its type, and the regret of the CSF that the stand-in ranks first, per roster and phase:

```
 stand_in roster       phase  spearman  regret  cells
    blend   full   converged     0.819   0.708     67
    blend   full unconverged     0.721   2.493    153
    blend no_fit   converged     0.549   0.452     67
    blend no_fit unconverged     0.500   1.790    153
id_errors   full   converged     0.512   1.238     67
id_errors   full unconverged     0.477   4.376    153
id_errors no_fit   converged    -0.147   0.661     67
id_errors no_fit unconverged    -0.288   3.891    153
  shuffle   full   converged    -0.383  15.235     67
  shuffle   full unconverged    -0.089  17.177    153
  shuffle no_fit   converged    -0.156   4.982     67
  shuffle no_fit unconverged     0.081   4.198    153
```

Held-out classes at the last epoch (ranking averaged over the held-out models of the regime, tested on the main runs):

```
roster recipe  classes  spearman  regret  class_agreement_min  class_agreement_median  cells
  full      A     10.0     0.415   1.796               -0.039                   0.715      5
  full      B     10.0     0.768   2.321                0.655                   0.897      5
no_fit      A     10.0     0.399   0.507                0.115                   0.873      5
no_fit      B     10.0     0.716   0.046                0.297                   0.952      5
```

Regret against the number of held-out classes at the last epoch:

```
recipe  epoch  classes  regret roster
     A    200        1   3.380   full
     A    200        2   3.365   full
     A    200        3   3.198   full
     A    200        5   2.711   full
     A    200       10   1.796   full
     B    200        1   1.626   full
     B    200        2   1.883   full
     B    200        3   2.006   full
     B    200        5   2.224   full
     B    200       10   2.321   full
     A    200        1   0.501 no_fit
     A    200        2   0.485 no_fit
     A    200        3   0.472 no_fit
     A    200        5   0.505 no_fit
     A    200       10   0.507 no_fit
     B    200        1   0.091 no_fit
     B    200        2   0.077 no_fit
     B    200        3   0.066 no_fit
     B    200        5   0.055 no_fit
     B    200       10   0.046 no_fit
```

## E6: agreement between metrics

Converged cells only, full roster.

```
      set         pair  kendall  same_best  cells
imagewoof  augrc, aurc    0.887      0.851     67
imagewoof auroc, augrc    0.748      0.433     67
imagewoof  auroc, aurc    0.707      0.433     67
imagewoof auroc, fpr95    0.689      0.448     67
imagewoof fpr95, augrc    0.565      0.179     67
imagewoof  fpr95, aurc    0.499      0.179     67
 textures  augrc, aurc    0.906      0.925     67
 textures auroc, augrc    0.841      0.851     67
 textures  auroc, aurc    0.764      0.806     67
 textures auroc, fpr95    0.749      0.701     67
 textures fpr95, augrc    0.641      0.582     67
 textures  fpr95, aurc    0.568      0.537     67
```

## E7: seed noise

Spread over seeds at the last epoch, in AUROC points:

```
recipe       set  epoch  seeds  auroc_sd  gap_mean  gap_sd
     A imagewoof    200      5     1.181     0.293   0.146
     A  textures    200      5     1.975     3.709   1.599
     B imagewoof    200      5     0.687     0.457   0.380
     B  textures    200      5     0.720     0.011   0.010
```

Materiality threshold for the freeze: 3.20 points (the larger of 1.0 and twice the largest spread of the best-to-second gap).
