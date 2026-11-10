"""The paper's CSF and NC classes, called with the features of our own model.

The paper code is the read-only submodule external/ood_systematic. Its classes read
the last layer through `src.utils.get_model_and_last_layer` and log through
`fd_shifts.logger`. `install` replaces both with small stand-ins, so the classes
run the same way on the HPC and on a laptop, with no fd_shifts configuration.

Hyperparameters that the paper tunes on validation data are fixed here (see
`paper_scores`). Tuning 23 checkpoints of every run would be slow, and the random
search would break the determinism gate G3.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PAPER_CODE = Path(__file__).resolve().parents[1] / "external" / "ood_systematic"
STUDY = "devries"
NC_METRICS = (
    "var_collapse", "equiangular_uc", "equiangular_wc", "equinorm_uc",
    "equinorm_wc", "max_equiangular_uc", "max_equiangular_wc", "self_duality",
)
LOGIT_ONLY = ("MSR", "MLS", "Energy", "GEN", "REN", "PCE", "GE", "PE")
FEATURE_CSFS = ("GradNorm", "pNML", "CTM", "fDBD", "NNGuide", "NeCo", "Maha",
                "MahaPP", "NCI", "ViM", "Residual", "PCA_RecError", "KPCA_RecError")
PAPER_CSFS = (*LOGIT_ONLY, *FEATURE_CSFS)


def _last_layer(module, study_name, return_model=True):
    if return_model:
        return module.model, module.w, module.b
    return module.w, module.b


def install() -> None:
    """Make `src.csfs` and `src.neural_collapse` importable.

    Raises:
        RuntimeError: If the submodule is not checked out.
    """
    if not (PAPER_CODE / "src" / "csfs").is_dir():
        raise RuntimeError("the paper code is missing: run "
                           "`git submodule update --init`")
    os.environ.setdefault("TQDM_DISABLE", "1")

    def quiet(*_args: object, **_kwargs: object) -> None:
        return None

    if "fd_shifts" not in sys.modules:
        fd_shifts = types.ModuleType("fd_shifts")
        vars(fd_shifts).update(logger=SimpleNamespace(
            info=quiet, warning=quiet, error=quiet, debug=quiet, disable=quiet,
            enable=quiet))
        sys.modules["fd_shifts"] = fd_shifts
    if "src.utils" in sys.modules:
        vars(sys.modules["src.utils"]).update(get_model_and_last_layer=_last_layer)
    else:
        utils = types.ModuleType("src.utils")
        vars(utils).update(get_model_and_last_layer=_last_layer)
        sys.modules["src.utils"] = utils
    for name in ("faiss", "bayes_opt"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            vars(stub).update(BayesianOptimization=None)
            sys.modules[name] = stub
    if str(PAPER_CODE) not in sys.path:
        sys.path.insert(0, str(PAPER_CODE))


def _config(n_classes: int) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(num_classes=n_classes),
        eval=SimpleNamespace(ext_confid_name=STUDY, confidence_measures=None),
        model=SimpleNamespace(test_mcd_samples=1,
                              network=SimpleNamespace(backbone="resnet18",
                                                      name=None)),
    )


def _module(w: torch.Tensor, b: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(), w=w, b=b)


def paper_scores(train: torch.Tensor, labels: torch.Tensor,
                 logits_train: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                 evaluation: torch.Tensor, logits: torch.Tensor, n_classes: int
                 ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The 21 CSFs of the paper code that need no gradient training.

    A higher score means more confidence, as in the paper pipeline. The logit-only
    scores use float64, so that a saturated softmax does not create ties. The
    feature scores use float32, as in the paper pipeline. A CSF that raises an error
    gets NaN scores, and the error text is returned.

    Returns:
        Scores per CSF on the rows of `evaluation`, and the error text per failed CSF.
    """
    install()
    from src import scores_funcs
    from src.csfs import (
        NCI,
        ClassTypicalMatching,
        GradNorm,
        KernelPCA,
        MahalanobisDistance,
        MahalanobisPP,
        NeCo,
        NNGuide,
        ProjectionFiltering,
        ResidualScore,
        ViMScore,
        fDBD,
        pNML,
    )
    from src.csfs.entropy_funcs import generalized_entropy, renyi_entropy

    module, cf = _module(w, b), _config(n_classes)
    logits64 = logits.double()
    softmax64 = torch.softmax(logits64, dim=1)

    def fit_ctm():
        ctm = ClassTypicalMatching(module, STUDY, cf, mode="global")
        ctm.compute_CTM_params(train, labels)
        return ctm.get_scores(evaluation, similarity="weight")

    def fit_pnml():
        pnml = pNML(module, STUDY, cf)
        pnml.compute_pNML_params(train)
        return pnml.get_scores(evaluation)

    def fit_fdbd():
        fdbd = fDBD(module, STUDY, cf)
        fdbd.compute_fDBD_params(train)
        return fdbd.get_scores(evaluation, logits_eval=logits)

    def fit_nnguide():
        nnguide = NNGuide(module, STUDY, cf)
        nnguide.compute_NNGuide_params(train, labels_train=labels,
                                       logits_train=logits_train,
                                       proportion=0.1, k_clusters=10)
        return nnguide.get_scores(evaluation)

    def fit_neco():
        neco = NeCo(module, STUDY, cf)
        neco.compute_NeCo_params(train)
        return neco.get_scores(evaluation)

    def fit_maha(cls):
        maha = cls(cf)
        maha.compute_MahaDist_params(train, labels)
        return maha.get_scores(evaluation)

    def fit_nci():
        nci = NCI(module, STUDY, cf)
        nci.compute_NCI_params(train)
        return nci.get_scores(evaluation, logits_eval=logits)

    def fit_vim():
        vim = ViMScore(module, STUDY, cf)
        vim.compute_ViM_params(train)
        return vim.get_scores(evaluation)

    def fit_residual():
        residual = ResidualScore(module, STUDY, cf)
        residual.compute_Residual_params(train)
        return residual.get_scores(evaluation)

    def fit_pca():
        projection = ProjectionFiltering(module, STUDY, cf, mode="global")
        projection.compute_ProjectionFiltering_params(
            train, labels=labels, only_correct=False, variance_explained=0.9)
        return projection.get_scores(evaluation)

    def fit_kpca():
        kernel = KernelPCA(module, STUDY, cf, mode="global")
        kernel.compute_KernelPCA_params(
            train, labels=labels, only_correct=False, temperature=1.0,
            n_landmarks=200, explained_variance=0.95, gamma=0.2, center_on="all",
            kernel="rbf")
        return kernel.get_scores(evaluation)

    recipes = {
        "MSR": lambda: scores_funcs.maximum_softmax_response(softmax64),
        "MLS": lambda: scores_funcs.maximum_logit_score(logits64),
        "Energy": lambda: scores_funcs.energy(logits64),
        "GEN": lambda: generalized_entropy(softmax64, gamma=0.1, M=n_classes),
        "REN": lambda: renyi_entropy(softmax64, gamma=0.1, M=n_classes),
        "PCE": lambda: scores_funcs.predictive_collision_entropy(softmax64),
        "GE": lambda: scores_funcs.guessing_entropy(softmax64),
        "PE": lambda: scores_funcs.predictive_entropy(softmax64),
        "GradNorm": lambda: GradNorm(module, STUDY, cf).get_scores(
            evaluation, use_cuda=False),
        "pNML": fit_pnml,
        "CTM": fit_ctm,
        "fDBD": fit_fdbd,
        "NNGuide": fit_nnguide,
        "NeCo": fit_neco,
        "Maha": lambda: fit_maha(MahalanobisDistance),
        "MahaPP": lambda: fit_maha(MahalanobisPP),
        "NCI": fit_nci,
        "ViM": fit_vim,
        "Residual": fit_residual,
        "PCA_RecError": fit_pca,
        "KPCA_RecError": fit_kpca,
    }
    scores: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for name in PAPER_CSFS:
        torch.manual_seed(0)
        np.random.seed(0)
        try:
            values = recipes[name]()
            scores[name] = np.asarray(torch.as_tensor(values).detach().cpu(),
                                      dtype=np.float64).ravel()
        except Exception as error:  # noqa: BLE001 - one CSF must not stop the run
            scores[name] = np.full(len(evaluation), np.nan)
            errors[name] = f"{type(error).__name__}: {error}"
    return scores, errors


def paper_nc(train: torch.Tensor, labels: torch.Tensor, w: torch.Tensor,
             b: torch.Tensor, n_classes: int) -> dict[str, float]:
    """The 8 NC metrics of the paper, on penultimate training features."""
    install()
    from src.neural_collapse import NeuralCollapseMetrics

    metrics = NeuralCollapseMetrics(_module(w, b), STUDY, _config(n_classes))
    metrics.compute_NeuralCollapse_params(train, labels)
    return {name: float(metrics.nc_metrics[name]) for name in NC_METRICS}
