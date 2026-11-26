import json

import pytest
import torch
from test_build_subset import wnid, write_parquet

from dynamics import extract_imagenet as ex
from dynamics import zoo_list as zl


def test_frozen_list_of_public_models():
    torchvision = pytest.importorskip("torchvision")
    stored = json.loads(zl.PATH.read_text())
    models = stored["models"]
    if torchvision.__version__.startswith(zl.TORCHVISION):
        assert stored == zl.table()                  # the metadata give the file
    assert len(models) == 100 and len({m["id"] for m in models}) == 100
    assert {m["weights"] for m in models} == set(zl.WEIGHTS)
    assert len({m["family"] for m in models}) == 21
    sizes = {}
    for m in models:
        sizes[m["lineage"]] = sizes.get(m["lineage"], 0) + 1
        assert m["lineage"] == zl.LINEAGES[m["family"]]
        assert m["family"] not in zl.NO_LINEAR_HEAD
        assert 50 < m["acc1"] < 90 and m["crop_size"][0] >= 224
    assert sizes == {"regnet": 28, "resnet_kind": 17, "mobile": 13, "efficientnet": 12,
                     "vgg_kind": 9, "swin": 6, "convnext": 4, "densenet": 4, "vit": 4,
                     "inception_kind": 2, "maxvit": 1}
    cells = {}
    for m in models:
        key = (m["arch_type"], m["recipe_tag"])
        cells[key] = cells.get(key, 0) + 1
    assert cells == {("convolutional", "no"): 35, ("convolutional", "ported"): 19,
                     ("convolutional", "yes"): 35, ("transformer", "yes"): 11}
    by_id = {m["id"]: m for m in models}
    assert by_id["resnet50-v1"]["recipe_tag"] == "no"
    assert by_id["resnet50-v2"]["recipe_tag"] == "yes"
    assert by_id["vit_b_16-v1"]["arch_type"] == "transformer"
    assert by_id["efficientnet_v2_l-v1"]["recipe_tag"] == "ported"
    assert by_id["inception_v3-v1"]["crop_size"] == [299]


def test_tags_follow_the_documentation_of_a_weight():
    base = "https://github.com/pytorch/vision/tree/main/references/classification"
    simple = "These weights reproduce closely the results of the paper using a simple "
    assert zl.recipe_tag(base + "#resnet", simple + "training recipe.") == "no"
    assert zl.recipe_tag(base + "#convnext", "These weights improve upon ...") == "yes"
    assert zl.recipe_tag(base + "#efficientnet-v2",
                         "These weights are ported from the original paper.") == "ported"
    assert zl.recipe_tag("https://github.com/pytorch/vision/issues/3995#new-recipe",
                         "improve upon the results") == "yes"
    assert zl.recipe_tag("https://github.com/pytorch/vision/pull/5906",
                         "trained from scratch by using TorchVision's `new training "
                         "recipe <...>`_.") == "yes"
    assert zl.recipe_tag("https://github.com/1e100/mnasnet_trainer",
                         "These weights reproduce closely the results.") == "ported"
    assert zl.recipe_tag("https://github.com/pytorch/vision/pull/116",
                         "These weights are ported from LuaTorch.") == "ported"
    assert zl.family_of("resnext50_32x4d") == "resnext"
    assert zl.family_of("swin_v2_t") == "swin_v2" and zl.family_of("swin_t") == "swin"
    assert zl.model_id("resnet50", "IMAGENET1K_V2") == "resnet50-v2"
    with pytest.raises(ValueError):
        zl.family_of("unknown_net")


def test_extraction_of_the_imagenet_images(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    labels = [label for label in range(4) for _ in range(9)]
    half = len(labels) // 2
    for part, chunk in enumerate((labels[:half], labels[half:])):
        write_parquet(folder / f"train-0000{part}-of-00002.parquet", chunk,
                      [f"{wnid(label)}_{part}{i:03d}.JPEG"
                       for i, label in enumerate(chunk)], sizes=(40, 64))
    labels = [label for label in range(4) for _ in range(3)]
    write_parquet(folder / "validation-00000-of-00001.parquet", labels,
                  [f"ILSVRC2012_val_{i:08d}_{wnid(label)}.JPEG"
                   for i, label in enumerate(labels)], sizes=(40, 64))
    out = tmp_path / "out"
    manifest = ex.extract(folder, out, per_class=5, workers=2)
    assert manifest["classes"] == 4 and manifest["counts"] == {"train": 20, "val": 12}
    for label in range(4):
        assert len(list((out / "train" / wnid(label)).iterdir())) == 5
        assert len(list((out / "val" / wnid(label)).iterdir())) == 3
    assert json.loads((out / "extract_manifest.json").read_text()) == manifest
    # The image bytes are those of the parquet files: the size does not change.
    from PIL import Image

    image = next((out / "train" / wnid(0)).iterdir())
    assert Image.open(image).size == (40, 64)
    # A second extraction gives the same lists, and another seed gives another one.
    assert ex.extract(folder, tmp_path / "again", per_class=5, workers=2) == manifest
    other = ex.extract(folder, tmp_path / "other", per_class=5, seed=1, workers=2)
    assert other["sha256"]["train"] != manifest["sha256"]["train"]
    assert other["sha256"]["val"] == manifest["sha256"]["val"]
    with pytest.raises(ValueError, match="need 10"):
        ex.extract(folder, tmp_path / "few", per_class=10, workers=2)
    names = [f"a{i}" for i in range(9)]
    assert ex.choose(names, 5, (16, 0)) == ex.choose(names[::-1], 5, (16, 0))


# ---- the evaluation of a public model ----

class Noise(torch.utils.data.Dataset):
    """A few random images of one size."""

    def __init__(self, count: int, size: int) -> None:
        self.images = torch.rand(count, 3, size, size,
                                 generator=torch.Generator().manual_seed(0))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], index % 3


def small_networks():
    from torchvision import models

    return {
        "resnet18": lambda: models.resnet18(num_classes=7),
        "vgg11": lambda: models.vgg11(num_classes=7),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(num_classes=7),
        "efficientnet_b0": lambda: models.efficientnet_b0(num_classes=7),
        "convnext_tiny": lambda: models.convnext_tiny(num_classes=7),
        "googlenet": lambda: models.googlenet(num_classes=7, aux_logits=False,
                                              init_weights=False),
        "vit": lambda: models.VisionTransformer(
            image_size=64, patch_size=16, num_layers=1, num_heads=2, hidden_dim=32,
            mlp_dim=64, num_classes=7),
    }


@pytest.mark.parametrize("name", sorted(small_networks()))
def test_the_feature_is_the_input_of_the_last_linear_layer(name):
    from dynamics import zoo

    torch.manual_seed(0)
    model = small_networks()[name]().eval()
    head = zoo.last_linear(model)
    assert head.out_features == 7
    out = zoo.extract(model, head, Noise(5, 64), torch.device("cpu"), batch_size=2,
                      workers=0)
    assert out["feature"].shape == (5, head.in_features)
    assert out["labels"].tolist() == [0, 1, 2, 0, 1]
    again = out["feature"] @ head.weight.detach().T + head.bias.detach()
    assert torch.allclose(again, out["output"], atol=1e-4)
    # The hook is removed after the extraction.
    assert not head._forward_pre_hooks


def test_a_network_without_a_linear_layer_is_refused():
    from torchvision import models

    from dynamics import zoo

    with pytest.raises(ValueError, match="no linear layer"):
        zoo.last_linear(models.squeezenet1_0(num_classes=7))


def test_stand_in_sets_for_any_image_size():
    import numpy as np

    from dynamics import zoo
    from dynamics.data import patch_permutations, shuffle_patches

    permutation = patch_permutations(1)[0]
    even = torch.rand(3, 32, 32)
    assert torch.equal(zoo.shuffle_any(even, permutation),
                       shuffle_patches(even, permutation))
    odd = torch.rand(3, 30, 31)                     # as the 299 px of Inception V3
    out = zoo.shuffle_any(odd, permutation)
    assert out.shape == odd.shape and not torch.equal(out, odd)
    assert torch.equal(out[:, 28:, :], odd[:, 28:, :])
    assert torch.equal(out[:, :, 28:], odd[:, :, 28:])
    assert torch.equal(out[:, :28, :28], shuffle_patches(odd[:, :28, :28].contiguous(),
                                                         permutation))
    labels = np.repeat(np.arange(5), 8)
    kept = zoo.standin_indices(labels, per_class=3)
    assert len(kept) == 15 and np.bincount(labels[kept]).tolist() == [3] * 5
    assert np.array_equal(kept, zoo.standin_indices(labels, per_class=3))
    assert len(zoo.standin_indices(labels, per_class=20)) == 40
    assert zoo.batch_size_of(224) == 256 and zoo.batch_size_of(600) == 35
    assert zoo.batch_size_of(4000) == 16


def test_evaluation_of_a_model_on_tiny_images(tiny_data, tmp_path, monkeypatch):
    from torchvision import models, transforms

    from dynamics import zoo
    from dynamics.scores import ALL_CSFS

    def fake(entry):
        torch.manual_seed(0)
        return models.resnet18(num_classes=10).eval(), transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor()])

    monkeypatch.setattr(zoo, "load_model", fake)
    entry = {"id": "resnet18-v1", "model": "resnet18", "weights": "IMAGENET1K_V1",
             "acc1": 69.758, "crop_size": [32]}
    extra = [("places", tiny_data.imagenette.parent / "openood" / "places")]
    device = torch.device("cpu")
    zoo.evaluate_model(entry, tiny_data.imagenette, tiny_data.textures, extra,
                       tmp_path, device, workers=0)
    path = tmp_path / "zoo" / "resnet18-v1.json"
    out = json.loads(path.read_text())
    assert out["model"] == "resnet18-v1" and out["n_classes"] == 10
    assert out["feature_dim"] == 512 and out["published_accuracy"] == 0.69758
    assert out["counts"] == {"val": 120, "blend": 100, "shuffle": 100, "textures": 40,
                             "places": 30, "train": 240}
    sets = {row["set"] for row in out["metrics"]}
    assert sets == {"blend", "shuffle", "textures", "places", "id_errors"}
    assert {row["csf"] for row in out["metrics"]} == set(ALL_CSFS)
    assert len(out["metrics"]) == 5 * len(ALL_CSFS)
    row = next(r for r in out["metrics"] if r["set"] == "places")
    assert {"auroc", "augrc", "auroc_correct", "augrc_correct"} <= set(row)
    assert out["checks"]["head_gap"] < 1e-4
    assert out["checks"]["argmax_agreement"] == 1.0
    clock = {r["name"]: r["value"] for r in out["descriptors"]}
    assert clock["val_accuracy"] == out["val_accuracy"] and "train_loss" in clock
    assert out["errors"] == {} and not any(out["nonfinite"].values())
    # The time of each score is in the result file.
    assert set(out["seconds_by_score"]) == {*zoo.PAPER_CSFS, "kNN, FeatureNorm"}
    # A model with a result file is skipped.
    stamp = path.stat().st_mtime_ns
    zoo.evaluate_model(entry, tiny_data.imagenette, tiny_data.textures, extra,
                       tmp_path, device, workers=0)
    assert path.stat().st_mtime_ns == stamp
    # A head with another number of outputs is refused.
    monkeypatch.setattr(zoo, "load_model", lambda entry: (
        models.resnet18(num_classes=7).eval(), transforms.ToTensor()))
    with pytest.raises(ValueError, match="7 outputs"):
        zoo.evaluate_model({**entry, "id": "other"}, tiny_data.imagenette, None, [],
                           tmp_path, device, workers=0)
    listed = zl.load()
    assert [e["id"] for e in zoo.chosen(["pilot"], listed)] == list(zoo.PILOT)
    assert len(zoo.chosen(["all"], listed)) == 100
    assert zoo.chosen(["vgg16-v1"], listed)[0]["family"] == "vgg"
    with pytest.raises(ValueError, match="not in the frozen list"):
        zoo.chosen(["resnet9000-v1"], listed)


def test_every_entry_of_the_list_has_its_weights_and_preprocessing():
    torchvision = pytest.importorskip("torchvision")
    if not torchvision.__version__.startswith(zl.TORCHVISION):
        pytest.skip("another torchvision")
    from torchvision import models

    for entry in zl.load():
        weights = models.get_model_weights(entry["model"])[entry["weights"]]
        preset = weights.transforms()
        assert list(preset.crop_size) == entry["crop_size"]
        assert list(preset.resize_size) == entry["resize_size"]
        assert weights.meta["_metrics"]["ImageNet-1K"]["acc@1"] == entry["acc1"]


# ---- the analysis ----

def test_weighted_rule_is_the_rule_of_the_project_with_equal_weights():
    import numpy as np
    import pandas as pd

    from dynamics import analyze_i, rules

    rng = np.random.default_rng(0)
    metrics = pd.DataFrame(rng.uniform(0.4, 0.95, (30, 6)),
                           columns=[f"score/blend/{i}" for i in range(6)])
    scores = pd.DataFrame(rng.uniform(55, 95, (30, 4)), columns=list("abcd"))
    plain = rules.fit_ridge_rule(metrics, scores, list(metrics.columns))
    same = analyze_i.fit_weighted(metrics, scores, list(metrics.columns), np.ones(30))
    for name in ("mean", "std", "beta", "offset"):
        assert np.allclose(getattr(plain, name), getattr(same, name))
    assert plain.choose(metrics) == same.choose(metrics)
    # A row with twice the weight is the same as the row twice.
    weights = np.ones(30)
    weights[0] = 2.0
    doubled = analyze_i.fit_weighted(metrics, scores, list(metrics.columns), weights)
    twice = rules.fit_ridge_rule(pd.concat([metrics, metrics.iloc[[0]]]),
                                 pd.concat([scores, scores.iloc[[0]]]),
                                 list(metrics.columns))
    assert np.allclose(doubled.mean, twice.mean)
    lineage = pd.Series(["a"] * 6 + ["b"] * 2 + ["c"])
    weights = analyze_i.balanced(lineage)
    assert weights.mean() == pytest.approx(1.0)
    totals = pd.Series(weights).groupby(lineage).sum()
    assert np.allclose(totals, totals.iloc[0])


def fabricate(tmp_path, listed, wrong_accuracy=(), failing=None):
    """Result files with the layout of `dynamics.zoo`, from random scores."""
    import numpy as np

    from dynamics import analyze_i
    from dynamics.metrics import error_metrics, set_metrics
    from dynamics.scores import ALL_CSFS

    sets = [name for group in analyze_i.GROUPS.values() for name in group]
    folder = tmp_path / "zoo"
    folder.mkdir(exist_ok=True)
    for number, entry in enumerate(listed):
        rng = np.random.default_rng(number)
        correct = rng.random(200) < 0.8
        metrics = []
        for position, csf in enumerate(ALL_CSFS):
            # In lineage "a" the first score is the best, elsewhere the second.
            lift = 2.0 if position == (0 if entry["lineage"] == "a" else 1) else 0.5
            scores = rng.normal(lift, 1.0, 200)
            for name in (*sets, "blend", "shuffle"):
                unseen = rng.normal(0.0 if name in sets else lift / 2, 1.0, 100)
                metrics.append({"set": name, "csf": csf,
                                **set_metrics(scores, correct, unseen)})
            metrics.append({"set": "id_errors", "csf": csf,
                            **error_metrics(scores, correct)})
        accuracy = float(correct.mean())
        errors = {failing[1]: "LinAlgError"} if failing and failing[0] == entry[
            "id"] else {}
        out = {"model": entry["id"], "metrics": metrics,
               "descriptors": [{"group": "clock", "name": "val_accuracy",
                                "layer": "", "value": accuracy}],
               "checks": {"head_gap": 0.01, "argmax_agreement": 0.999},
               "errors": errors, "nonfinite": dict.fromkeys(ALL_CSFS, 0),
               "counts": {"val": 200, "blend": 100, "shuffle": 100, "train": 500,
                          **dict.fromkeys(sets, 100)},
               "val_accuracy": accuracy, "feature_dim": 64, "seconds": 600.0,
               "published_accuracy": accuracy + (0.05 if entry["id"] in wrong_accuracy
                                                 else 0.0), "limit": None}
        (folder / f"{entry['id']}.json").write_text(json.dumps(out))


def test_analysis_of_fabricated_models(tmp_path, monkeypatch):
    from dynamics import analyze_i

    listed = [{"id": f"m{i:02d}", "model": f"m{i:02d}", "weights": "IMAGENET1K_V1",
               "family": f"f{i % 6}", "lineage": "abcd"[i % 4],
               "arch_type": "convolutional" if i % 3 else "transformer",
               "recipe_tag": "yes" if i % 2 else "no", "acc1": 80.0,
               "crop_size": [224]} for i in range(16)]
    path, report = tmp_path / "models.json", tmp_path / "i.md"
    path.write_text(json.dumps({"models": listed}))

    def run(*extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_i", "--results-root", str(tmp_path), "--models-file", str(path),
            "--out", str(report), *extra])
        analyze_i.main()
        return report.read_text()

    with pytest.raises(SystemExit, match="not the frozen one"):
        run()
    with pytest.raises(SystemExit, match="no result files"):
        run("--allow-other-list")
    fabricate(tmp_path, listed[:15])
    with pytest.raises(SystemExit, match="not complete.*m15"):
        run("--allow-other-list")
    fabricate(tmp_path, listed)
    text = run("--allow-other-list")
    assert "**Gate W.**" in text and "for 16 of 16 models" in text
    assert "removed for every model: none. 23 of 23 scores stay" in text
    assert "**Gate M passes.**" in text and "Models in the analysis: 16, in 4" in text
    assert "## Target far: the outcome" in text and "**Outcome:" in text
    for title in ("## The family as the unit", "## The model as the unit",
                  "## By tag", "## Shortlists and AUGRC",
                  "## The inputs that carry the choice", "## Models",
                  "## Target near (secondary", "## Target textures (secondary",
                  "## Target imagenet_o (secondary"):
        assert title in text
    assert "The best fixed score in hindsight is" in text
    # The lineage is left out: in lineage "a" another score is the best, and the
    # fixed score comes from the other lineages.
    tables, _ = analyze_i.read(tmp_path)
    frame = pd_frame(listed)
    csfs = list(analyze_i.ALL_CSFS)
    metrics = analyze_i.inputs_of(tables, csfs).loc[frame.index]
    assert metrics.shape == (16, 3 * len(csfs))
    scores = analyze_i.targets_of(tables, analyze_i.GROUPS["far"], csfs).loc[
        frame.index]
    picks, short = analyze_i.leave_out(metrics, scores, frame)
    inside = (frame["lineage"] == "a").to_numpy()
    assert set(picks.loc[inside, analyze_i.ONE]) == {csfs[1]}
    assert set(scores[inside].idxmax(axis=1)) == {csfs[0]}
    assert all(len(row) == 3 for row in short["rule_short"])
    # Gate W: a model with another accuracy leaves. Gate S: a score with an error
    # for one model is removed for every model.
    fabricate(tmp_path, listed, wrong_accuracy=("m03",), failing=("m05", "pNML"))
    text = run("--allow-other-list")
    assert "for 15 of 16 models" in text and "m03" in text
    assert "removed for every model: pNML. 22 of 23 scores stay" in text
    assert "Models in the analysis: 15" in text
    # More than a quarter of the models leave: the test is not decided.
    fabricate(tmp_path, listed, wrong_accuracy=tuple(e["id"] for e in listed[:5]))
    with pytest.raises(SystemExit, match="not decided"):
        run("--allow-other-list")
    assert "**The test is not decided.**" in report.read_text()
    assert "**Outcome:" not in report.read_text()
    (tmp_path / "zoo" / "other.json").write_text(
        (tmp_path / "zoo" / "m00.json").read_text().replace('"m00"', '"other"'))
    with pytest.raises(SystemExit, match="not in the list"):
        run("--allow-other-list")


def pd_frame(listed):
    import pandas as pd

    return pd.DataFrame(listed).set_index("id")


def test_scores_one_at_a_time_are_the_scores_of_one_call():
    import numpy as np

    from dynamics import zoo
    from dynamics.scores import all_scores

    torch.manual_seed(0)
    n_classes, d = 6, 16
    w, b = torch.randn(n_classes, d), torch.randn(n_classes)
    labels = torch.arange(n_classes).repeat_interleave(30)
    train = w[labels] + torch.randn(len(labels), d)
    rows = torch.randn(80, d)
    fit = {"feature": train, "labels": labels, "logits": train @ w.T + b}
    joint, joint_errors = all_scores(train, labels, fit["logits"], w, b, rows,
                                     rows @ w.T + b, n_classes, None, "float64")
    single, errors, seconds = zoo.timed_scores(fit, w, b, rows, rows @ w.T + b,
                                               n_classes, None, "float64")
    assert errors == joint_errors == {} and list(single) == list(joint)
    for name in joint:
        assert np.array_equal(single[name], joint[name]), name
    assert len(seconds) == len(zoo.PAPER_CSFS) + 1


def test_mahalanobis_scores_for_many_classes_are_those_of_the_paper_code():
    from dynamics import paper_api

    paper_api.install()
    from src.csfs import MahalanobisDistance, MahalanobisPP

    torch.manual_seed(0)
    n_classes, d = 7, 24
    means = torch.randn(n_classes, d, dtype=torch.float64) * 2
    labels = torch.arange(n_classes).repeat_interleave(60)
    train = means[labels] + torch.randn(len(labels), d, dtype=torch.float64)
    rows = torch.randn(300, d, dtype=torch.float64) * 1.5
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for cls in (MahalanobisDistance, MahalanobisPP):
            maha = cls(paper_api._config(n_classes))
            maha.compute_MahaDist_params(train, labels)
            old = maha.get_scores(rows)
            new = paper_api.maha_scores(maha, rows, normalize=cls is MahalanobisPP,
                                        chunk=64)
            assert torch.allclose(old, new, rtol=1e-10, atol=1e-10)
    finally:
        torch.set_default_dtype(previous)
    # Every earlier setting stays on the path of the paper code: 10 and 100
    # classes with features of 512, 1,024, or 2,048 values.
    def bytes_of(d, classes):
        return paper_api.MAHA_CHUNK * d * classes * 8

    for d, classes in ((512, 10), (1024, 10), (2048, 10), (512, 100)):
        assert bytes_of(d, classes) <= paper_api.MAHA_BUDGET
    for d, classes in ((512, 1000), (2048, 1000), (4096, 1000)):
        assert bytes_of(d, classes) > paper_api.MAHA_BUDGET


def test_gradnorm_scores_for_many_rows_are_those_of_the_paper_code():
    from dynamics import paper_api

    paper_api.install()
    from src.csfs import GradNorm

    torch.manual_seed(0)
    n_classes, d = 9, 20
    w = torch.randn(n_classes, d, dtype=torch.float64)
    b = torch.randn(n_classes, dtype=torch.float64)
    rows = torch.randn(200, d, dtype=torch.float64) * 1.5
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        old = GradNorm(paper_api._module(w, b), paper_api.STUDY,
                       paper_api._config(n_classes)).get_scores(rows, use_cuda=False)
        new = paper_api.gradnorm_scores(w, b, rows, chunk=64)
    finally:
        torch.set_default_dtype(previous)
    assert torch.allclose(old.double(), new, rtol=1e-10, atol=1e-10)
    # Every earlier evaluation stays on the path of the paper code.
    for rows_, classes in ((53_344, 10), (56_569, 100)):
        assert rows_ * classes <= paper_api.GRADNORM_LIMIT
    assert 107_640 * 1000 > paper_api.GRADNORM_LIMIT


def test_a_head_with_no_bias_has_a_bias_of_zero():
    from dynamics import zoo

    torch.manual_seed(0)
    # The classifier of MaxViT ends in a linear layer with no bias.
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 7, bias=False)).eval()
    head = zoo.last_linear(model)
    assert head.bias is None
    weight, bias = zoo.head_parameters(head)
    assert weight.shape == (7, 16) and torch.equal(bias, torch.zeros(7))
    out = zoo.extract(model, head, Noise(5, 8), torch.device("cpu"), batch_size=2,
                      workers=0)
    assert torch.allclose(out["feature"] @ weight.T + bias, out["output"], atol=1e-6)
    with_bias = torch.nn.Linear(16, 7)
    assert torch.equal(zoo.head_parameters(with_bias)[1], with_bias.bias.detach())
