import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_stage_i import fabricate

from dynamics import analyze_j as aj
from dynamics import rules, zoo, zoo_list
from dynamics import zoo_timm_list as zt
from dynamics import zoo_timm_tags as tags
from dynamics.scores import ALL_CSFS

# ---- the list of timm models ----


def test_the_tag_of_a_weight_decides_if_it_is_eligible():
    for tag in ("in1k", "a1_in1k", "fb_in1k", "apple_in1k", "msft_in1k", "sam_in1k",
                "snap_in1k", "c2ns_in1k", "ra4_e3600_r224_in1k", "miil_in1k"):
        assert zt.tag_eligible(tag), tag
    for tag in ("tv_in1k", "tv2_in1k", "fb_in22k_ft_in1k", "fb_dist_in1k",
                "ap_in1k", "ft_in1k", "ns_jft_in1k", "tf_adv_in1k", "ssld_in1k",
                "fcmae_ft_in1k", "sbb_in12k_ft_in1k", "in21k", "swag_lc_in1k",
                "fb_ssl_yfcc100m_ft_in1k", "mae_in1k_ft_in1k", "in1k_dist"):
        assert not zt.tag_eligible(tag), tag


def test_the_module_of_a_model_gives_its_group_and_lineage():
    assert zt.lineage_of("resnet", "resnet50") == ("S", "resnet_kind")
    assert zt.lineage_of("resnetv2", "resnetv2_50") == ("S", "resnet_kind")
    assert zt.lineage_of("efficientnet", "tf_efficientnet_b2") == ("S", "efficientnet")
    assert zt.lineage_of("efficientnet", "mobilenetv2_050") == ("S", "mobile")
    assert zt.lineage_of("naflexvit", "naflexvit_base_patch16_gap") == ("S", "vit")
    assert zt.lineage_of("xcit", "xcit_tiny_12_p8_224") == ("N", "xcit")
    assert zt.lineage_of("byobnet", "repvgg_b1") == ("N", "byobnet")
    # Every lineage of group S is a lineage of amendment A16.
    known = {entry["lineage"] for entry in zoo_list.load()}
    assert set((zt.SEEN | zt.SEEN_ADDED).values()) | {"efficientnet"} <= known


def rows_of(spec: dict[str, list[str]]) -> list[dict]:
    """Eligible weights: lineage -> names, with `S:` before a seen lineage."""
    rows = []
    for key, names in spec.items():
        group, lineage = key.split(":") if ":" in key else ("N", key)
        rows += [{"id": name, "architecture": name.split(".")[0], "group": group,
                  "lineage": lineage} for name in names]
    return rows


def test_the_draw_takes_two_architectures_of_each_lineage():
    spec = {"xcit": [f"xcit_{i}.in1k" for i in range(6)] + ["xcit_0.fb_in1k"],
            "pit": ["pit_a.in1k"],
            "S:resnet_kind": ["resnet50.a1_in1k", "resnet50.ra_in1k",
                              "resnet18.a1_in1k", "resnet34.a1_in1k",
                              "resnet101.a1_in1k"]}
    rows = rows_of(spec)
    chosen, removed = zt.draw(rows, lambda row: "")
    assert removed == [] and chosen == zt.draw(rows, lambda row: "")[0]
    by_group = {group: [e for e in chosen if e["group"] == group] for group in "NRS"}
    assert [e["id"] for e in by_group["R"]] == ["resnet50.a1_in1k", "resnet50.ra_in1k"]
    xcit = [e for e in by_group["N"] if e["lineage"] == "xcit"]
    assert len(xcit) == 2 and len({e["architecture"] for e in xcit}) == 2
    assert [e["id"] for e in by_group["N"] if e["lineage"] == "pit"] == ["pit_a.in1k"]
    seen = by_group["S"]
    assert len(seen) == 2 and all(e["architecture"] != "resnet50" for e in seen)
    # A weight that a check removes gives its place to the next one of the walk,
    # and the other lineages do not change.
    first = xcit[0]["id"]
    again, removed = zt.draw(rows, lambda row: "no" if row["id"] == first else "")
    assert removed == [{"id": first, "group": "N", "lineage": "xcit", "reason": "no"}]
    other = [e for e in again if e["lineage"] == "xcit"]
    assert first not in {e["id"] for e in other} and len(other) == 2
    assert other[0]["id"] == xcit[1]["id"] or xcit[1]["architecture"] == first.split(
        ".")[0]
    assert [e for e in again if e["lineage"] != "xcit"] == [
        e for e in chosen if e["lineage"] != "xcit"]
    assert zt.walk_order(spec["xcit"], "xcit") == zt.walk_order(
        list(reversed(spec["xcit"])), "xcit")


class Tokens(torch.nn.Module):
    """A head that takes one vector for each token."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(3, zt.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.flatten(2).transpose(1, 2)).mean(dim=1)


class TwoHeads(torch.nn.Module):
    """The output is the mean of two heads."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, zt.N_CLASSES)
        self.second = torch.nn.Linear(12, zt.N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = x.flatten(1)[:, :12]
        return (self.head(feature) + self.second(feature)) / 2

    def get_classifier(self):
        return self.head, self.second


class Named(torch.nn.Module):
    """A network that names its classifier, with a later linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(12, zt.N_CLASSES)
        self.unused = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1)[:, :12])

    def get_classifier(self):
        return self.fc


def test_the_head_check_removes_heads_that_are_not_one_linear_layer():
    plain = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 32),
                                torch.nn.ReLU(), torch.nn.Linear(32, zt.N_CLASSES))
    assert zt.head_problem(plain, 8) == ""
    named = Named()
    assert zoo.head_of(named) is named.fc and zt.head_problem(named, 8) == ""
    assert zoo.head_of(plain) is plain[3]                    # no named classifier
    conv = torch.nn.Sequential(torch.nn.Conv2d(3, zt.N_CLASSES, 1),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    assert "no linear layer" in zt.head_problem(conv, 8)
    assert "one vector" in zt.head_problem(Tokens(), 8)
    assert "differs from the output" in zt.head_problem(TwoHeads(), 8)
    assert zoo.head_of(TwoHeads()) is not None       # a tuple is not the head
    small = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 7))
    assert "1,000 values" in zt.head_problem(small, 8)
    assert "the pass at 4 px fails" in zt.head_problem(plain, 4)


class SEModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_the_type_of_a_network_comes_from_its_layers():
    convolutional = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), SEModule(),
                                        torch.nn.Conv2d(8, 8, 3))
    assert zt.arch_type_of(convolutional) == "convolutional"
    attention = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 4, 4),
                                    torch.nn.MultiheadAttention(8, 2))
    assert zt.arch_type_of(attention) == "transformer"
    linear = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 4, 4), torch.nn.Linear(8, 8))
    assert zt.arch_type_of(linear) == "other"
    recurrent = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 8, 3),
                                    torch.nn.LSTM(8, 8))
    assert zt.arch_type_of(recurrent) == "other"


def test_frozen_list_of_timm_models():
    stored = json.loads(zt.PATH.read_text())
    models = stored["models"]
    assert zt.file_hash(zt.PATH) == aj.MODELS_SHA256
    assert stored["timm"] == zt.TIMM and stored["results_sha256"] == zt.RESULTS_SHA256
    ids = [m["id"] for m in models]
    assert len(set(ids)) == len(ids) and set(stored["pilot"]) <= set(ids)
    groups = pd.DataFrame(models).groupby("group")
    assert set(groups.groups) == {"N", "R", "S"}
    assert (groups.get_group("R")["architecture"] == zt.RECIPE_ARCHITECTURE).all()
    known = {entry["lineage"] for entry in zoo_list.load()}
    assert set(groups.get_group("S")["lineage"]) <= known
    assert not set(groups.get_group("N")["lineage"]) & known
    for group in ("N", "S"):
        part = groups.get_group(group)
        assert part.groupby("lineage").size().max() <= zt.PER_LINEAGE
        assert not part.duplicated(["lineage", "architecture"]).any()
        assert (part["architecture"] != zt.RECIPE_ARCHITECTURE).all()
    for m in models:
        assert m["library"] == "timm" and zt.tag_eligible(m["tag"])
        assert m["arch_type"] in ("convolutional", "transformer", "other")
        assert m["recipe_tag"] in ("yes", "no", "unknown") and m["recipe_source"]
        assert m["crop_size"][0] <= zt.MAX_SIZE and 50 < m["acc1"] < 90
        assert m["id"] not in tags.NOT_ELIGIBLE
        assert zt.lineage_of(m["module"], m["architecture"]) == (
            "S" if m["group"] == "R" else m["group"], m["lineage"])
    kinds = {m["id"]: m["arch_type"] for m in models}
    assert [kinds[name] for name in stored["pilot"][:2]] == ["transformer",
                                                             "convolutional"]
    assert stored["pilot"][2] == "resnet50.a1_in1k"


def test_the_metadata_of_timm_give_the_frozen_list():
    timm = pytest.importorskip("timm")
    results = zt.HERE.parent / "dyn_results" / "timm_meta" / (
        "results-imagenet-v1.0.31.csv")
    if timm.__version__ != zt.TIMM or not results.exists():
        pytest.skip("needs timm 1.0.31 and its results file")
    assert zt.table(results) == json.loads(zt.PATH.read_text())


def test_the_loader_of_a_timm_weight(monkeypatch):
    timm = pytest.importorskip("timm")
    if timm.__version__ != zt.TIMM:
        pytest.skip("another timm")
    from PIL import Image

    real = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained: real(
        name, pretrained=False))                               # no download
    entry = next(m for m in zt.load() if m["id"] == "resnet50.a1_in1k")
    model, transform = zoo.load_model(entry)
    image = transform(Image.new("RGB", (500, 375)))
    assert image.shape == (3, 288, 288) and entry["crop_size"] == [288]
    assert zoo.head_of(model) is model.fc and not model.training
    assert zt.head_problem(model, 288) == ""
    with pytest.raises(ValueError, match="preprocessing"):
        zoo.load_model({**entry, "crop_size": [224]})
    monkeypatch.setattr(timm, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="needs timm"):
        zoo.load_model(entry)


def test_the_lists_of_the_evaluation():
    listed = zt.load()
    pilot = zt.pilot()
    assert [e["id"] for e in zoo.chosen(["pilot"], listed, pilot)] == pilot
    assert zoo.chosen(["all"], listed) == listed
    with pytest.raises(ValueError, match="not in the frozen list"):
        zoo.chosen(["resnet50-v1"], listed)
    load, pilot_of, folder = zoo.LISTS["a17"]
    assert load() == listed and pilot_of() == pilot and folder == aj.RESULTS
    assert zoo.LISTS["a16"][1]() == list(zoo.PILOT) and zoo.LISTS["a16"][2] == "zoo"


# ---- the frozen choices ----

def test_frozen_choices_of_amendment_a17():
    assert rules.file_hash(aj.RULES_PATH) == aj.RULES_SHA256
    frozen = aj.load()
    assert frozen["fit_list_sha256"] == aj.ai.MODELS_SHA256
    assert frozen["fit_models"] == [entry["id"] for entry in zoo_list.load()]
    assert set(frozen["groups"]) == set(aj.ai.GROUPS) and len(frozen["input_range"]) == 69
    cells = {"convolutional/no", "convolutional/ported", "convolutional/yes",
             "transformer/yes"}
    for name, group in frozen["groups"].items():
        assert tuple(group["sets"]) == aj.ai.GROUPS[name]
        assert len(group["rule"].columns) == 69 and group["rule"].csfs == list(ALL_CSFS)
        assert group["accuracy_rule"].columns == [aj.ACCURACY_INPUT]
        assert set(group[aj.TAG]) == cells == set(group[aj.TAG_MODELS])
        assert group[aj.ONE] in ALL_CSFS and len(group["shortlist"]) == 3
    far = frozen["groups"]["far"]
    assert far[aj.ONE] == "fDBD"
    assert far[aj.TAG] == {"convolutional/no": "MLS", "convolutional/ported": "fDBD",
                           "convolutional/yes": "fDBD", "transformer/yes": "MahaPP"}
    assert far[aj.TAG_MODELS]["convolutional/yes"] == "MahaPP"
    # A recipe that the library does not document takes the cell `ported`, and a
    # tag with no cell takes the fixed score.
    listed = pd.DataFrame({"arch_type": ["convolutional", "transformer", "other"],
                           "recipe_tag": ["unknown", "unknown", "yes"]},
                          index=["a", "b", "c"])
    assert list(aj.cells_of(listed)) == ["convolutional/ported", "transformer/ported",
                                         "other/yes"]
    metrics = pd.DataFrame(0.7, index=listed.index, columns=far["rule"].columns)
    picks = aj.choose(far, metrics, pd.Series(0.8, index=listed.index),
                      aj.cells_of(listed))
    assert list(picks[aj.TAG]) == ["fDBD", "fDBD", "fDBD"]
    assert set(picks[aj.RULE]) <= set(ALL_CSFS) and picks.shape == (3, 5)


def test_the_results_of_amendment_a16_give_the_frozen_choices():
    root = zt.HERE.parent / "dyn_results"
    if len(list((root / "zoo").glob("*.json"))) != 100:
        pytest.skip("needs the 100 result files of amendment A16")
    built, stored = aj.build(root), json.loads(aj.RULES_PATH.read_text())
    assert built.keys() == stored.keys()
    for name, group in stored["groups"].items():
        for key in (aj.ONE, aj.TAG, aj.TAG_MODELS, "shortlist", "sets"):
            assert built["groups"][name][key] == group[key]
        for rule in ("rule", "accuracy_rule"):
            for part in ("mean", "std", "beta", "offset"):
                assert np.allclose(built["groups"][name][rule][part],
                                   group[rule][part], rtol=1e-9, atol=1e-12)
    assert built["input_range"] == stored["input_range"]
    assert built["images"] == stored["images"]


# ---- the test of the frozen choices ----

def timm_list(lineages: int = 28) -> list[dict]:
    """A small list with the three groups."""
    models = []
    for number in range(lineages):
        for copy in range(2):
            models.append({"id": f"n{number:02d}_{copy}", "group": "N",
                           "lineage": "a" if number % 4 == 0 else f"new{number:02d}",
                           "arch_type": ("transformer", "convolutional", "other")[
                               number % 3],
                           "recipe_tag": ("yes", "no", "unknown")[(number + copy) % 3]})
    models[0]["lineage"] = models[1]["lineage"] = "new00"
    models += [{"id": f"r{number}", "group": "R", "lineage": "resnet_kind",
                "arch_type": "convolutional", "recipe_tag": "yes"}
               for number in range(5)]
    models += [{"id": f"s{number}", "group": "S", "lineage": f"seen{number // 2}",
                "arch_type": "convolutional", "recipe_tag": "no"}
               for number in range(6)]
    return models


def test_the_test_of_the_frozen_choices_on_fabricated_models(tmp_path, monkeypatch):
    fabricate(tmp_path, zoo_list.load())                  # the 100 models of A16
    path, models, report = (tmp_path / "rules.json", tmp_path / "models.json",
                            tmp_path / "j.md")
    listed = timm_list()
    models.write_text(json.dumps({"models": listed, "pilot": []}))

    def run(*extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_j", "--results-root", str(tmp_path), "--rules", str(path),
            "--models-file", str(models), "--out", str(report), *extra])
        aj.main()
        return report.read_text()

    text = run("freeze")
    assert "Models of the fit: 100, in 11 lineages. Inputs of the rule: 69" in text
    with pytest.raises(SystemExit, match="frozen once"):
        run("freeze")
    with pytest.raises(SystemExit, match="does not have the frozen hash"):
        run("test")
    with pytest.raises(SystemExit, match="no result files"):
        run("test", "--allow-other-files")
    fabricate(tmp_path, listed[:-1], folder=aj.RESULTS)
    with pytest.raises(SystemExit, match="not complete.*s5"):
        run("test", "--allow-other-files")
    fabricate(tmp_path, listed, folder=aj.RESULTS)
    text = run("test", "--allow-other-files")
    assert "No model leaves." in text and "**Gate M passes:**" in text
    assert "## Group N on the far sets: the outcome" in text
    assert "**Outcome: " in text and "**Second planned comparison: T" in text
    for title in ("## Group N: the selection problem", "## Group N: fits that saw",
                  "## Group N: negative controls", "## Group N: shortlists and AUGRC",
                  "## Group N: inputs outside the range",
                  "## Group R, one architecture and many recipes",
                  "## Group S, seen lineages", "## Models",
                  "## Target near (secondary", "## Target textures (secondary",
                  "## Target imagenet_o (secondary"):
        assert title in text
    assert "5 weights of ResNet-50" in text
    assert "**The rule adds inside one architecture: " in text
    assert "500 draws" in text and "By type, over the models" in text
    # Each choice comes from the frozen file, with no fit on the new models.
    frozen = aj.load(path)
    tables, facts = aj.ai.read(tmp_path, aj.RESULTS)
    frame = pd.DataFrame(listed).set_index("id")
    metrics = aj.ai.inputs_of(tables, list(ALL_CSFS)).loc[frame.index]
    picks = aj.choose(frozen["groups"]["far"], metrics,
                      facts.set_index("model")["val_accuracy"], aj.cells_of(frame))
    assert set(picks[aj.ONE]) == {frozen["groups"]["far"][aj.ONE]}
    assert list(picks[aj.RULE]) == frozen["groups"]["far"]["rule"].choose(metrics)
    # Gate W removes a model. A group with more than 10% removed has no reading.
    fabricate(tmp_path, listed, wrong_accuracy=("n03_0", "r1"), folder=aj.RESULTS)
    text = run("test", "--allow-other-files")
    assert "Models that leave (2)" in text and "gate W" in text
    assert "**No reading:** more than 10% of the models of group R leave" in text
    assert "**Outcome: " in text
    # A score with an error removes its model.
    fabricate(tmp_path, listed, failing=("n05_1", "pNML"), folder=aj.RESULTS)
    assert "gate S: pNML" in run("test", "--allow-other-files")
    # A model with no result file counts as a model that leaves.
    (tmp_path / aj.RESULTS / "s5.json").unlink()
    assert "no result file" in run("test", "--allow-other-files", "--allow-missing")
    # Fewer than 20 lineages of group N: no outcome letter.
    for file in (tmp_path / aj.RESULTS).glob("*.json"):
        file.unlink()
    few = timm_list(12)
    models.write_text(json.dumps({"models": few, "pilot": []}))
    fabricate(tmp_path, few, folder=aj.RESULTS)
    with pytest.raises(SystemExit, match="no outcome letter"):
        run("test", "--allow-other-files")
    text = report.read_text()
    assert "**No outcome letter:** fewer than 20 lineages" in text
    assert "**Outcome: " not in text and "## Group S, seen lineages" in text
    (tmp_path / aj.RESULTS / "other.json").write_text(
        (tmp_path / aj.RESULTS / "r0.json").read_text().replace('"r0"', '"other"'))
    with pytest.raises(SystemExit, match="not in the list"):
        run("test", "--allow-other-files")


def test_labels_of_the_second_comparison():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({aj.RULE: rng.uniform(0.2, 0.6, 30)})
    below = base.assign(**{aj.ONE: base[aj.RULE] + 1.0, aj.TAG: base[aj.RULE] + 0.5,
                           aj.TAG_MODELS: base[aj.RULE] + 0.4})
    assert aj.label(below).startswith("T+") and aj.outcome(below).startswith("A")
    one = below.assign(**{aj.TAG_MODELS: base[aj.RULE] + rng.normal(0, 0.3, 30)})
    assert aj.label(one).startswith("T0")
    above = below.assign(**{aj.TAG: base[aj.RULE] - 0.2})
    assert aj.label(above).startswith("T-")
    high = below + 2.0
    assert aj.outcome(high).startswith("F") and "below the fixed score" in aj.outcome(
        high)
    tie = below.assign(**{aj.ONE: base[aj.RULE] + rng.normal(0, 0.3, 30)})
    assert aj.outcome(tie).startswith("H")
