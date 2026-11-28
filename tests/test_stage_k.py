import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_stage_i import fabricate

from dynamics import analyze_k as ak
from dynamics import standin as st
from dynamics import standin_list as sl
from dynamics import zoo_list, zoo_timm_list
from dynamics.scores import ALL_CSFS

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image(seed: int, height: int = 64, width: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(3, height, width, generator=generator) - MEAN) / STD


# ---- the transformations ----

@pytest.mark.parametrize("task", st.IMAGE_TASKS)
def test_each_task_gives_one_fixed_image_of_the_same_size(task):
    for height, width in ((64, 64), (75, 83)):
        first, others = image(0, height, width), [image(k, height, width)
                                                  for k in (1, 2, 3)]
        out = st.apply(task, first, others, MEAN, STD, st.task_rng(task, 4))
        again = st.apply(task, first, others, MEAN, STD, st.task_rng(task, 4))
        assert out.shape == first.shape and torch.isfinite(out).all()
        assert torch.equal(out, again)                    # the seed fixes the image
        assert not torch.allclose(out, first)             # the image changes
    with pytest.raises(ValueError, match="not an image task"):
        st.apply("featblend", first, others, MEAN, STD, st.task_rng("featblend", 0))


def test_what_each_task_keeps():
    first, others = image(0), [image(1), image(2), image(3)]

    def pixels(tensor):
        return tensor * STD + MEAN

    def run(task, index=0):
        return st.apply(task, first, others, MEAN, STD, st.task_rng(task, index))

    # Patches: the same pixel values in another order, for every grid.
    for task in st.GRIDS:
        assert torch.allclose(run(task).flatten(1).sort(dim=1).values,
                              first.flatten(1).sort(dim=1).values)
    # The phase task keeps the amplitude spectrum and the mean of each channel.
    out = run("phase")
    assert torch.allclose(torch.fft.fft2(out).abs(), torch.fft.fft2(first).abs(),
                          atol=1e-3)
    assert torch.allclose(out.mean(dim=(1, 2)), first.mean(dim=(1, 2)), atol=1e-4)
    # The swap has the amplitude of the partner and the phase of the image.
    out = run("ampswap")
    assert torch.allclose(torch.fft.fft2(out).abs(), torch.fft.fft2(others[0]).abs(),
                          atol=1e-3)
    # The pasted rectangle has half of the area and comes from the partner.
    out = run("cutpaste")
    changed = (out != first).any(dim=0)
    assert 0.45 < changed.float().mean() < 0.55
    assert torch.equal(out[:, changed], others[0][:, changed])
    # The mosaic has one quadrant of each of the four images.
    out = run("mosaic")
    assert torch.equal(out[:, :32, :32], first[:, :32, :32])
    assert torch.equal(out[:, :32, 32:], others[0][:, :32, 32:])
    assert torch.equal(out[:, 32:, :32], others[1][:, 32:, :32])
    assert torch.equal(out[:, 32:, 32:], others[2][:, 32:, 32:])
    assert torch.allclose(run("blend30"), 0.7 * first + 0.3 * others[0])
    # Tasks in pixel values.
    assert torch.allclose(pixels(run("invert")), 1 - pixels(first), atol=1e-5)
    assert torch.allclose(pixels(run("geoblend")),
                          torch.sqrt(pixels(first) * pixels(others[0])), atol=1e-5)
    noise = pixels(run("noise"))
    assert 0 <= noise.min() and noise.max() <= 1 and abs(noise.mean() - 0.5) < 0.02
    assert torch.equal(run("noise"), st.apply("noise", others[0], others, MEAN, STD,
                                              st.task_rng("noise", 0)))
    assert not torch.equal(run("noise"), run("noise", 1))
    # The blur removes the high frequencies, and the other task keeps only them.
    blurred, high = run("blur"), run("highpass")
    assert blurred.var() < 0.2 * first.var()
    # The blur changes the mean a little at the border, so the mean stays nearly.
    assert torch.allclose(high.mean(dim=(1, 2)), first.mean(dim=(1, 2)), atol=0.01)
    assert torch.allclose(high - first.mean(dim=(1, 2), keepdim=True),
                          first - blurred, atol=1e-5)
    # The mask hides 75% of the cells of a 14 by 14 grid.
    larger = image(5, 140, 140)
    masked = st.apply("mask75", larger, [], MEAN, STD, st.task_rng("mask75", 0))
    hidden = (masked != larger).any(dim=0)
    assert abs(float(hidden.float().mean()) - 0.75) < 0.01
    assert torch.allclose(masked[:, hidden].mean(dim=1), larger.mean(dim=(1, 2)),
                          atol=1e-5)


def test_subset_and_partners_of_the_new_tasks():
    labels = np.repeat(np.arange(6), 12)
    kept = np.sort(np.concatenate([np.flatnonzero(labels == c)[:10]
                                   for c in range(6)]))
    small = st.subset_of(kept, labels)
    assert len(small) == 30 and set(small) <= set(kept)
    assert all((labels[small] == c).sum() == st.PER_CLASS for c in range(6))
    partners = st.partners_of(labels[small])
    assert partners.shape == (30, 3)
    own = labels[small]
    for index in range(30):
        classes = own[partners[index]]
        assert own[index] not in classes and len(set(classes)) == 3
    assert np.array_equal(partners, st.partners_of(labels[small]))
    with pytest.raises(ValueError, match="four classes"):
        st.partners_of(np.array([0, 0, 1, 1, 2, 2]))
    features = torch.arange(12.0).view(6, 2)
    blend = st.feature_blend(features, np.array([[1, 0, 0]] * 6))
    assert torch.allclose(blend, 0.5 * (features + features[1]))
    with pytest.raises(ValueError, match="not an image task"):
        st.TaskSet([], "featblend", partners, MEAN, STD)


# ---- the list and the runner ----

def test_frozen_list_of_the_pilot_models():
    stored = json.loads(sl.PATH.read_text())
    assert sl.file_hash(sl.PATH) == ak.MODELS_SHA256
    assert stored == sl.table()                       # the two lists give the file
    models = pd.DataFrame(stored["models"])
    assert len(models) == 40 and models["id"].is_unique
    first, second = models[models["pool"] == "torchvision"], models[
        models["pool"] == "timm"]
    assert len(first) == len(second) == sl.PER_POOL
    known = {entry["id"]: entry["lineage"] for entry in zoo_list.load()}
    assert set(first["id"]) <= set(known)
    assert set(first["lineage"]) == set(known.values())       # every lineage of A16
    new = {entry["id"] for entry in zoo_timm_list.load() if entry["group"] == "N"}
    assert set(second["id"]) <= new - set(sl.LEFT_A17)
    assert second["lineage"].is_unique                      # one model for a lineage
    assert stored["pilot"] == [first["id"].iloc[0], second["id"].iloc[0]]


def test_the_pilot_run_on_tiny_images(tiny_data, tmp_path, monkeypatch):
    from torchvision import models, transforms

    from dynamics import zoo, zoo_pilot

    def fake(entry):
        torch.manual_seed(0)
        return models.resnet18(num_classes=10).eval(), transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist())])

    monkeypatch.setattr(zoo, "load_model", fake)
    entry = {"id": "resnet18-v1", "model": "resnet18", "weights": "IMAGENET1K_V1",
             "acc1": 69.758, "crop_size": [32], "pool": "torchvision"}
    mean, std = zoo_pilot.normalization(entry)
    assert torch.allclose(mean, MEAN) and torch.allclose(std, STD)
    zoo_pilot.evaluate_model(entry, tiny_data.imagenette, tmp_path,
                             torch.device("cpu"), workers=0)
    path = tmp_path / zoo_pilot.RESULTS / "resnet18-v1.json"
    out = json.loads(path.read_text())
    tasks = {*zoo_pilot.OLD_TASKS, *st.NEW_TASKS}
    assert {row["set"] for row in out["metrics"]} == {*tasks, "id_errors"}
    assert len(out["metrics"]) == (len(tasks) + 1) * len(ALL_CSFS)
    assert out["counts"]["blend"] == out["counts"]["shuffle"] == 100
    assert all(out["counts"][task] == 50 for task in st.NEW_TASKS)
    assert out["counts"]["val"] == 120 and out["counts"]["train"] == 240
    assert out["new_tasks"] == list(st.NEW_TASKS) and out["pool"] == "torchvision"
    assert out["errors"] == {} and not any(out["nonfinite"].values())
    assert out["checks"]["argmax_agreement"] == 1.0
    stamp = path.stat().st_mtime_ns                       # a second call skips it
    zoo_pilot.evaluate_model(entry, tiny_data.imagenette, tmp_path,
                             torch.device("cpu"), workers=0)
    assert path.stat().st_mtime_ns == stamp
    # The two existing image tasks are the sets of the evaluation of A16.
    sets, small, partners = zoo_pilot.task_sets(fake(entry)[1], tiny_data.imagenette,
                                                mean, std)
    old = zoo.datasets(fake(entry)[1], tiny_data.imagenette, None, [])
    for name in zoo_pilot.OLD_TASKS:
        assert torch.equal(sets[name][3][0], old[name][3][0])
    assert len(small) == 50 and partners.shape == (50, 3)
    listed = sl.load()
    assert [e["id"] for e in zoo.chosen(["pilot"], listed, sl.pilot())] == sl.pilot()


def test_the_normalization_of_each_pilot_model():
    from dynamics import zoo_pilot

    have_timm = True
    try:
        import timm
        have_timm = timm.__version__ == zoo_timm_list.TIMM
    except ImportError:
        have_timm = False
    seen = set()
    for entry in sl.load():
        if entry["pool"] == "timm" and not have_timm:
            continue
        mean, std = zoo_pilot.normalization(entry)
        assert mean.shape == std.shape == (3, 1, 1)
        assert (mean >= 0).all() and (mean < 1).all() and (std > 0).all()
        seen.add((tuple(mean.flatten().tolist()), tuple(std.flatten().tolist())))
    assert seen                                   # at least the torchvision half


# ---- the readings ----

def pilot_list(per_pool: int = 8) -> list[dict]:
    return [{"id": f"{pool[:2]}{number}", "lineage": f"{pool}{number % 4}",
             "pool": pool} for pool in ak.POOLS for number in range(per_pool)]


def fabricate_pilot(tmp_path, listed, changed=(), failing=(), unstable=()):
    """Pilot files for fabricated stored files. `good` follows the far target,
    `copy` is the blend task, `flat` has no spread, and `weak` is noise. For a
    model in `unstable`, the score that gate C leaves out has another value."""
    folder = tmp_path / ak.RESULTS
    folder.mkdir(exist_ok=True)
    for number, entry in enumerate(listed):
        stored = json.loads((tmp_path / ak.STORED[entry["pool"]]
                             / f"{entry['id']}.json").read_text())
        rng = np.random.default_rng(100 + number)
        rows = [dict(row) for row in stored["metrics"] if row["set"] in ak.OLD_TASKS]
        far = pd.DataFrame([row for row in stored["metrics"]
                            if row["set"] in ak.ai.GROUPS["far"]]).groupby(
                                "csf")["auroc_correct"].mean()
        blend = {row["csf"]: row["auroc"] for row in rows if row["set"] == "blend"}
        for csf in ALL_CSFS:
            values = {"good": far[csf] + rng.normal(0, 0.002), "copy": blend[csf],
                      "flat": 0.7 + rng.normal(0, 0.001), "weak": rng.uniform(0.3, 0.9)}
            rows += [{"set": task, "csf": csf, "auroc": float(value)}
                     for task, value in values.items()]
        if entry["id"] in changed:
            rows[0]["auroc"] += 0.05
        if entry["id"] in unstable:
            next(row for row in rows if row["set"] == "shuffle"
                 and row["csf"] in ak.UNSTABLE)["auroc"] += 0.05
        out = {**stored, "metrics": rows, "pool": entry["pool"],
               "errors": {"pNML": "LinAlgError"} if entry["id"] in failing else {}}
        (folder / f"{entry['id']}.json").write_text(json.dumps(out))


def test_readings_of_the_pilot_on_fabricated_models(tmp_path, monkeypatch):
    listed = pilot_list()
    for pool in ak.POOLS:
        fabricate(tmp_path, [e for e in listed if e["pool"] == pool],
                  folder=ak.STORED[pool])
    path, out = tmp_path / "models.json", tmp_path / "k.md"
    path.write_text(json.dumps({"models": listed, "pilot": []}))

    def run(*extra: str) -> str:
        monkeypatch.setattr("sys.argv", ["analyze_k", "--results-root", str(tmp_path),
                                         "--models-file", str(path), "--out",
                                         str(out), *extra])
        ak.main()
        return out.read_text()

    with pytest.raises(SystemExit, match="frozen hash"):
        run()
    with pytest.raises(SystemExit, match="no result files"):
        run("--allow-other-list")
    fabricate_pilot(tmp_path, listed)
    text = run("--allow-other-list")
    assert "Models that stay: 16 (torchvision 8, timm 8)" in text
    assert "No model leaves." in text and "**Gate C** (22 scores" in text
    for title in ("## Reading 1: rank agreement", "## Each new task against",
                  "## Readings 2 and 3", "## Decision: accepted tasks",
                  "## Reading 4: the direct choice"):
        assert title in text
    assert "**Accepted (1 of at most 2):**" in text
    frame = pd.DataFrame(listed).set_index("id")
    tables, _ = ak.ai.read(tmp_path, ak.RESULTS)
    new = ak.wide(tables, [*ak.OLD_TASKS, "good", "copy", "flat", "weak"]).reindex(
        frame.index)
    _, targets, _ = ak.stored_tables(tmp_path, frame)
    found = ak.readings(frame, new, targets)
    table = found["candidates"].set_index(["task", "target"])
    # `good` follows the far sets: it passes for them, with a large gain.
    assert table.loc[("good", "far"), "passes"] and table.loc[("good", "far"),
                                                              "gain"] > ak.LARGE
    assert list(found["accepted"]["task"]) == ["good"]
    assert "A gain of 0.15 or more on the far sets:** good" in text
    # `copy` is an existing task and `flat` has no spread: they do not pass.
    assert found["between"].loc["copy", "blend"] == pytest.approx(1.0)
    assert not table.loc["copy", "passes"].any()
    assert found["spread"]["flat"] < ak.SPREAD_LIMIT
    assert not table.loc["flat", "passes"].any() and not table.loc["weak",
                                                                   "passes"].any()
    # The direct choice with the accepted task is in the table.
    assert "the three and good" in set(found["direct"]["tasks"])
    assert set(found["direct"]["choice"]) == {"mean_rank", "maximin"}
    assert "**Gate C** (22 scores, without KPCA_RecError)" in text
    assert "is above 0.01 for 0 of 16 models" in text
    # Gates: a changed value of an existing task, and a score with an error.
    fabricate_pilot(tmp_path, listed, changed=("to1",), failing=("ti2",))
    text = run("--allow-other-list")
    assert "Models that leave (2)" in text and "gate C: an existing task" in text
    assert "gate S: pNML" in text and "Models that stay: 14" in text
    # The score that gate C leaves out removes no model, and its change is reported.
    fabricate_pilot(tmp_path, listed, unstable=("to1", "to3", "ti5"))
    text = run("--allow-other-list")
    assert "No model leaves." in text and "Models that stay: 16" in text
    assert "is above 0.01 for 3 of 16 models" in text
    assert "value is 0.0500 over the models" in text
    # More than four models of a half leave: no decision.
    fabricate_pilot(tmp_path, listed, changed=tuple(f"to{n}" for n in range(5)))
    with pytest.raises(SystemExit, match="no decision"):
        run("--allow-other-list")
    assert "**The pilot gives no decision:**" in out.read_text()
    assert "## Reading 1" not in out.read_text()


def test_gate_c_leaves_the_unstable_score_out():
    assert ak.UNSTABLE == ("KPCA_RecError",) and set(ak.UNSTABLE) < set(ALL_CSFS)
    columns = pd.MultiIndex.from_product([list(ak.OLD_TASKS), list(ALL_CSFS)])
    old = pd.DataFrame(0.7, index=["m"], columns=columns)
    new = old.copy()
    new.loc["m", ("blend", "KPCA_RecError")] += 0.2
    new.loc["m", ("shuffle", "MSR")] += 0.004
    assert ak.old_gap(new, old, ["m"]).iloc[0] == pytest.approx(0.004)
    assert ak.old_gap(new, old, ["m"], unstable=True).iloc[0] == pytest.approx(0.2)


def test_at_most_two_tasks_are_accepted_for_different_targets():
    rows = [("a", "far", 0.30, True), ("a", "near", 0.10, True),
            ("b", "far", 0.25, True), ("c", "textures", 0.20, True),
            ("d", "near", 0.40, False), ("e", "far", 0.12, True)]
    table = pd.DataFrame(rows, columns=["task", "target", "gain", "passes"])
    chosen = ak.accepted(table)
    # `a` has the largest gain; `c` is the best task for another target.
    assert list(chosen["task"]) == ["a", "c"] and list(chosen["target"]) == [
        "far", "textures"]
    same = table[table["target"] == "far"]
    assert list(ak.accepted(same)["task"]) == ["a", "b"]
    assert list(ak.accepted(table, most=1)["task"]) == ["a"]
    assert ak.accepted(table[~table["passes"]]).empty


def test_rank_agreement_and_direct_choices():
    scores = ["x", "y", "z", "w"]
    first = pd.DataFrame([[0.9, 0.8, 0.7, 0.6], [0.5, 0.6, 0.7, 0.8]], columns=scores)
    second = pd.DataFrame([[0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6]], columns=scores)
    assert list(ak.rank_agreement(first, second).round(6)) == [1.0, -1.0]
    tasks = pd.concat({"one": pd.DataFrame([[0.9, 0.8, 0.2, 0.1]], columns=scores),
                       "two": pd.DataFrame([[0.1, 0.7, 0.9, 0.2]], columns=scores)},
                      axis=1)
    target = pd.DataFrame([[70.0, 90.0, 80.0, 60.0]], columns=scores)
    # `y` has the best mean rank and the best worst rank: regret 0.
    for how in ("mean_rank", "maximin"):
        assert ak.direct_regret(tasks, ["one", "two"], target, how).iloc[0] == 0.0
    # With the first task only, the choice is `x`: regret 20.
    assert ak.direct_regret(tasks, ["one"], target, "maximin").iloc[0] == 20.0
