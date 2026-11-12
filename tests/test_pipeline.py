"""End to end on tiny images: train, evaluate, recheck, and analyze."""
import csv
import json

import pytest
import torch

from dynamics import analyze_e, analyze_step0, run_stage, train
from dynamics.config import RunSpec
from dynamics.scores import ALL_CSFS


@pytest.fixture(scope="module")
def pipeline(tiny_data, tmp_path_factory):
    """The output root of two tiny runs, and a function that runs a stage again."""
    root = tmp_path_factory.mktemp("out")
    common = ["--out-root", str(root), "--imagenette-dir", str(tiny_data.imagenette),
              "--imagewoof-dir", str(tiny_data.imagewoof), "--textures-dir",
              str(tiny_data.textures), "--epochs", "1", "--image-size", "32",
              "--workers", "0", "--device", "cpu"]

    def stage(*extra: str) -> None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("sys.argv", ["run_stage", *extra, *common])
            run_stage.main()

    stage("--only", "B_seed0", "A_seed0_hold1")
    stage("--only", "B_seed0", "A_seed0_hold1", "--recheck")
    stage("--only", "B_seed0", "--skip-train", "--precision", "float32")
    extra = tiny_data.imagenette.parent / "openood"
    stage("--only", "B_seed0", "--skip-train", "--suite", "step0", "--extra-limit",
          "20", "--extra-set", f"places={extra / 'places'}", "--extra-set",
          f"ssb_hard={extra / 'ssb_hard'}")
    return root, stage


@pytest.fixture(scope="module")
def out_root(pipeline):
    return pipeline[0]


def test_run_folders(out_root):
    for name, classes in (("B_seed0", 10), ("A_seed0_hold1", 9)):
        folder = out_root / "runs" / name
        assert (folder / "train.done").exists() and not (folder / "last.pt").exists()
        assert sorted(p.name for p in (folder / "checkpoints").iterdir()) == [
            "epoch_000.pt", "epoch_001.pt"]
        assert len((folder / "train_log.csv").read_text().splitlines()) == 2
        meta = json.loads((folder / "meta.json").read_text())
        assert len(meta["classes"]) == classes
        out = json.loads((folder / "results_f64" / "epoch_001.json").read_text())
        assert out["n_classes"] == classes and out["errors"] == {}
        assert out["precision"] == "float64"
        sets = {row["set"] for row in out["metrics"]}
        assert ("heldout" in sets) == (classes == 9)
        assert len(out["metrics"]) == len(ALL_CSFS) * len(sets)
        assert out["checks"]["g4_max_gap"] < 1e-9
    assert not (out_root / "runs" / "A_seed0_hold1" / "results_f64_recheck").exists()


def test_second_extraction_gives_the_same_metrics(out_root):
    folder = out_root / "runs" / "B_seed0"
    first = json.loads((folder / "results_f64" / "epoch_001.json").read_text())
    again = json.loads((folder / "results_f64_recheck" / "epoch_001.json")
                       .read_text())
    assert first["metrics"] == again["metrics"]


def test_each_precision_writes_to_its_own_folder(out_root, tmp_path, monkeypatch):
    folder = out_root / "runs" / "B_seed0"
    single = json.loads((folder / "results" / "epoch_001.json").read_text())
    double = json.loads((folder / "results_f64" / "epoch_001.json").read_text())
    assert single["precision"] == "float32" and double["precision"] == "float64"
    assert single["errors"] == {} and len(double["metrics"]) == len(single["metrics"])
    report = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["analyze_e", "--results-root", str(out_root),
                                     "--precision", "float32", "--out", str(report)])
    analyze_e.main()
    assert "Precision of the fitted CSFs: float32" in report.read_text()


def test_suite_adds_sets_and_keeps_the_base_results(out_root, tmp_path, monkeypatch):
    folder = out_root / "runs" / "B_seed0"
    base = json.loads((folder / "results_f64" / "epoch_001.json").read_text())
    suite = json.loads((folder / "results_f64_step0" / "epoch_001.json").read_text())
    assert suite["counts"]["places"] == 20 and suite["counts"]["ssb_hard"] == 18
    assert suite["errors"] == {} and suite["layer_errors"] == {}
    key = ("textures", "Maha")
    first = {(row["set"], row["csf"]): row["auroc"] for row in base["metrics"]}
    second = {(row["set"], row["csf"]): row["auroc"] for row in suite["metrics"]}
    assert set(first) < set(second) and ("places", "kNN") in second
    assert all(first[cell] == second[cell] for cell in first), key
    assert {row["layer"] for row in suite["layer_metrics"]} == {
        "stage1", "stage2", "stage3"}
    report = tmp_path / "step0.md"
    monkeypatch.setattr("sys.argv", ["analyze_step0", "--results-root",
                                     str(out_root), "--out", str(report)])
    analyze_step0.main()
    assert "Rule 1: not decided" in report.read_text()


def test_extra_sets_need_a_suite_name(pipeline):
    with pytest.raises(SystemExit):
        pipeline[1]("--only", "B_seed0", "--skip-train", "--extra-set", "places=/tmp")


def test_a_complete_run_is_skipped(pipeline, capsys):
    pipeline[1]("--only", "B_seed0")
    printed = capsys.readouterr().out
    assert "already complete" in printed and "nothing to do" in printed


def test_analysis_runs_on_one_main_run(out_root, tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["analyze_e", "--results-root", str(out_root),
                                     "--out", str(report)])
    analyze_e.main()
    text = report.read_text()
    assert "e_main`: 1 of 10 runs" in text and "## E7" in text
    gate = next(line for line in text.splitlines() if "G3, determinism" in line)
    assert " pass " in gate


def test_training_resumes_after_an_interruption(tiny_data, tmp_path, monkeypatch):
    spec, device = RunSpec("A", 1), torch.device("cpu")
    settings = {"epochs": 2, "workers": 0, "image_size": 32}
    real, calls = train.save_atomic, []

    def stop_in_epoch_two(payload, path):
        calls.append(path.name)
        if len(calls) == 4:
            raise RuntimeError("stopped")
        real(payload, path)

    monkeypatch.setattr(train, "save_atomic", stop_in_epoch_two)
    with pytest.raises(RuntimeError, match="stopped"):
        train.train_run(spec, tiny_data, tmp_path, device, **settings)
    folder = tmp_path / "runs" / spec.name
    assert (folder / "last.pt").exists() and not (folder / "train.done").exists()

    monkeypatch.setattr(train, "save_atomic", real)
    train.train_run(spec, tiny_data, tmp_path, device, **settings)
    assert (folder / "train.done").exists() and not (folder / "last.pt").exists()
    assert sorted(p.name for p in (folder / "checkpoints").iterdir()) == [
        "epoch_000.pt", "epoch_001.pt", "epoch_002.pt"]
    with (folder / "train_log.csv").open() as handle:
        assert [row[0] for row in csv.reader(handle)] == ["epoch", "1", "2"]
