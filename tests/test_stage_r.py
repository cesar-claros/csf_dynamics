import json

import pandas as pd
import pytest
from test_stage_z import part_1_world

from dynamics import analyze_e, analyze_r
from dynamics.analyze_selection import FAMILY
from dynamics.config import (
    A13_EVAL_EPOCHS,
    CHECKPOINT_EPOCHS,
    RECIPES,
    STAGES,
    RunSpec,
    results_folder,
)


def test_recipes_and_stages_of_amendment_a13():
    assert len(STAGES["r"]) == 16 and len(STAGES["n"]) == 16
    assert [spec.name for spec in STAGES["r_first"]] == ["R1_seed0"]
    assert [spec.name for spec in STAGES["n_first"]] == ["N1_seed0"]
    for letter, arch in (("R", "resnet50"), ("N", "densenet121")):
        for number in range(1, 9):
            name, source = f"{letter}{number}", RECIPES[f"Z{number}"]
            # The recipe of stage `z` with one change: the architecture.
            assert RECIPES[name].arch == arch and source.arch == "resnet18"
            assert {**vars(RECIPES[name]), "arch": "resnet18"} == vars(source)
            assert FAMILY[name] == FAMILY[f"Z{number}"]
    assert all(recipe.arch == "resnet18" for name, recipe in RECIPES.items()
               if name[0] not in "RN")
    # The evaluated checkpoints are saved checkpoints, and the last one is the end.
    assert set(A13_EVAL_EPOCHS) <= set(CHECKPOINT_EPOCHS)
    assert A13_EVAL_EPOCHS[-1] == RECIPES["R1"].epochs
    assert analyze_r.PARTS[1][1:] == ("r", "resnet50")
    assert analyze_r.PARTS[2][1:] == ("n", "densenet121")


def test_gate_s_reads_the_final_checkpoint_only():
    checks = pd.DataFrame([
        {"run": "R1_seed0", "epoch": 100, "failed": "kNN", "nonfinite": ""},
        {"run": "R1_seed0", "epoch": 200, "failed": "", "nonfinite": ""},
        {"run": "R2_seed0", "epoch": 200, "failed": "", "nonfinite": "pNML"},
        {"run": "R3_seed0", "epoch": 200, "failed": "Maha", "nonfinite": ""}])
    tables = analyze_e.Tables(metrics=pd.DataFrame(), descriptors=pd.DataFrame(),
                              placement=pd.DataFrame(), checks=checks)
    failures = analyze_r.score_failures(tables)
    assert list(failures["run"]) == ["R2_seed0", "R3_seed0"]


def test_frozen_rule_on_fabricated_runs_of_another_architecture(tmp_path,
                                                                monkeypatch):
    write, run_z, columns, _ = part_1_world(tmp_path, monkeypatch)
    monkeypatch.setitem(STAGES, "z", [RunSpec("Z1", 0)])
    path, report = tmp_path / "z.json", tmp_path / "r.md"
    run_z("freeze", 1, path)
    new = [RunSpec("R1", 0), RunSpec("R2", 0), RunSpec("R3", 0)]
    monkeypatch.setitem(STAGES, "r", new)

    def run(*extra: str) -> str:
        monkeypatch.setattr("sys.argv", [
            "analyze_r", "--part", "1", "--results-root", str(tmp_path), "--rules",
            str(path), "--out", str(report), *extra])
        analyze_r.main()
        return report.read_text()

    def meta(arch: str) -> None:
        for spec in new:
            (tmp_path / "runs" / spec.name / "meta.json").write_text(
                json.dumps({"arch": arch}))

    with pytest.raises(SystemExit, match="not the frozen one"):
        run()
    with pytest.raises(SystemExit, match="no results of stage r"):
        run("--allow-other-rules")
    for spec in new:
        write(spec, results_folder(suite="step0"), columns, True, kept=True)
    meta("resnet18")
    with pytest.raises(SystemExit, match="runs that are not resnet50"):
        run("--allow-other-rules")
    meta("resnet50")
    text = run("--allow-other-rules")
    assert text.startswith("# Amendment A13, part 1, ResNet-50")
    assert "**Gate S passes.**" in text and "**Gate M passes**" in text
    assert "3 of 3 runs" in text and "**Outcome:" in text
    assert "## Question 3 (secondary)" in text and "the rule has" in text
    # A score with an error at the final checkpoint stops the part before a reading.
    folder = tmp_path / "runs" / "R2_seed0" / results_folder(suite="step0")
    last = json.loads((folder / "epoch_200.json").read_text())
    last["errors"] = {"pNML": "singular matrix"}
    (folder / "epoch_200.json").write_text(json.dumps(last))
    with pytest.raises(SystemExit, match="gate S fails"):
        run("--allow-other-rules")
    assert "**Gate S fails.**" in report.read_text()
    assert "**Outcome:" not in report.read_text()
