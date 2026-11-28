"""Amendment A18, part 1: the 40 models of the pilot of stand-in tasks, drawn once.

    python -m dynamics.standin_list

20 of the 100 torchvision models of amendment A16 (one for each lineage and 9
more), and 20 models of the new timm lineages of amendment A17 (20 lineages, one
model for each), from the models that stayed in the planned analyses. The list is
written to `dynamics/frozen/a18_pilot_models.json`, which is frozen.

Protocol: ../documentation/standin_tasks_pilot_plan_2026-11-27.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dynamics import zoo_list, zoo_timm_list

HERE = Path(__file__).resolve().parent
PATH = HERE / "frozen" / "a18_pilot_models.json"
SEED = 18
PER_POOL = 20
#: The models of amendment A17 that a gate removed (run record of 2026-11-27).
LEFT_A17 = ("resmlp_big_24_224.fb_in1k", "sequencer2d_s.in1k", "sequencer2d_l.in1k",
            "convmixer_768_32.in1k", "starnet_s2.in1k")


def draw(torchvision: list[dict], timm: list[dict], seed: int = SEED,
         per_pool: int = PER_POOL) -> list[dict]:
    """The models of the pilot, the torchvision half first.

    Args:
        torchvision: The entries of the list of amendment A16.
        timm: The entries of the new lineages of amendment A17 that stayed.
    """
    rng = np.random.default_rng(seed)
    by_id = {entry["id"]: entry for entry in torchvision}
    lineages = sorted({entry["lineage"] for entry in torchvision})
    first = [str(rng.choice(sorted(e["id"] for e in torchvision
                                   if e["lineage"] == lineage)))
             for lineage in lineages]
    rest = sorted(set(by_id) - set(first))
    more = [str(name) for name in rng.choice(rest, size=per_pool - len(first),
                                             replace=False)]
    chosen = [{**by_id[name], "pool": "torchvision"} for name in [*first, *more]]
    new = sorted({entry["lineage"] for entry in timm})
    for lineage in rng.choice(new, size=per_pool, replace=False):
        names = sorted(e["id"] for e in timm if e["lineage"] == lineage)
        name = str(rng.choice(names))
        entry = next(e for e in timm if e["id"] == name)
        chosen.append({**entry, "pool": "timm"})
    return chosen


def table() -> dict:
    """The content of the frozen file."""
    timm = [entry for entry in zoo_timm_list.load()
            if entry["group"] == "N" and entry["id"] not in LEFT_A17]
    models = draw(zoo_list.load(), timm)
    return {"amendment": "A18, part 1", "seed": SEED,
            "pilot": [models[0]["id"], models[PER_POOL]["id"]], "models": models}


def load(path: Path = PATH) -> list[dict]:
    return json.loads(Path(path).read_text())["models"]


def pilot(path: Path = PATH) -> list[str]:
    return json.loads(Path(path).read_text())["pilot"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    if PATH.exists():
        raise SystemExit(f"{PATH.name} exists: the list is made once "
                         f"(sha256 {file_hash(PATH)})")
    PATH.write_text(json.dumps(table(), indent=1) + "\n")
    print(f"wrote {PATH.name}, sha256 {file_hash(PATH)}; {len(load())} models")


if __name__ == "__main__":
    main()
