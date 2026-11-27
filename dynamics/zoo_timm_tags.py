"""Amendment A17: the tables that a person made for the list of timm models.

Every table is fixed before any model of the list is evaluated. The list script
`dynamics.zoo_timm_list` copies the rows into the frozen list.

`zoo_timm_recipes.json` holds, for each drawn weight, what its documentation says
about label smoothing, mixup, CutMix, distillation, and other training data, with
the source. The recipe tag follows from these fields by one rule.

Protocol: ../documentation/imagenet1k_second_library_amendment_2026-11-26.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPES_PATH = HERE / "zoo_timm_recipes.json"
#: Class-name marks of attention between positions. The marks also cover linear
#: attention, cross-covariance attention, and lambda layers.
ATTENTION_MARKS = ("Attention", "Attn", "XCA", "GPSA", "MHSA", "SHSA", "LiteMLA",
                   "LambdaLayer")
#: Layers with such a mark that are attention over channels, or a pooling head.
CHANNEL_ATTENTION = (
    "SEModule", "SqueezeExcite", "SqueezeExciteCl", "EffectiveSEModule",
    "EcaModule", "CecaModule", "CbamModule", "LightCbamModule", "ChannelAttn",
    "LightChannelAttn", "SpatialAttn", "LightSpatialAttn", "GlobalContext",
    "GatherExcite", "SelectiveKernelAttn", "SelectiveKernel", "SplitAttn",
    "AttentionPool2d", "RotAttentionPool2d", "AttentionPoolLatent",
    "AttentionPoolPrr")
#: Architecture -> (type, reason), where the code test gives another type.
TYPE_FIXES: dict[str, tuple[str, str]] = {}
#: The fields of a recipe row that decide the tag.
RECIPE_FIELDS = ("label_smoothing", "mixup", "cutmix")
#: The fields of a recipe row that remove a weight if they are `yes`.
REMOVING_FIELDS = ("distillation", "extra_data")


def recipe_tag(row: dict) -> str:
    """`yes` if the documented recipe has label smoothing, mixup, or CutMix, `no`
    if the documentation shows that it has none, and `unknown` in the other cases."""
    values = [row[field] for field in RECIPE_FIELDS]
    if "yes" in values:
        return "yes"
    return "no" if all(value == "no" for value in values) else "unknown"


def load_rows(path: Path = RECIPES_PATH) -> dict[str, dict]:
    return json.loads(Path(path).read_text())["weights"]


ROWS = load_rows()
#: Weight -> reason, for a weight whose documentation shows distillation or other
#: data that its tag does not mark.
NOT_ELIGIBLE = {
    name: "the documentation shows " + " and ".join(
        field.replace("_", " ") for field in REMOVING_FIELDS if row[field] == "yes")
    + f" ({row['source']})"
    for name, row in ROWS.items()
    if any(row[field] == "yes" for field in REMOVING_FIELDS)}
#: Weight -> (recipe tag, source, evidence).
RECIPES = {name: (recipe_tag(row), row["source"], row["evidence"])
           for name, row in ROWS.items() if name not in NOT_ELIGIBLE}
