import itertools

import pytest
import torch
from torchvision import models

from dynamics.config import ARCHS, LAYERS, RECIPES
from dynamics.model import (
    block_of,
    build_model,
    extract,
    forward_stages,
    head,
    normalize,
    weight_norms,
)
from dynamics.train import EpochSampler, learning_rate


def test_stage_features_and_logits_match_the_model():
    torch.manual_seed(0)
    model = build_model(7).eval()
    images = torch.rand(3, 3, 64, 64)
    with torch.no_grad():
        pooled, logits = forward_stages(model, images)
        direct = model(normalize(images))
    assert [p.shape[1] for p in pooled] == [64, 128, 256, 512]
    assert torch.allclose(logits, direct, atol=1e-6)
    assert torch.allclose(model.fc(pooled[-1]), logits)


@pytest.mark.parametrize(("arch", "widths"), [
    ("resnet18", [64, 128, 256, 512]), ("resnet50", [256, 512, 1024, 2048]),
    ("densenet121", [256, 512, 1024, 1024])])
def test_every_architecture_gives_four_stages_and_its_own_logits(arch, widths):
    torch.manual_seed(0)
    model = build_model(7, arch).eval()
    images = torch.rand(3, 3, 64, 64)
    with torch.no_grad():
        pooled, logits = forward_stages(model, images)
        direct = model(normalize(images))
    assert [p.shape[1] for p in pooled] == widths
    # The last stage is the penultimate feature of the model's own forward pass.
    assert torch.allclose(logits, direct, atol=1e-5)
    assert torch.allclose(head(model)(pooled[-1]), logits)
    assert head(model).in_features == widths[-1] and head(model).out_features == 7


@pytest.mark.parametrize("arch", ARCHS)
def test_weight_norms_count_every_convolution_once(arch):
    model = build_model(5, arch)
    norms = weight_norms(model)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    total = sum(float(m.weight.detach().double().pow(2).sum()) for m in convs)
    assert norms["conv_norm"] ** 2 == pytest.approx(total)
    blocks = [norms[f"conv_norm_{name}"] for name in ("stem", *LAYERS)]
    assert all(value > 0 for value in blocks)
    assert sum(value ** 2 for value in blocks) == pytest.approx(total)
    assert norms["fc_norm"] == pytest.approx(float(head(model).weight.norm()))
    assert set(norms) == {"conv_norm", "bn_scale_norm", "fc_norm",
                          *(f"conv_norm_{name}" for name in ("stem", *LAYERS))}


def test_blocks_of_the_two_families_and_unknown_architecture():
    assert block_of("conv1") == "stem" and block_of("layer3.0.downsample.0") == "stage3"
    assert block_of("features.conv0") == "stem"
    assert block_of("features.denseblock2.denselayer1.conv1") == "stage2"
    assert block_of("features.transition1.conv") == "stage2"
    assert isinstance(build_model(3, "densenet121"), models.DenseNet)
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model(3, "vit_b_16")


def test_extract_is_deterministic_and_ordered():
    torch.manual_seed(0)
    model = build_model(4)
    data = torch.utils.data.TensorDataset(torch.rand(10, 3, 32, 32),
                                          torch.arange(10) % 4)
    first = extract(model, data, torch.device("cpu"), batch_size=4, workers=0)
    second = extract(model, data, torch.device("cpu"), batch_size=3, workers=0)
    assert set(first) == {*LAYERS, "logits", "labels"}
    assert first["labels"].tolist() == (torch.arange(10) % 4).tolist()
    assert torch.allclose(first["logits"], second["logits"], atol=1e-5)
    assert first["stage4"].dtype == torch.float32


def test_learning_rate_schedule():
    steps = 10
    peak = 0.05
    assert learning_rate(5 * steps - 1, steps, 200, peak) == peak
    assert learning_rate(0, steps, 200, peak) == peak / (5 * steps)
    assert learning_rate(200 * steps - 1, steps, 200, peak) < 1e-6
    after = [learning_rate(step, steps, 200, peak)
             for step in range(5 * steps, 200 * steps)]
    assert all(a >= b for a, b in itertools.pairwise(after))


def test_epoch_sampler_depends_only_on_seed_and_epoch():
    sampler, other = EpochSampler(50, seed=3), EpochSampler(50, seed=3)
    sampler.epoch = other.epoch = 4
    first = list(sampler)
    assert first == list(other) and sorted(first) == list(range(50))
    other.epoch = 5
    assert first != list(other)


def test_stage_d_schedule_decays_to_a_floor_and_stays_there():
    recipe, steps = RECIPES["D1"], 10

    def rate(epoch: float) -> float:
        return learning_rate(int(epoch * steps), steps, recipe.epochs, recipe.lr,
                             floor=recipe.lr_floor, decay_epochs=recipe.decay_epochs)

    assert rate(5) == recipe.lr
    assert recipe.lr_floor < rate(50) < recipe.lr
    assert rate(99.9) > recipe.lr_floor
    assert rate(100) == rate(300) == rate(499.9) == recipe.lr_floor == 0.01
    decay = [rate(epoch) for epoch in range(5, 101)]
    assert all(a >= b for a, b in itertools.pairwise(decay))


def test_a_short_run_of_a_long_recipe_ends_at_the_floor():
    recipe, steps = RECIPES["D2"], 10
    last = learning_rate(2 * steps - 1, steps, 2, recipe.lr, warmup_epochs=1,
                         floor=recipe.lr_floor, decay_epochs=recipe.decay_epochs)
    assert abs(last - recipe.lr_floor) < 0.005


def test_stage_d_arms_differ_from_the_control_in_one_setting():
    control = RECIPES["D1"]
    assert not control.augment and control.amp and control.epochs == 500
    changes = {"D2": {"augment": True}, "D3": {"label_smoothing": 0.1},
               "D4": {"weight_decay": 5e-5}, "D5": {"amp": False}}
    for name, change in changes.items():
        other = RECIPES[name]
        different = {key for key in vars(control)
                     if getattr(control, key) != getattr(other, key)}
        assert different == set(change), name
        assert all(getattr(other, key) == value for key, value in change.items())
    assert RECIPES["A"].epochs == 200 and RECIPES["A"].lr_floor == 0.0
