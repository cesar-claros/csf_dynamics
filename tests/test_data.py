from dataclasses import replace

import numpy as np
import pytest
import torch

from dynamics.config import (
    CHECKPOINT_EPOCHS,
    STAGES,
    RunSpec,
    checkpoint_epochs,
    parse_run,
    results_folder,
)
from dynamics.data import (
    ImageList,
    blend_partners,
    cache,
    eval_transform,
    evaluation_sets,
    labelled_split,
    manifest,
    patch_permutations,
    shuffle_patches,
    thin,
    training_set,
)


def test_stages_match_the_protocol():
    assert len(CHECKPOINT_EPOCHS) == 23 and CHECKPOINT_EPOCHS[-1] == 200
    assert len(STAGES["e_main"]) == 10
    assert len(STAGES["e_heldout"]) == 20
    assert len(STAGES["c"]) == 15
    names = [spec.name for stage in STAGES.values() for spec in stage]
    assert len(names) == len(set(names))


def test_run_names_round_trip():
    for spec in (RunSpec("A", 0), RunSpec("C2", 7), RunSpec("B", 0, 9)):
        assert parse_run(spec.name) == spec
    assert RunSpec("B", 0, 9).name == "B_seed0_hold9"


def test_each_precision_has_its_own_results_folder():
    assert results_folder() == "results_f64"
    assert results_folder("float32") == "results"
    assert results_folder("float32", recheck=True) == "results_recheck"
    assert results_folder("float64", recheck=True) == "results_f64_recheck"
    with pytest.raises(ValueError):
        results_folder("float16")


def test_checkpoint_epochs_of_a_short_run():
    assert checkpoint_epochs(4) == (0, 1, 2, 3, 4)
    assert checkpoint_epochs(5) == (0, 1, 2, 3, 4, 5)
    assert checkpoint_epochs(200) == CHECKPOINT_EPOCHS


def test_thin_keeps_evenly_spaced_items():
    assert thin(list(range(10)), 3) == [0, 4, 9]
    assert thin(list(range(3)), 5) == [0, 1, 2]


def test_shuffle_patches_moves_whole_patches():
    grid, size = 4, 8
    patch_id = torch.arange(grid * grid).reshape(grid, grid)
    image = patch_id.repeat_interleave(2, 0).repeat_interleave(2, 1).float()
    image = image.expand(3, size, size).clone()
    permutation = np.random.default_rng(0).permutation(grid * grid)
    moved = shuffle_patches(image, permutation, grid)
    seen = moved[0, ::2, ::2].reshape(-1).long().numpy()
    assert np.array_equal(seen, permutation)
    assert torch.equal(shuffle_patches(image, np.arange(grid * grid), grid), image)


def test_patch_permutations_are_fixed_and_never_the_identity():
    first, second = patch_permutations(200), patch_permutations(200)
    assert np.array_equal(first, second)
    assert not (first == np.arange(16)).all(axis=1).any()
    assert (np.sort(first, axis=1) == np.arange(16)).all()


def test_blend_partners_come_from_another_class():
    labels = np.arange(60) % 5
    partners = blend_partners(labels)
    assert (labels[partners] != labels).all()
    assert np.array_equal(partners, blend_partners(labels))


def test_held_out_class_leaves_nine_remapped_classes(tiny_data):
    files, labels, classes, held = labelled_split(tiny_data.imagenette, "train", 3)
    assert len(classes) == 9 and "n1003" not in classes
    assert sorted(set(labels)) == list(range(9))
    assert len(held) == 24 and all(path.parent.name == "n1003" for path in held)
    assert all(path.parent.name == classes[label]
               for path, label in zip(files, labels))


def test_cache_returns_the_same_items(tiny_data):
    files, labels, _, _ = labelled_split(tiny_data.imagenette, "val", None, 20)
    plain = ImageList(files, labels, eval_transform(32))
    cached = cache(plain, workers=0)
    assert len(cached) == len(plain)
    for index in (0, 7, len(plain) - 1):
        assert torch.equal(cached[index][0], plain[index][0])
        assert cached[index][1] == plain[index][1]


def test_evaluation_sets_of_a_held_out_run(tiny_data):
    sets = evaluation_sets(tiny_data, held_out=2, size=32, workers=0)
    assert set(sets) == {"train", "val", "blend", "shuffle", "imagewoof", "textures",
                         "heldout"}
    assert len(sets["train"]) == 9 * 24 and len(sets["val"]) == 9 * 12
    assert len(sets["heldout"]) == 24 + 12
    assert len(sets["blend"]) == len(sets["shuffle"]) == len(sets["val"])
    assert len(sets["imagewoof"]) == 60 and len(sets["textures"]) == 40
    image, label = sets["blend"][0]
    assert image.shape == (3, 32, 32) and label == -1
    assert 0 <= float(image.min()) and float(image.max()) <= 1


def test_regime_a_training_images_equal_the_evaluation_images(tiny_data):
    train, classes = training_set(tiny_data, None, 32, augment=False)
    sets = evaluation_sets(tiny_data, None, 32, workers=0)
    assert len(classes) == 10
    assert torch.equal(train[5][0], sets["train"][5][0])


def test_extra_sets_are_capped_and_named(tiny_data):
    extra = tiny_data.imagenette.parent / "openood"
    paths = replace(tiny_data, extra=(("places", extra / "places"),
                                      ("ssb_hard", extra / "ssb_hard")),
                    extra_limit=20)
    sets = evaluation_sets(paths, None, 32, workers=0)
    assert len(sets["places"]) == 20 and len(sets["ssb_hard"]) == 18
    assert sets["places"][0][0].shape == (3, 32, 32) and sets["places"][0][1] == -1
    info = manifest(paths)
    assert info["places"] == {"images": 30, "used": 20, "id_class_folders": []}
    assert info["ssb_hard"]["used"] == 18


def test_extra_set_names_cannot_shadow_a_pipeline_set(tiny_data):
    extra = tiny_data.imagenette.parent / "openood"
    for name in ("textures", "val", "id_errors"):
        paths = replace(tiny_data, extra=((name, extra / "places"),))
        with pytest.raises(ValueError, match="reserved"):
            evaluation_sets(paths, None, 32, workers=0)


def test_manifest_flags_an_extra_set_with_an_id_class_folder(tiny_data, tmp_path):
    folder = tmp_path / "leaky" / "n1003"
    folder.mkdir(parents=True)
    source = next((tiny_data.imagenette / "val" / "n1003").iterdir())
    (folder / source.name).write_bytes(source.read_bytes())
    paths = replace(tiny_data, extra=(("leaky", tmp_path / "leaky"),))
    assert manifest(paths)["leaky"]["id_class_folders"] == ["n1003"]
