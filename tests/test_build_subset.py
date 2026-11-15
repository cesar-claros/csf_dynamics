import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from dynamics import build_subset as bs


def jpeg(width: int, height: int, value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(
        buffer, format="JPEG")
    return buffer.getvalue()


def write_parquet(path, labels, names, sizes=(200, 320)) -> None:
    images = [{"bytes": jpeg(*sizes, 10 * label % 255), "path": name}
              for label, name in zip(labels, names)]
    pq.write_table(pa.table({"image": images, "label": labels}), path,
                   row_group_size=3)


def wnid(label: int) -> str:
    return f"n{label:08d}"


@pytest.fixture
def parquet_dir(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    for part in range(2):
        labels = [(part * 7 + i) % 12 for i in range(14)]
        write_parquet(folder / f"train-0000{part}-of-00002.parquet", labels,
                      [f"{wnid(label)}_{i}_{wnid(label)}.JPEG"
                       for i, label in enumerate(labels)])
    labels = list(range(12)) * 2
    write_parquet(folder / "validation-00000-of-00001.parquet", labels,
                  [f"ILSVRC2012_val_{i:08d}_{wnid(label)}.JPEG"
                   for i, label in enumerate(labels)])
    write_parquet(folder / "test-00000-of-00001.parquet", [-1, -1], [None, None])
    return folder


def test_file_names_give_the_class_of_each_label(parquet_dir):
    assert bs.wnid_of("n01440764_10026.JPEG") == "n01440764"
    assert bs.wnid_of("ILSVRC2012_val_00000293_n01751748.JPEG") == "n01751748"
    assert bs.wnid_of(None) is None and bs.wnid_of("image.JPEG") is None
    files = bs.split_files(parquet_dir, "train")
    assert [path.name for path in files] == ["train-00000-of-00002.parquet",
                                             "train-00001-of-00002.parquet"]
    assert bs.label_names(files) == {label: wnid(label) for label in range(12)}
    with pytest.raises(SystemExit, match="no file name"):
        bs.label_names(bs.split_files(parquet_dir, "test"))


def test_class_choice_keeps_and_excludes_and_is_fixed():
    available = [wnid(i) for i in range(40)]
    keep, exclude = available[:3], available[3:8]
    chosen = bs.choose_classes(available, keep, exclude, count=10)
    assert len(chosen) == 10 and chosen == sorted(chosen)
    assert set(keep) <= set(chosen) and not set(exclude) & set(chosen)
    assert chosen == bs.choose_classes(available[::-1], keep, exclude, count=10)
    assert chosen != bs.choose_classes(available, keep, exclude, count=10, seed=1)
    with pytest.raises(ValueError, match="not in the parquet"):
        bs.choose_classes(available, ["n99999999"], exclude, count=10)


def test_resize_sets_the_shorter_side():
    assert bs.resize(jpeg(200, 320, 50)).size == (160, 256)
    assert bs.resize(jpeg(640, 320, 50)).size == (320, 160)
    assert bs.resize(jpeg(100, 100, 50), size=160).size == (160, 160)


def test_extract_writes_only_the_kept_classes(parquet_dir, tmp_path):
    kept = {0: wnid(0), 5: wnid(5)}
    out = tmp_path / "subset"
    counts = bs.Counter()
    for path in bs.split_files(parquet_dir, "train"):
        counts.update(bs.extract((path, str(out), kept, "train", 160)))
    written = {folder.name: len(list(folder.iterdir()))
               for folder in (out / "train").iterdir()}
    assert written == dict(counts) and set(written) == {wnid(0), wnid(5)}
    assert sum(written.values()) == 5
    image = Image.open(next((out / "train" / wnid(0)).iterdir()))
    assert min(image.size) == 160
    again = bs.extract((bs.split_files(parquet_dir, "train")[0], str(out), kept,
                        "train", 160))
    assert sum(again.values()) + 0 <= 5, "a second pass writes no new file"
    assert sum(len(list(f.iterdir())) for f in (out / "train").iterdir()) == 5


def test_build_from_the_command_line(parquet_dir, tmp_path, monkeypatch):
    for name, classes in (("nette", [0, 1]), ("woof", [2, 3])):
        for label in classes:
            (tmp_path / name / "train" / wnid(label)).mkdir(parents=True)
    monkeypatch.setattr(bs, "N_CLASSES", 6)
    listing = tmp_path / "classes.txt"
    out = tmp_path / "subset"
    monkeypatch.setattr("sys.argv", [
        "build_subset", "--parquet-dir", str(parquet_dir), "--imagenette-dir",
        str(tmp_path / "nette"), "--imagewoof-dir", str(tmp_path / "woof"), "--out",
        str(out), "--classes-file", str(listing), "--workers", "1"])
    bs.main()
    chosen = listing.read_text().split()
    assert len(chosen) == 6 and {wnid(0), wnid(1)} <= set(chosen)
    assert not {wnid(2), wnid(3)} & set(chosen)
    manifest = json.loads((out / "build_manifest.json").read_text())
    assert manifest["classes"] == chosen
    assert set(manifest["counts"]["val"].values()) == {2}
    assert sorted(p.name for p in (out / "val").iterdir()) == chosen
