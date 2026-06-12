from organize_data import split_dataset


def make_source(tmp_path, counts):
    source = tmp_path / "raw"
    for class_name, n in counts.items():
        class_dir = source / class_name
        class_dir.mkdir(parents=True)
        for i in range(n):
            (class_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
    return source


def test_split_is_80_20_per_class(tmp_path):
    source = make_source(tmp_path, {"amd": 10, "normal": 20})
    output = tmp_path / "processed"
    split_dataset(source, output, val_split=0.2, seed=42)

    for class_name, n in [("amd", 10), ("normal", 20)]:
        train = list((output / "train" / class_name).iterdir())
        val = list((output / "val" / class_name).iterdir())
        assert len(train) == int(n * 0.8)
        assert len(val) == n - int(n * 0.8)
        # No image leaks between splits
        assert not {f.name for f in train} & {f.name for f in val}


def test_split_is_deterministic(tmp_path):
    source = make_source(tmp_path, {"cataract": 15})
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    split_dataset(source, out_a, seed=42)
    split_dataset(source, out_b, seed=42)

    names_a = sorted(f.name for f in (out_a / "val" / "cataract").iterdir())
    names_b = sorted(f.name for f in (out_b / "val" / "cataract").iterdir())
    assert names_a == names_b
