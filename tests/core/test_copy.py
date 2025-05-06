from pathlib import Path

from lfmc.core.copy import copy_dir


def test_copy_dir(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (src_dir / f"file{i}.txt").write_text(f"file{i}.txt")
    (src_dir / "subdir").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (src_dir / "subdir" / f"file{i}.txt").write_text(f"subdir/file{i}.txt")

    dst_dir = tmp_path / "dst"
    dst_dir.mkdir(parents=True, exist_ok=False)
    copy_dir(src_dir, dst_dir)

    assert dst_dir.exists()
    for i in range(10):
        assert (dst_dir / f"file{i}.txt").read_text() == f"file{i}.txt"
    assert (dst_dir / "subdir").is_dir()
    for i in range(10):
        assert (dst_dir / "subdir" / f"file{i}.txt").read_text() == f"subdir/file{i}.txt"
