import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm


def _copy_file(paths: tuple[Path, Path]):
    src_file, dst_file = paths
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)


def copy_dir(src_dir: Path, dst_dir: Path, num_workers: int | None = None):
    def get_all_files(src_dir: Path, dst_dir: Path) -> list[tuple[Path, Path]]:
        files = []
        for src_file in src_dir.rglob("*"):
            if src_file.is_file():
                relative_path = src_file.relative_to(src_dir)
                dst_path = dst_dir / relative_path
                files.append((src_file, dst_path))
        return files

    if not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist or is not a directory.")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = get_all_files(src_dir, dst_dir)
    num_workers = num_workers or cpu_count()

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(files), desc="Copying files") as progress:
            for _ in pool.imap_unordered(_copy_file, files):
                progress.update(1)
