from pathlib import Path
import shutil


def clear_dir(dir_path: Path, *, must_contain: str | None = None):
    dir_path = dir_path.resolve()

    if must_contain is not None:
        if must_contain not in dir_path.parts:
            raise RuntimeError(
                f"Refuse to clear {dir_path}, "
                f"missing safety token '{must_contain}'"
            )

    if dir_path.exists():
        for p in dir_path.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        dir_path.mkdir(parents=True)