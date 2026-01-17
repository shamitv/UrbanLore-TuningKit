"""Initial setup script for UrbanLore-TuningKit.

Creates required data directories and optionally initializes a .env file
from .env.example if one does not exist.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def create_directories(base_dir: Path) -> list[Path]:
    required_dirs = [
        base_dir / "corpus",
        base_dir / "dataset",
        base_dir / "finetune" / "models",
        base_dir / "eval" / "results",
    ]

    created = []
    for path in required_dirs:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def ensure_env_file(base_dir: Path) -> bool:
    env_path = base_dir / ".env"
    example_path = base_dir / ".env.example"

    if env_path.exists():
        return False

    if not example_path.exists():
        raise FileNotFoundError(".env.example not found. Cannot initialize .env")

    shutil.copyfile(example_path, env_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize project folders and .env configuration."
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Do not create or overwrite the .env file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    created_dirs = create_directories(base_dir)

    env_created = False
    if not args.skip_env:
        env_created = ensure_env_file(base_dir)

    if created_dirs:
        print("Created directories:")
        for path in created_dirs:
            print(f"- {path.relative_to(base_dir)}")
    else:
        print("All required directories already exist.")

    if args.skip_env:
        print("Skipped .env creation.")
    elif env_created:
        print("Initialized .env from .env.example. Please update it with your settings.")
    else:
        print(".env already exists.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
