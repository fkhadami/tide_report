#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cartopy
from cartopy.io import shapereader


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "cartopy"
    data_dir.mkdir(parents=True, exist_ok=True)

    cartopy.config["data_dir"] = str(data_dir)
    cartopy.config["pre_existing_data_dir"] = str(data_dir)

    # Download all standard GSHHS scales and land/ocean boundary levels.
    scales = ["c", "l", "i", "h", "f"]
    levels = [1, 2, 3, 4, 5, 6]
    natural_earth_layers = [
        ("10m", "physical", "land"),
        ("10m", "physical", "coastline"),
    ]

    print(f"Using cartopy data dir: {data_dir}")
    for scale in scales:
        for level in levels:
            path = shapereader.gshhs(scale=scale, level=level)
            print(f"Downloaded/available: scale={scale}, level={level}, path={path}")

    for resolution, category, name in natural_earth_layers:
        path = shapereader.natural_earth(
            resolution=resolution,
            category=category,
            name=name,
        )
        print(
            "Downloaded/available: "
            f"resolution={resolution}, category={category}, name={name}, path={path}"
        )

    print("Offline coastline data is ready.")


if __name__ == "__main__":
    main()
