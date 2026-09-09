from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "model": ["name_or_path"],
        "data": ["train_file", "validation_file", "test_file"],
        "lora": ["r", "alpha", "target_modules"],
        "training": ["output_dir", "num_train_epochs"],
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: {section}.{key}")
    return config
