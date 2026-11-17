from __future__ import annotations

import json
import os
from pathlib import Path

from .models import CareRequest, Nurse

DEFAULT_DIR = Path(os.environ.get("NURSE_MATCH_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")


def save_dataset(out_dir: Path, nurses: list[Nurse], requests: list[CareRequest]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nurses.json").write_text(json.dumps([n.to_dict() for n in nurses], indent=1))
    (out_dir / "requests.json").write_text(json.dumps([r.to_dict() for r in requests], indent=1))


def load_dataset(data_dir: Path = DEFAULT_DIR) -> tuple[list[Nurse], list[CareRequest]]:
    data_dir = Path(data_dir)
    nurses_file = data_dir / "nurses.json"
    requests_file = data_dir / "requests.json"
    if not nurses_file.exists() or not requests_file.exists():
        raise FileNotFoundError(
            f"No data found in {data_dir}. Generate it with: python -m matcher.data_gen"
        )
    nurses = [Nurse.from_dict(d) for d in json.loads(nurses_file.read_text())]
    requests = [CareRequest.from_dict(d) for d in json.loads(requests_file.read_text())]
    return nurses, requests
