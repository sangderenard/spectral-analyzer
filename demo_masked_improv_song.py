"""Write a small mask-aware torch-composer song sketch as JSON.

The generated score has fixed author notes and note-stream-owned pitch slots.
The write mask travels alongside the score so downstream modules can see which
fields they are allowed to edit.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch_composer_engine import (
    PARAM_HZ,
    build_masked_improvised_song,
    composer_to_jsonable,
)


def main() -> None:
    score, write_mask, stream, sparse = build_masked_improvised_song()
    out = {
        "score": composer_to_jsonable(score),
        "write_mask": composer_to_jsonable(write_mask),
        "note_stream": composer_to_jsonable(stream),
        "sparse_score": composer_to_jsonable(sparse),
    }
    path = Path("masked_improv_song.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    fixed = score.metadata["fixed_slots"]
    free = score.metadata["module_slots"]
    hz = score.params[0, :, PARAM_HZ].tolist()
    print(f"wrote {path.resolve()}")
    print(f"fixed slots: {fixed}")
    print(f"module pitch slots: {free}")
    print(f"hz: {[round(x, 2) for x in hz]}")


if __name__ == "__main__":
    main()

