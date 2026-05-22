"""Pull a packed HF dataset repo onto local disk for training-side consumption."""
from __future__ import annotations

from pathlib import Path


def fetch_dataset(
    repo_id: str,
    dest: Path,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Snapshot-download a packed dataset repo. Returns the local repo root."""
    from huggingface_hub import snapshot_download

    from clae_data._creds import HF_TOKEN

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tok = token if token is not None else HF_TOKEN

    print(f"[fetch] downloading {repo_id} -> {dest}")
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=tok,
        allow_patterns=allow_patterns,
    )
    local_root = Path(local_root)

    manifests_dir = local_root / "manifests"
    if manifests_dir.exists():
        print("[fetch] resolved manifest paths:")
        for jp in sorted(manifests_dir.glob("*.jsonl")):
            print(f"  {jp}")
    else:
        print(f"[fetch] note: no manifests/ subdir under {local_root}")
    return local_root
