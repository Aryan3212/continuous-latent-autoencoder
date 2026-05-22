"""Upload a packed staging directory to a HF dataset repo."""
from __future__ import annotations

from pathlib import Path


_GITATTRIBUTES = (
    "*.flac filter=lfs diff=lfs merge=lfs -text\n"
    "*.wav filter=lfs diff=lfs merge=lfs -text\n"
    "*.mp3 filter=lfs diff=lfs merge=lfs -text\n"
)


def push_to_hub(
    staging_dir: Path,
    repo_id: str,
    token: str | None = None,
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    """Create-or-update a HF dataset repo from ``staging_dir``. Returns the repo URL."""
    from huggingface_hub import HfApi

    from clae_data._creds import HF_TOKEN
    from utils.checkpoint import try_git_hash

    staging_dir = Path(staging_dir)
    tok = token if token is not None else HF_TOKEN

    # Make LFS tracking explicit for the binary audio extensions.
    gitattributes = staging_dir / ".gitattributes"
    if not gitattributes.exists():
        gitattributes.write_text(_GITATTRIBUTES, encoding="utf-8")

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )

    msg = commit_message or f"pack {try_git_hash()}"
    print(f"[push] uploading {staging_dir} -> {repo_id} ({msg})")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=msg,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[push] done: {url}")
    return url
