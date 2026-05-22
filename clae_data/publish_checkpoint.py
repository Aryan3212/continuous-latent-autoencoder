"""Upload a training checkpoint plus an auto-generated model card to HF Hub."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _render_model_card(
    repo_id: str,
    step: object,
    cfg_yaml: str,
) -> str:
    """Compose a minimal HF model card README with YAML front matter."""
    wandb_project = os.environ.get("WANDB_PROJECT", "")
    wandb_line = (
        f"- W&B project: `{wandb_project}`\n" if wandb_project else ""
    )
    return (
        "---\n"
        "language:\n"
        "- bn\n"
        "license: other\n"
        "library_name: pytorch\n"
        "tags:\n"
        "- audio\n"
        "- self-supervised\n"
        "- bengali\n"
        "- conformer\n"
        "- lejepa\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        "Continuous latent autoencoder for Bengali speech.\n\n"
        "## Architecture\n\n"
        "- Encoder: Conformer\n"
        "- Self-supervised objective: LeJEPA\n"
        "- Reconstruction loss: multi-resolution STFT\n\n"
        "## Training\n\n"
        f"- Step: `{step}`\n"
        f"{wandb_line}"
        "\n"
        "## Config\n\n"
        "```yaml\n"
        f"{cfg_yaml}"
        "```\n\n"
        "## How to load\n\n"
        "```python\n"
        "import torch\n"
        "ckpt = torch.load('last.pt', map_location='cpu')\n"
        "state_dict = ckpt['model']\n"
        "cfg = ckpt['cfg']\n"
        "```\n"
    )


def publish_checkpoint(
    ckpt_path: Path,
    repo_id: str,
    token: str | None = None,
    extra_files: list[Path] | None = None,
    commit_message: str | None = None,
    private: bool = True,
) -> str:
    """Push ``last.pt`` + a generated model card + ``config.yaml`` to a HF model repo."""
    import torch
    import yaml
    from huggingface_hub import HfApi

    from clae_data._creds import HF_TOKEN
    from utils.checkpoint import try_git_hash

    ckpt_path = Path(ckpt_path)
    tok = token if token is not None else HF_TOKEN

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    step = ckpt.get("step", "?")
    cfg = ckpt.get("cfg", {})
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False)

    msg = commit_message or f"publish step={step} git={try_git_hash()}"
    print(f"[publish] uploading {ckpt_path} -> {repo_id} ({msg})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        readme_path = tmp_dir / "README.md"
        readme_path.write_text(
            _render_model_card(repo_id, step, cfg_yaml), encoding="utf-8"
        )
        config_path = tmp_dir / "config.yaml"
        config_path.write_text(cfg_yaml, encoding="utf-8")

        # Upload ckpt under its canonical name.
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo="last.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            repo_type="model",
            commit_message=msg,
        )
        for ef in extra_files or []:
            ef = Path(ef)
            api.upload_file(
                path_or_fileobj=str(ef),
                path_in_repo=ef.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=msg,
            )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[publish] done: {url}")
    return url
