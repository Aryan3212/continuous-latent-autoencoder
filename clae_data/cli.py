"""Single CLI entry point for the ``clae_data`` package.

Usage::

    python -m clae_data <subcommand> [args...]

Subcommands:
    download             Download raw archives for the given adapters.
    audit                Probe rows in staging manifests (debug; pack runs audit too).
    build                Pack records into a staging dir (audio + manifests).
    push                 Upload a staging dir to a HF dataset repo.
    fetch                Snapshot-download a packed HF dataset repo.
    publish-checkpoint   Upload a ``last.pt`` + model card to a HF model repo.
    pack-and-push        Convenience: build + push in one shot (prep instance).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator, List

from clae_data.adapters.base import DatasetAdapter
from clae_data.registry import REGISTRY, get_adapter
from clae_data.schema import Record


# --- creds shim ------------------------------------------------------------ #
#
# We lazy-import ``_creds`` inside ``main()`` so ``from clae_data.cli import main``
# doesn't blow up if the (gitignored) creds file is absent. Defaults below are
# the static fallbacks used for argparse help text only; the real values come
# from ``_creds`` at dispatch time.


_CREDS: dict[str, str] = {
    "HF_TOKEN": "",
    "KAGGLE_USERNAME": "",
    "KAGGLE_KEY": "",
    "CLAE_HF_REPO": "aryanrahman/clae-bengali",
    "CLAE_CKPT_REPO": "aryanrahman/clae-bengali-encoder",
    "CLAE_DATA_ROOT": "/data/clae",
}


def _load_creds() -> None:
    """Populate ``_CREDS`` from ``clae_data._creds`` if available. Idempotent."""
    try:
        from clae_data import _creds as c  # type: ignore[attr-defined]
    except Exception as e:
        print(
            f"[clae_data] warning: could not import clae_data._creds ({e}); "
            "using built-in defaults. Copy _creds.example.py to _creds.py."
        )
        return
    for key in list(_CREDS):
        val = getattr(c, key, None)
        if val:
            _CREDS[key] = val


# --- shared helpers -------------------------------------------------------- #


def _parse_datasets(s: str | None) -> List[str]:
    """Comma-separated list of adapter names; empty/None -> all registered."""
    if not s:
        return sorted(REGISTRY)
    out = [x.strip() for x in s.split(",") if x.strip()]
    for name in out:
        if name not in REGISTRY:
            raise SystemExit(
                f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}"
            )
    return out


def _ensure_kaggle_env() -> None:
    """Pipe Kaggle creds from ``_creds`` into the env (kaggle library reads env)."""
    if _CREDS["KAGGLE_USERNAME"]:
        os.environ["KAGGLE_USERNAME"] = _CREDS["KAGGLE_USERNAME"]
    if _CREDS["KAGGLE_KEY"]:
        os.environ["KAGGLE_KEY"] = _CREDS["KAGGLE_KEY"]


class _LimitedAdapter(DatasetAdapter):
    """Wrap an adapter to cap ``iter_records`` at ``limit`` rows (smoke testing)."""

    def __init__(self, inner: DatasetAdapter, limit: int) -> None:
        self._inner = inner
        self._limit = int(limit)
        self.name = inner.name
        self.language = inner.language
        self.requires_credentials = inner.requires_credentials

    def download(self, dest_root: Path) -> Path:
        return self._inner.download(dest_root)

    def iter_records(self, raw_dir: Path) -> Iterator[Record]:
        return itertools.islice(self._inner.iter_records(raw_dir), self._limit)


def _build_adapters(names: List[str], limit: int | None) -> List[DatasetAdapter]:
    adapters: List[DatasetAdapter] = [get_adapter(n) for n in names]
    if limit is not None and limit > 0:
        adapters = [_LimitedAdapter(a, limit) for a in adapters]
    return adapters


# --- download -------------------------------------------------------------- #


def _add_download(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated adapter names. Default: all registered.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root for raw archives. Default: CLAE_DATA_ROOT from _creds.",
    )
    p.set_defaults(func=_run_download)


def _run_download(args: argparse.Namespace) -> None:
    _ensure_kaggle_env()
    names = _parse_datasets(args.datasets)
    root = Path(args.data_root or _CREDS["CLAE_DATA_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        adapter = get_adapter(name)
        print(f"[clae_data] download: {name} -> {root}")
        adapter.download(root)


# --- audit ----------------------------------------------------------------- #


def _add_audit(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--staging-dir",
        required=True,
        help="Staging dir containing a manifests/ subdir of JSONL files.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.set_defaults(func=_run_audit)


def _run_audit(args: argparse.Namespace) -> None:
    """Standalone audit: probe every JSONL under ``<staging-dir>/manifests/``.

    Debug-only. ``pack.pack_to_dir`` runs ``audit_records`` internally before
    transcoding, so a normal ``build`` does not need this. Use this when you
    want to verify a packed manifest after the fact, e.g. on a different
    machine, without re-running the full pack.
    """
    from clae_data.audit import audit_records

    _ensure_kaggle_env()
    staging = Path(args.staging_dir)
    manifests_dir = staging / "manifests"
    if not manifests_dir.is_dir():
        raise SystemExit(f"[clae_data] no manifests/ under {staging}")

    for jp in sorted(manifests_dir.glob("*.jsonl")):
        print(f"[clae_data] audit: {jp}")
        with open(jp, "r", encoding="utf-8") as f:
            records: list[Record] = [json.loads(line) for line in f if line.strip()]
        # Manifests written by pack store paths relative to the staging dir
        # root. Absolutize so the audit worker's existence check is accurate
        # regardless of cwd.
        for r in records:
            ap = r.get("audio_filepath")
            if ap and not os.path.isabs(ap):
                r["audio_filepath"] = str(staging / ap)
        audit_records(
            records,
            num_workers=args.num_workers,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )


# --- build ----------------------------------------------------------------- #


def _add_build(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", default=None)
    p.add_argument(
        "--staging-dir",
        required=True,
        help="Output directory: receives audio/, manifests/, README.md, build_meta.yaml.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root used by adapters for raw archives. Default: CLAE_DATA_ROOT from _creds.",
    )
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--val-pct", type=float, default=0.05)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows per adapter (smoke testing). Default: no cap.",
    )
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_run_build)


def _run_build(args: argparse.Namespace) -> None:
    from clae_data.pack import pack_to_dir

    _ensure_kaggle_env()
    names = _parse_datasets(args.datasets)
    adapters = _build_adapters(names, args.limit)
    print(f"[clae_data] build: {names} -> {args.staging_dir}")
    pack_to_dir(
        adapters=adapters,
        download_root=Path(args.data_root or _CREDS["CLAE_DATA_ROOT"]),
        staging_dir=Path(args.staging_dir),
        target_sr=args.target_sr,
        val_pct=args.val_pct,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )


# --- push ------------------------------------------------------------------ #


def _add_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("--staging-dir", required=True)
    p.add_argument("--repo-id", default=None, help="Default: CLAE_HF_REPO from _creds.")
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_push)


def _run_push(args: argparse.Namespace) -> None:
    from clae_data.push import push_to_hub

    _ensure_kaggle_env()
    repo_id = args.repo_id or _CREDS["CLAE_HF_REPO"]
    staging = Path(args.staging_dir)
    # Quick row count for the progress message.
    n_files = sum(1 for _ in staging.rglob("*") if _.is_file())
    print(f"[clae_data] push: {n_files:,} files -> {repo_id}")
    push_to_hub(
        staging_dir=staging,
        repo_id=repo_id,
        token=_CREDS["HF_TOKEN"] or None,
        commit_message=args.commit_message,
        private=not args.public,
    )


# --- fetch ----------------------------------------------------------------- #


def _add_fetch(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", default=None, help="Default: CLAE_HF_REPO from _creds.")
    p.add_argument(
        "--dest",
        default=None,
        help="Local destination. Default: CLAE_DATA_ROOT from _creds.",
    )
    p.set_defaults(func=_run_fetch)


def _run_fetch(args: argparse.Namespace) -> None:
    from clae_data.fetch import fetch_dataset

    _ensure_kaggle_env()
    repo_id = args.repo_id or _CREDS["CLAE_HF_REPO"]
    dest = Path(args.dest or _CREDS["CLAE_DATA_ROOT"])
    print(f"[clae_data] fetch: {repo_id} -> {dest}")
    fetch_dataset(
        repo_id=repo_id,
        dest=dest,
        token=_CREDS["HF_TOKEN"] or None,
    )


# --- publish-checkpoint ---------------------------------------------------- #


def _add_publish_checkpoint(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True, help="Path to last.pt")
    p.add_argument(
        "--repo-id",
        default=None,
        help="Default: CLAE_CKPT_REPO from _creds.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_publish_checkpoint)


def _run_publish_checkpoint(args: argparse.Namespace) -> None:
    from clae_data.publish_checkpoint import publish_checkpoint

    _ensure_kaggle_env()
    repo_id = args.repo_id or _CREDS["CLAE_CKPT_REPO"]
    print(f"[clae_data] publish-checkpoint: {args.ckpt} -> {repo_id}")
    publish_checkpoint(
        ckpt_path=Path(args.ckpt),
        repo_id=repo_id,
        token=_CREDS["HF_TOKEN"] or None,
        commit_message=args.commit_message,
        private=not args.public,
    )


# --- pack-and-push (convenience) ------------------------------------------- #


def _add_pack_and_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("--datasets", default=None)
    p.add_argument("--repo-id", default=None, help="Default: CLAE_HF_REPO from _creds.")
    p.add_argument(
        "--data-root",
        default=None,
        help="Root used by adapters for raw archives. Default: CLAE_DATA_ROOT from _creds.",
    )
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--val-pct", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--staging-dir",
        default=None,
        help="If unset, uses a tmp dir. Set this to keep artifacts (implies --keep-staging).",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="When --staging-dir is unset, do not delete the tmp staging dir on exit.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default: private).",
    )
    p.add_argument("--commit-message", default=None)
    p.set_defaults(func=_run_pack_and_push)


def _run_pack_and_push(args: argparse.Namespace) -> None:
    from clae_data.pack import pack_to_dir
    from clae_data.push import push_to_hub

    _ensure_kaggle_env()
    names = _parse_datasets(args.datasets)
    adapters = _build_adapters(names, args.limit)
    repo_id = args.repo_id or _CREDS["CLAE_HF_REPO"]
    data_root = Path(args.data_root or _CREDS["CLAE_DATA_ROOT"])

    def _do(staging: Path) -> None:
        print(f"[clae_data] pack-and-push: build {names} -> {staging}")
        pack_to_dir(
            adapters=adapters,
            download_root=data_root,
            staging_dir=staging,
            target_sr=args.target_sr,
            val_pct=args.val_pct,
            seed=args.seed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        n_files = sum(1 for _ in staging.rglob("*") if _.is_file())
        print(f"[clae_data] pack-and-push: push {n_files:,} files -> {repo_id}")
        push_to_hub(
            staging_dir=staging,
            repo_id=repo_id,
            token=_CREDS["HF_TOKEN"] or None,
            commit_message=args.commit_message,
            private=not args.public,
        )

    if args.staging_dir:
        staging = Path(args.staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        _do(staging)
    elif args.keep_staging:
        staging = Path(tempfile.mkdtemp(prefix="clae_pack_"))
        print(f"[clae_data] pack-and-push: --keep-staging set, using {staging}")
        _do(staging)
    else:
        with tempfile.TemporaryDirectory(prefix="clae_pack_") as tmp:
            _do(Path(tmp))


# --- dispatch -------------------------------------------------------------- #


def main() -> None:
    _load_creds()

    ap = argparse.ArgumentParser(prog="clae_data")
    sub = ap.add_subparsers(dest="command", required=True)

    _add_download(sub.add_parser("download", help="Download raw archives."))
    _add_audit(
        sub.add_parser(
            "audit", help="Probe staging manifests (debug; pack runs audit internally)."
        )
    )
    _add_build(sub.add_parser("build", help="Pack records into a staging dir."))
    _add_push(sub.add_parser("push", help="Upload a staging dir to HF Hub."))
    _add_fetch(sub.add_parser("fetch", help="Snapshot-download a packed dataset repo."))
    _add_publish_checkpoint(
        sub.add_parser("publish-checkpoint", help="Upload a checkpoint to HF Hub.")
    )
    _add_pack_and_push(
        sub.add_parser("pack-and-push", help="Build + push in one shot.")
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
