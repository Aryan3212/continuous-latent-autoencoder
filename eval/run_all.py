from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any, Dict

from eval.run_probes import run_all_probes
from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="Training step for output naming; defaults to the step stored in the checkpoint.")
    ap.add_argument("--python_bin", default="python")
    ap.add_argument("--skip_probes", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = args.step
    if step is None:
        from eval.common import checkpoint_step
        step = checkpoint_step(args.ckpt) or 0
        print(f"[run_all] --step not given; using step {step} from checkpoint.", flush=True)

    results: Dict[str, Any] = {}

    recon_out = out_dir / "recon.json"
    try:
        subprocess.run(
            [
                args.python_bin,
                "-m",
                "eval.eval_recon",
                "--config",
                args.config,
                "--ckpt",
                args.ckpt,
                "--manifest",
                args.manifest,
                "--out",
                str(recon_out),
            ],
            check=True,
        )
        results["recon"] = json.loads(recon_out.read_text())
    except subprocess.CalledProcessError as e:
        print(f"[run_all] eval.eval_recon FAILED (exit {e.returncode}); continuing with probes.", flush=True)
        results["recon"] = {"error": f"eval_recon exited with code {e.returncode}"}

    if not args.skip_probes:
        cfg = load_config(args.config)
        cfg.resolved_config_path = args.config
        results["probes"] = run_all_probes(
            run_dir=str(out_dir),
            step=int(step),
            exp_cfg=cfg,
            ckpt_path=args.ckpt,
            python_bin=args.python_bin,
        )

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
