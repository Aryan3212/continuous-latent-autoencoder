"""Render the compact representation scorecard from completed JSON evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.repr_bench import EVAL_DIR, MODEL_ORDER, model_spec


def _results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else {}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--emotion", type=Path, default=EVAL_DIR / "emotion_probe.json")
    ap.add_argument("--speaker-id", type=Path, default=EVAL_DIR / "speaker_id_probe.json")
    ap.add_argument("--speaker-verif", type=Path, default=EVAL_DIR / "speaker_verif.json")
    ap.add_argument("--age", type=Path, default=EVAL_DIR / "age_probe.json")
    ap.add_argument("--asr", action="append", default=[], metavar="MODEL=JSON")
    ap.add_argument("--throughput", action="append", default=[], metavar="MODEL=AUDIO_SECONDS_PER_SECOND")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "compact_scorecard.md")
    args = ap.parse_args()

    emotion, sid, age = _results(args.emotion), _results(args.speaker_id), _results(args.age)
    verif = _results(args.speaker_verif)
    verif_meanstd = verif.get("meanstd", verif.get("mean", {})) if isinstance(verif, dict) else {}
    asr = {}
    for item in args.asr:
        name, raw_path = item.split("=", 1)
        asr[name] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    throughput = {}
    for item in args.throughput:
        name, value = item.split("=", 1)
        throughput[name] = float(value)

    lines = [
        "# Compact representation scorecard", "",
        "All values are test metrics; higher is better except EER, CER, and WER.", "",
        "| model | params | component | frame Hz | audio sec/s | emotion F1 | speaker ID | speaker EER | age F1 | age bal. acc. | ASR CER | ASR WER |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        spec = model_spec(name)
        e, s, v, a, r = emotion.get(name, {}), sid.get(name, {}), verif_meanstd.get(name, {}), age.get(name, {}), asr.get(name, {}).get("dev", {})
        lines.append(
            f"| {name} | {spec.reported_params or 'measure at load'} | {spec.component} | {spec.frame_rate_hz or 'utterance'} | {throughput.get(name, '—')} | "
            f"{_fmt(e.get('macro_f1'))} | {_fmt(s.get('test_acc'))} | {_fmt(v.get('eer'))} | "
            f"{_fmt(a.get('macro_f1'))} | {_fmt(a.get('balanced_accuracy'))} | {_fmt(r.get('cer'))} | {_fmt(r.get('wer'))} |"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
