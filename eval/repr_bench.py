"""Shared plumbing for the representation-quality benchmark.

Loads a fixed set of Bengali Common Voice utterances and extracts one
mean-pooled encoder embedding per utterance for each model under test:

    ours          our trained encoder (z, before any decoder)
    ours_random   same architecture, random init (lower-bound control)
    mimi          kyutai/mimi continuous encoder output (before quantization)
    wavlm         microsoft/wavlm-base-plus final hidden state
    mms           facebook/mms-300m final hidden state

Every embedding is mean-pooled over time so a single vector represents each
utterance. Embeddings are cached to ``runs/eval/embeddings/<model>.npz`` so the
UMAP and EER scripts can share one extraction pass.

Used by ``eval/eval_repr_umap.py`` and ``eval/eval_repr_eer.py``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EVAL_DIR = _REPO_ROOT / "runs" / "eval"
EMB_DIR = EVAL_DIR / "embeddings"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000  # rate the utterances are cached at; each model resamples as needed

# Local speaker-labelled source: OpenSLR-53 Bengali (already downloaded).
OPENSLR53_TSV = _REPO_ROOT / "datasets" / "OpenSLR53" / "asr_bengali" / "utt_spk_text.tsv"

# Default model under test (our checkpoint on the Hub).
OUR_HF_REPO = "aryan3212/my-model"

MODEL_ORDER = ["ours", "ours_random", "mimi", "wavlm", "mms"]


# --------------------------------------------------------------------------- #
# Data: Bengali Common Voice 17 test set
# --------------------------------------------------------------------------- #


@dataclass
class Utterance:
    id: str
    speaker: str
    wav: torch.Tensor  # 1-D float32 mono @ TARGET_SR
    emotion: Optional[str] = None
    gender: Optional[str] = None


def _resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    import torchaudio.functional as AF

    return AF.resample(wav, src_sr, dst_sr)


def load_openslr53_utterances(
    max_utts: int = 300,
    *,
    max_per_speaker: int = 6,
    min_per_speaker: int = 2,
    seed: int = 0,
) -> List[Utterance]:
    """Load up to ``max_utts`` clips from the local OpenSLR-53 Bengali set.

    Speaker ids come from ``utt_spk_text.tsv`` (cols: utt_id, spk_id, text).
    To make speaker-verification pairs meaningful we sample several utterances
    per speaker (``max_per_speaker``) and only use speakers with at least
    ``min_per_speaker`` clips, rather than scattering ``max_utts`` over hundreds
    of speakers (which would leave almost no same-speaker pairs).
    """
    import random

    import torchaudio

    if not OPENSLR53_TSV.exists():
        raise FileNotFoundError(
            f"OpenSLR-53 tsv not found at {OPENSLR53_TSV}. "
            "Download it first: housekeeping.py download --datasets openslr53"
        )
    data_root = OPENSLR53_TSV.parent / "data"

    # Group existing utterances by speaker.
    by_spk: Dict[str, List[tuple]] = {}
    with open(OPENSLR53_TSV, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            utt_id, spk_id = parts[0], parts[1]
            path = data_root / utt_id[:2] / f"{utt_id}.flac"
            if path.exists():
                by_spk.setdefault(spk_id, []).append((utt_id, path))

    rng = random.Random(seed)
    speakers = [s for s, rows in by_spk.items() if len(rows) >= min_per_speaker]
    rng.shuffle(speakers)

    print(
        f"[data] OpenSLR-53: sampling <= {max_per_speaker} utts/speaker "
        f"over {len(speakers)} eligible speakers (target {max_utts})",
        flush=True,
    )
    utts: List[Utterance] = []
    for spk in speakers:
        rows = by_spk[spk][:]
        rng.shuffle(rows)
        for utt_id, path in rows[:max_per_speaker]:
            wav, sr = torchaudio.load(str(path))
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = _resample(wav.squeeze(0), int(sr), TARGET_SR)
            utts.append(Utterance(id=utt_id, speaker=spk, wav=wav.contiguous()))
            if len(utts) >= max_utts:
                break
        if len(utts) >= max_utts:
            break

    n_spk = len({u.speaker for u in utts})
    print(f"[data] collected {len(utts)} utterances across {n_spk} speakers", flush=True)
    return utts


SUBESCO_DIR = _REPO_ROOT / "datasets" / "SUBESCO"
SUBESCO_EMOTIONS = ("ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE")


def load_subesco_utterances(max_utts: Optional[int] = None, seed: int = 0) -> List[Utterance]:
    """Load SUBESCO (Bangla emotional speech) clips with emotion + speaker labels.

    Filenames look like ``F_02_MONIKA_S_1_NEUTRAL_1.wav``: tokens are
    gender(M/F), speaker number, name, "S", sentence, EMOTION, take. We parse
    emotion as the token matching the known 7-emotion set and speaker as
    ``<gender>_<num>`` so the 20 speakers are distinct groups.
    """
    import random

    import torchaudio

    wavs = sorted(SUBESCO_DIR.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No .wav under {SUBESCO_DIR}. Download+unzip SUBESCO first."
        )

    emo_set = set(SUBESCO_EMOTIONS)
    parsed: List[tuple] = []
    for p in wavs:
        toks = p.stem.upper().split("_")
        emo = next((t for t in toks if t in emo_set), None)
        if emo is None or len(toks) < 2:
            continue
        gender = toks[0]
        if gender not in ("M", "F"):
            gender = None
        speaker = f"{toks[0]}_{toks[1]}"
        parsed.append((p, speaker, emo, gender))

    rng = random.Random(seed)
    rng.shuffle(parsed)
    if max_utts is not None:
        parsed = parsed[:max_utts]

    utts: List[Utterance] = []
    for p, speaker, emo, gender in parsed:
        wav, sr = torchaudio.load(str(p))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = _resample(wav.squeeze(0), int(sr), TARGET_SR)
        utts.append(Utterance(id=p.stem, speaker=speaker, wav=wav.contiguous(),
                              emotion=emo, gender=gender))

    n_spk = len({u.speaker for u in utts})
    from collections import Counter
    dist = Counter(u.emotion for u in utts)
    print(f"[data] SUBESCO: {len(utts)} utts, {n_spk} speakers, emotions={dict(dist)}",
          flush=True)
    return utts


def load_cv_utterances(max_utts: int = 300) -> List[Utterance]:
    """Stream Common Voice 17 (bn, test) and collect up to ``max_utts`` clips
    that carry a non-empty speaker id (``client_id``). Streaming avoids pulling
    the whole split; we stop as soon as we have enough.
    """
    from datasets import load_dataset

    print(f"[data] streaming common_voice_17_0 bn:test (target {max_utts} utts)", flush=True)
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "bn",
        split="test",
        streaming=True,
    )

    utts: List[Utterance] = []
    for row in ds:
        speaker = (row.get("client_id") or "").strip()
        if not speaker:
            continue
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        if arr.size == 0:
            continue
        wav = torch.from_numpy(arr)
        wav = _resample(wav, int(audio["sampling_rate"]), TARGET_SR)
        uid = Path(str(row.get("path") or f"utt{len(utts)}")).stem
        utts.append(Utterance(id=uid, speaker=speaker, wav=wav.contiguous()))
        if len(utts) >= max_utts:
            break

    n_spk = len({u.speaker for u in utts})
    print(f"[data] collected {len(utts)} utterances across {n_spk} speakers", flush=True)
    return utts


def load_utterances(source: str = "openslr53", max_utts: int = 300) -> List[Utterance]:
    """Dispatch to a speaker-labelled utterance source."""
    if source == "openslr53":
        return load_openslr53_utterances(max_utts=max_utts)
    if source == "cv":
        return load_cv_utterances(max_utts=max_utts)
    raise ValueError(f"unknown source {source!r}; choose 'openslr53' or 'cv'")


# --------------------------------------------------------------------------- #
# Embedders: each maps a 1-D 16k waveform -> a 1-D mean-pooled embedding
# --------------------------------------------------------------------------- #


@dataclass
class Embedder:
    name: str
    fn: Callable[[torch.Tensor], np.ndarray]
    # Filled lazily so importing this module is cheap.
    _ready: bool = field(default=False, repr=False)


def _our_encoder_embedder(name: str, *, random_init: bool, ckpt: Optional[str]) -> Embedder:
    """Build an embedder around our frontend+encoder.

    The architecture comes from the checkpoint's embedded ``cfg`` so no separate
    config file is needed. ``random_init=True`` keeps the freshly-initialised
    weights (lower-bound control) instead of loading the trained ones.
    """
    from config import load_config  # noqa: F401  (ensures repo import path works)
    from models.encoder import Encoder
    from models.frontend_conv import ConvFrontend
    from schema import Config

    ckpt_path = _resolve_our_ckpt(ckpt)
    print(f"[{name}] loading checkpoint {ckpt_path}", flush=True)
    state = torch.load(ckpt_path, map_location="cpu")
    cfg_data = dict(state["cfg"])
    cfg_data.pop("resolved_config_path", None)

    aug_data = dict(cfg_data.get("aug", {}))
    if "wave_aug" in aug_data:
        aug_data["waveform_aug_global"] = aug_data.pop("wave_aug")
    if "wave_chunk_mask" in aug_data:
        aug_data["waveform_aug_local_mask"] = aug_data.pop("wave_chunk_mask")
    for key in (
        "waveform_aug_local_mask",
        "frontend_frame_local_mask",
        "decoder_input_mask",
    ):
        mask = dict(aug_data.get(key, {}))
        if "target_ratio" in mask:
            mask["ratio"] = mask.pop("target_ratio")
        if "token_ratio" in mask:
            mask["ratio"] = mask.pop("token_ratio")
            mask["min_span_frames"] = mask.pop("token_min_span")
            mask["max_span_frames"] = mask.pop("token_max_span")
        if mask:
            aug_data[key] = mask
    cfg_data["aug"] = aug_data

    loss_data = dict(cfg_data.get("loss", {}))
    mel_data = dict(loss_data.get("mel", {}))
    mel_data.pop("sample_rate", None)
    loss_data["mel"] = mel_data
    cfg_data["loss"] = loss_data
    cfg = Config.model_validate(cfg_data)

    frontend = ConvFrontend(cfg.model.frontend)
    encoder = Encoder(frontend.out_channels, cfg.model.encoder)
    model = torch.nn.ModuleDict({"frontend": frontend, "encoder": encoder})

    if not random_init:
        filtered = {
            k: v for k, v in state["model"].items()
            if k.split(".", 1)[0] in {"frontend", "encoder"}
        }
        model.load_state_dict(filtered, strict=True)

    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    sr = int(cfg.data.sample_rate)
    seg_samples = int(round(cfg.data.segment_seconds * sr))

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        wav = _resample(wav16k, TARGET_SR, sr)
        # The encoder has global attention + BatchNorm and only ever saw
        # segment-length inputs, so encode in non-overlapping segment windows
        # and concatenate the frames before pooling (matches eval/common.py).
        n_chunks = max(1, int(np.ceil(wav.numel() / seg_samples)))
        pad = n_chunks * seg_samples - wav.numel()
        if pad > 0:
            wav = torch.nn.functional.pad(wav, (0, pad))
        chunks = wav.view(n_chunks, 1, seg_samples).to(DEVICE)  # (n,1,S)
        h0 = frontend(chunks)
        z = encoder(h0)              # (n, D, T')
        z = z.permute(1, 0, 2).reshape(z.size(1), -1)  # (D, n*T')
        return z.t().float().cpu().numpy()  # (n*T', D) frame features

    return Embedder(name=name, fn=embed)


def _resolve_our_ckpt(ckpt: Optional[str]) -> str:
    """Return a local path to our checkpoint, downloading from the Hub if needed."""
    if ckpt and Path(ckpt).is_file():
        return ckpt
    # Look for a local last.pt under runs/ before hitting the Hub.
    if not ckpt:
        local = sorted((_REPO_ROOT / "runs").rglob("last.pt"))
        if local:
            return str(local[-1])
    import os

    from huggingface_hub import hf_hub_download

    repo = ckpt if (ckpt and "/" in ckpt and not Path(ckpt).exists()) else OUR_HF_REPO
    print(f"[ours] no local ckpt; downloading last.pt from {repo}", flush=True)
    return hf_hub_download(
        repo_id=repo, filename="last.pt", token=os.environ.get("HF_TOKEN") or None
    )


def _mimi_embedder() -> Embedder:
    """Mimi continuous encoder output *before* quantization, mean-pooled."""
    from transformers import AutoFeatureExtractor, MimiModel

    repo = "kyutai/mimi"
    print(f"[mimi] loading {repo}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = MimiModel.from_pretrained(repo).eval().to(DEVICE)
    mimi_sr = int(fe.sampling_rate)

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        wav = _resample(wav16k, TARGET_SR, mimi_sr).numpy()
        inputs = fe(raw_audio=wav, sampling_rate=mimi_sr, return_tensors="pt")
        iv = inputs["input_values"].to(DEVICE)  # (1,1,S)
        # Reproduce MimiModel._encode_frame up to (but not including) the
        # quantizer to get the continuous latent fed into the VQ.
        emb = model.encoder(iv)                              # (1, C, T)
        enc_out = model.encoder_transformer(emb.transpose(1, 2))
        emb = enc_out[0].transpose(1, 2)                     # (1, C, T)
        emb = model.downsample(emb)                          # (1, C, T')
        return emb.squeeze(0).t().float().cpu().numpy()      # (T', C) frame features

    return Embedder(name="mimi", fn=embed)


def _hf_hidden_state_embedder(name: str, repo: str) -> Embedder:
    """Final-hidden-state mean-pool for a wav2vec2-style HF model (WavLM, MMS)."""
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"[{name}] loading {repo}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval().to(DEVICE)
    msr = int(getattr(fe, "sampling_rate", TARGET_SR))

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        wav = _resample(wav16k, TARGET_SR, msr).numpy()
        inputs = fe(wav, sampling_rate=msr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model(**inputs)
        hs = out.last_hidden_state  # (1, T, D)
        return hs.squeeze(0).float().cpu().numpy()  # (T, D) frame features

    return Embedder(name=name, fn=embed)


def build_embedder(name: str, *, ckpt: Optional[str] = None) -> Embedder:
    if name == "ours":
        return _our_encoder_embedder("ours", random_init=False, ckpt=ckpt)
    if name == "ours_random":
        return _our_encoder_embedder("ours_random", random_init=True, ckpt=ckpt)
    if name == "mimi":
        return _mimi_embedder()
    if name == "wavlm":
        return _hf_hidden_state_embedder("wavlm", "microsoft/wavlm-base-plus")
    if name == "mms":
        return _hf_hidden_state_embedder("mms", "facebook/mms-300m")
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_ORDER}")


# --------------------------------------------------------------------------- #
# Extraction + cache
# --------------------------------------------------------------------------- #


_UTMOS_MODEL = None


def _utmos_model():
    """Lazily load the UTMOSv2 ensemble once (pretrained weights download on
    first use)."""
    global _UTMOS_MODEL
    if _UTMOS_MODEL is None:
        import utmosv2

        print("[utmos] loading UTMOSv2 pretrained ensemble", flush=True)
        _UTMOS_MODEL = utmosv2.create_model(pretrained=True)
    return _UTMOS_MODEL


def compute_utmos_scores(
    utts: List[Utterance], *, use_cache: bool = True
) -> np.ndarray:
    """Predicted MOS (naturalness/quality) per utterance, shape (N,).

    Cached to ``runs/eval/embeddings/utmos_mos.npz`` keyed by utterance ids.
    Used to color the cluster plots.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMB_DIR / "utmos_mos.npz"
    ids = np.array([u.id for u in utts])
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == list(ids):
            print(f"[utmos] using cached MOS ({len(data['mos'])})", flush=True)
            return data["mos"]

    model = _utmos_model()
    mos: List[float] = []
    for i, u in enumerate(utts):
        # predict(data=..., sr=...) returns a scalar/array for one clip.
        out = model.predict(data=u.wav.numpy(), sr=TARGET_SR)
        mos.append(float(np.asarray(out).reshape(-1)[0]))
        if (i + 1) % 50 == 0:
            print(f"[utmos] {i + 1}/{len(utts)} scored", flush=True)
    mos_arr = np.asarray(mos, dtype=np.float32)
    print(f"[utmos] done: MOS range [{mos_arr.min():.2f}, {mos_arr.max():.2f}]", flush=True)
    np.savez(cache, mos=mos_arr, ids=ids)
    return mos_arr


def _pool(frames: np.ndarray, mode: str) -> np.ndarray:
    """Pool frame features ``(T, D)`` to a single utterance vector.

    ``mean``    -> (D,)        ``meanstd`` -> (2D,) concat of mean and std.
    Mean+std keeps the temporal variance, which carries strong paralinguistic
    signal (standard in x-vector/ECAPA speaker and emotion systems).
    """
    m = frames.mean(axis=0)
    if mode == "mean":
        return m
    if mode == "meanstd":
        return np.concatenate([m, frames.std(axis=0)])
    raise ValueError(f"unknown pool mode {mode!r}")


def extract(
    name: str,
    utts: List[Utterance],
    *,
    ckpt: Optional[str] = None,
    pool: str = "mean",
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Return ``{"X": (N,D), "speakers": (N,), "ids": (N,)}`` for one model.

    ``pool`` selects time pooling (``mean`` or ``meanstd``). Cached to
    ``runs/eval/embeddings/<name>[.<pool>].npz`` keyed by the utterance ids so a
    stale cache (different utterance set) is detected and recomputed.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMB_DIR / (f"{name}.npz" if pool == "mean" else f"{name}.{pool}.npz")
    ids = np.array([u.id for u in utts])

    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == list(ids):
            print(f"[{name}] using cached embeddings ({data['X'].shape}, pool={pool})", flush=True)
            return {"X": data["X"], "speakers": data["speakers"], "ids": data["ids"]}

    emb = build_embedder(name, ckpt=ckpt)
    vecs: List[np.ndarray] = []
    for i, u in enumerate(utts):
        vecs.append(_pool(emb.fn(u.wav), pool))
        if (i + 1) % 50 == 0:
            print(f"[{name}] {i + 1}/{len(utts)} embedded", flush=True)
    X = np.stack(vecs, axis=0).astype(np.float32)
    speakers = np.array([u.speaker for u in utts])
    print(f"[{name}] done: {X.shape} (pool={pool})", flush=True)

    np.savez(cache, X=X, speakers=speakers, ids=ids)
    return {"X": X, "speakers": speakers, "ids": ids}
