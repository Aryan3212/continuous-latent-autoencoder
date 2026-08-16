"""Shared adapters, data loaders, and versioned caches for representation evals."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Evaluation entry points do not go through train.py/housekeeping.py, so load
# the repository credentials here as well.  Shell variables still take
# precedence over .env values.
load_dotenv(_REPO_ROOT / ".env", override=False)


def _hf_token() -> Optional[str]:
    """Return the Hub token without ever exposing it in logs or metadata."""
    return os.environ.get("HF_TOKEN") or os.environ.get("hf_token") or None

EVAL_DIR = _REPO_ROOT / "runs" / "eval"
EMB_DIR = EVAL_DIR / "embeddings"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000  # rate the utterances are cached at; each model resamples as needed

# Local speaker-labelled source: OpenSLR-53 Bengali (already downloaded).
OPENSLR53_TSV = _REPO_ROOT / "datasets" / "OpenSLR53" / "asr_bengali" / "utt_spk_text.tsv"

# Default model under test (our checkpoint on the Hub).
OUR_HF_REPO = "aryan3212/my-model"

MODEL_ORDER = [
    "ours", "ours_random", "wavlm", "whisper_tiny", "ecapa", "emotion2vec",
    "mimi", "higgs_audio_v2", "xcodec2",
]
DEFAULT_MODELS = ["ours", "ours_random", "wavlm", "mimi"]
RANDOM_BASELINE_SEED = 0


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
    age: Optional[str] = None
    text: Optional[str] = None


def _resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    import torchaudio.functional as AF

    return AF.resample(wav, src_sr, dst_sr)


def _utterance_fingerprint(utts: List[Utterance]) -> str:
    """Hash ordered IDs and decoded samples so edited audio invalidates caches."""
    digest = hashlib.sha256()
    for utterance in utts:
        digest.update(utterance.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(utterance.wav.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


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


def load_subesco_utterances(
    max_utts: Optional[int] = None,
    seed: int = 0,
    root: str | Path | None = None,
) -> List[Utterance]:
    """Load SUBESCO (Bangla emotional speech) clips with emotion + speaker labels.

    Filenames look like ``F_02_MONIKA_S_1_NEUTRAL_1.wav``: tokens are
    gender(M/F), speaker number, name, "S", sentence, EMOTION, take. We parse
    emotion as the token matching the known 7-emotion set and speaker as
    ``<gender>_<num>`` so the 20 speakers are distinct groups.
    """
    import random

    import torchaudio

    subesco_dir = Path(root).expanduser().resolve() if root is not None else SUBESCO_DIR
    wavs = sorted(subesco_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No .wav under {subesco_dir}. Download+materialize SUBESCO first."
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


def load_common_voice_age_utterances(
    cv_root: str,
    max_utts: Optional[int] = None,
    seed: int = 0,
    label_column: str = "age",
) -> List[Utterance]:
    """Load age-labelled Bengali Common Voice clips from a local release.

    ``cv_root`` may be the release directory itself or any parent containing a
    ``validated.tsv`` and sibling ``clips/`` directory.  Age is a speaker-level
    field, so consumers must split by ``Utterance.speaker``.
    """
    import random

    import pandas as pd
    import torchaudio

    root = Path(cv_root)
    candidates = sorted(root.rglob("validated.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No validated.tsv found under {root}")
    tsv = candidates[0]
    clips = tsv.parent / "clips"
    if not clips.is_dir():
        raise FileNotFoundError(f"No clips/ directory next to {tsv}")
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    required = {"path", "client_id", label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{tsv} is missing required columns: {sorted(missing)}")
    rows = []
    columns = ["path", "client_id", label_column] + (["sentence"] if "sentence" in df.columns else [])
    for row in df[columns].fillna("").to_dict("records"):
        path = clips / str(row["path"])
        label = str(row[label_column]).strip()
        if str(row["client_id"]).strip() and label and path.is_file():
            rows.append((path, str(row["client_id"]).strip(), label, str(row.get("sentence", ""))))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_utts is not None:
        rows = rows[:max_utts]

    utts: List[Utterance] = []
    for path, speaker, age, text in rows:
        wav, sr = torchaudio.load(str(path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        utts.append(Utterance(
            id=path.stem, speaker=speaker,
            wav=_resample(wav.squeeze(0), int(sr), TARGET_SR).contiguous(),
            age=age, text=text,
        ))
    print(f"[data] Common Voice age: {len(utts)} clips, {len({u.speaker for u in utts})} speakers", flush=True)
    return utts


def load_utterances(
    source: str = "openslr53",
    max_utts: int = 300,
    *,
    subesco_dir: str | Path | None = None,
    seed: int = 0,
) -> List[Utterance]:
    """Dispatch to a speaker-labelled utterance source."""
    if source == "openslr53":
        return load_openslr53_utterances(max_utts=max_utts, seed=seed)
    if source == "cv":
        return load_cv_utterances(max_utts=max_utts)
    if source == "subesco":
        return load_subesco_utterances(
            max_utts=max_utts, root=subesco_dir, seed=seed
        )
    raise ValueError(f"unknown source {source!r}; choose 'openslr53', 'cv', or 'subesco'")


# --------------------------------------------------------------------------- #
# Embedders: each maps a 1-D 16k waveform -> a 1-D mean-pooled embedding
# --------------------------------------------------------------------------- #


@dataclass
class Embedder:
    name: str
    fn: Callable[[torch.Tensor], np.ndarray]
    spec: "ModelSpec"
    # Filled lazily so importing this module is cheap.
    _ready: bool = field(default=False, repr=False)


def _our_encoder_embedder(
    name: str,
    *,
    random_init: bool,
    ckpt: Optional[str],
    random_seed: int = RANDOM_BASELINE_SEED,
) -> Embedder:
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

    if random_init:
        # Keep the random control stable without modifying the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            frontend = ConvFrontend(cfg.model.frontend)
            encoder = Encoder(frontend.out_channels, cfg.model.encoder)
        print(f"[{name}] deterministic random baseline seed={random_seed}", flush=True)
    else:
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

    return Embedder(name=name, fn=embed, spec=model_spec(name))


def _resolve_our_ckpt(ckpt: Optional[str]) -> str:
    """Return a local path to our checkpoint, downloading from the Hub if needed."""
    if ckpt and Path(ckpt).is_file():
        return ckpt
    # Look for a local last.pt under runs/ before hitting the Hub.
    if not ckpt:
        local = sorted((_REPO_ROOT / "runs").rglob("last.pt"))
        if local:
            return str(local[-1])
    from huggingface_hub import hf_hub_download

    repo = ckpt if (ckpt and "/" in ckpt and not Path(ckpt).exists()) else OUR_HF_REPO
    print(f"[ours] no local ckpt; downloading last.pt from {repo}", flush=True)
    return hf_hub_download(
        repo_id=repo, filename="last.pt", token=_hf_token()
    )


def _mimi_embedder() -> Embedder:
    """Mimi continuous encoder output *before* quantization, mean-pooled."""
    from transformers import AutoFeatureExtractor, MimiModel

    spec = model_spec("mimi")
    print(f"[mimi] loading {spec.repo}@{spec.revision}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    )
    model = MimiModel.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    ).eval().to(DEVICE)
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

    return Embedder(name="mimi", fn=embed, spec=spec)


def _hf_hidden_state_embedder(name: str, repo: str) -> Embedder:
    """Final-hidden-state mean-pool for a wav2vec2-style HF model (WavLM, MMS)."""
    from transformers import AutoFeatureExtractor, AutoModel

    spec = model_spec(name)
    print(f"[{name}] loading {repo}@{spec.revision}", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(
        repo, revision=spec.revision, token=_hf_token()
    )
    model = AutoModel.from_pretrained(
        repo, revision=spec.revision, token=_hf_token()
    ).eval().to(DEVICE)
    msr = int(getattr(fe, "sampling_rate", TARGET_SR))

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        wav = _resample(wav16k, TARGET_SR, msr).numpy()
        inputs = fe(wav, sampling_rate=msr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model(**inputs)
        hs = out.last_hidden_state  # (1, T, D)
        return hs.squeeze(0).float().cpu().numpy()  # (T, D) frame features

    return Embedder(name=name, fn=embed, spec=model_spec(name))


@dataclass(frozen=True)
class ModelSpec:
    """Reproducibility metadata for a frozen feature extractor."""

    name: str
    repo: str
    revision: str = "main"
    feature_layer: str = "final_hidden_state"
    native_sample_rate: int = TARGET_SR
    frame_rate_hz: Optional[float] = None
    component: str = "encoder"
    supports_asr_probe: bool = True
    reported_params: Optional[str] = None


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "ours": ModelSpec("ours", "local-or-aryan3212/my-model", feature_layer="encoder.z", frame_rate_hz=12.5, reported_params="checkpoint"),
    "ours_random": ModelSpec("ours_random", "local-random-init", feature_layer="encoder.z", frame_rate_hz=12.5, reported_params="checkpoint"),
    "wavlm": ModelSpec("wavlm", "microsoft/wavlm-base-plus", revision="4c66d4806a428f2e922ccfa1a962776e232d487b", frame_rate_hz=50.0, reported_params="95M"),
    "mms": ModelSpec("mms", "facebook/mms-300m", revision="4ee317ce793c53dbc041fc4376c7558292dd38dc", frame_rate_hz=50.0),
    "whisper_tiny": ModelSpec("whisper_tiny", "openai/whisper-tiny", revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af", feature_layer="encoder.last_hidden_state", frame_rate_hz=50.0, reported_params="39M"),
    "ecapa": ModelSpec("ecapa", "speechbrain/spkrec-ecapa-voxceleb", revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286", feature_layer="utterance_embedding", frame_rate_hz=None, component="speaker_embedder", supports_asr_probe=False, reported_params="14.7M"),
    "emotion2vec": ModelSpec("emotion2vec", "emotion2vec/emotion2vec_base", revision="0c9a3152734f9d7a7a05b4ee6bfb3c109d288664", feature_layer="continuous_hidden_state"),
    "mimi": ModelSpec("mimi", "kyutai/mimi", revision="89091b3e466eb6a9d11e537bf26b144f194978f7", native_sample_rate=24000, frame_rate_hz=12.5, feature_layer="pre_quantization", component="codec_encoder"),
    "higgs_audio_v2": ModelSpec("higgs_audio_v2", "bosonai/higgs-audio-v2-tokenizer", revision="403fbacf2f60caaa102f893fdfabb694619b2417", native_sample_rate=24000, frame_rate_hz=25.0, feature_layer="quantizer_decoded_continuous", component="codec_encoder"),
    "xcodec2": ModelSpec("xcodec2", "HKUSTAudio/xcodec2-hf", revision="64bd034d12d441299cdd535b15c33efd6ccdf252", frame_rate_hz=50.0, feature_layer="quantized_continuous_latents", component="codec_encoder", reported_params="0.8B"),
}


def model_spec(name: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_ORDER}") from exc


def _output_frames(output: object, *, model_name: str) -> torch.Tensor:
    """Return continuous ``(T, D)`` features without ever using token ids."""
    for attr in ("last_hidden_state", "hidden_states", "continuous_latents", "pre_quantization", "embeddings", "latents"):
        value = getattr(output, attr, None)
        if isinstance(value, (tuple, list)):
            value = value[-1]
        if isinstance(value, torch.Tensor):
            break
    else:
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            value = output[0]
        else:
            raise RuntimeError(
                f"{model_name} did not expose continuous hidden features. "
                "Update its adapter; do not substitute discrete audio codes."
            )
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise RuntimeError(f"{model_name} feature tensor must be 3-D, got {tuple(value.shape)}")
    # Remote adapters must expose Hugging Face's time-major (B,T,D) convention.
    # Do not guess/transmute dimensions here: a short utterance can have T < D.
    return value.squeeze(0).float().cpu()


def _whisper_embedder() -> Embedder:
    from transformers import WhisperModel, WhisperProcessor

    spec = model_spec("whisper_tiny")
    processor = WhisperProcessor.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    )
    model = WhisperModel.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    ).eval().to(DEVICE)

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        inputs = processor(wav16k.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
        out = model.encoder(inputs.input_features.to(DEVICE))
        return _output_frames(out, model_name=spec.name).numpy()

    return Embedder(name=spec.name, fn=embed, spec=spec)


def _ecapa_embedder() -> Embedder:
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier

    spec = model_spec("ecapa")
    snapshot = snapshot_download(
        repo_id=spec.repo,
        revision=spec.revision,
        token=_hf_token(),
    )
    model = EncoderClassifier.from_hparams(
        source=snapshot,
        run_opts={"device": str(DEVICE)},
    )

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        out = model.encode_batch(wav16k.unsqueeze(0).to(DEVICE))
        return out.reshape(1, -1).float().cpu().numpy()

    return Embedder(name=spec.name, fn=embed, spec=spec)


def _emotion2vec_embedder() -> Embedder:
    """Extract emotion2vec's documented 50 Hz frame features via FunASR."""
    from funasr import AutoModel as FunASRAutoModel
    from huggingface_hub import snapshot_download

    spec = model_spec("emotion2vec")
    print(f"[{spec.name}] loading {spec.repo} with FunASR", flush=True)
    # FunASR delegates Hub downloads to huggingface_hub, which reads the
    # standard uppercase environment variable rather than a call-time token.
    token = _hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    snapshot = snapshot_download(
        repo_id=spec.repo,
        revision=spec.revision,
        token=token,
    )
    model = FunASRAutoModel(model=snapshot, device=str(DEVICE))

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        result = model.generate(
            input=wav16k.numpy(), fs=TARGET_SR, granularity="frame"
        )
        try:
            frames = result[0]["feats"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "emotion2vec did not return its documented `feats` output"
            ) from exc
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 3 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 2:
            raise RuntimeError(
                f"emotion2vec features must be (T, D), got {tuple(frames.shape)}"
            )
        return frames

    return Embedder(name=spec.name, fn=embed, spec=spec)


def _xcodec2_embedder() -> Embedder:
    """Use XCodec2's documented continuous ``latents`` output, not code ids."""
    from transformers import AutoFeatureExtractor, AutoModel

    spec = model_spec("xcodec2")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    )
    model = AutoModel.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    ).eval().to(DEVICE)

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        inputs = feature_extractor(audio=wav16k.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        encoded = model.encode(**inputs, output_latents=True)
        latents = encoded.latents  # documented (B, D, T)
        if latents is None:
            raise RuntimeError("XCodec2 did not return continuous latents")
        return latents.transpose(1, 2).squeeze(0).float().cpu().numpy()

    return Embedder(name=spec.name, fn=embed, spec=spec)


def _higgs_embedder() -> Embedder:
    """Decode Higgs codebooks back to continuous quantizer vectors for probes."""
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

    spec = model_spec("higgs_audio_v2")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    )
    model = HiggsAudioV2TokenizerModel.from_pretrained(
        spec.repo, revision=spec.revision, token=_hf_token()
    ).eval().to(DEVICE)

    @torch.no_grad()
    def embed(wav16k: torch.Tensor) -> np.ndarray:
        wav = _resample(wav16k, TARGET_SR, spec.native_sample_rate).numpy()
        inputs = feature_extractor(raw_audio=wav, sampling_rate=spec.native_sample_rate, return_tensors="pt")
        codes = model.encode(inputs["input_values"].to(DEVICE)).audio_codes
        # The public model exposes codes, while the quantizer exposes their
        # continuous reconstruction.  This remains a tokenizer representation,
        # not an integer-ID feature.
        for owner in (model, getattr(model, "acoustic_model", None), getattr(model, "semantic_model", None)):
            quantizer = getattr(owner, "quantizer", None)
            if quantizer is not None and hasattr(quantizer, "decode"):
                # model.encode() returns (B, Q, T), whereas the internal
                # residual quantizer decodes (Q, B, T) and returns (B, D, T).
                # Passing (B, Q, T) directly makes the quantizer axis look like
                # batch and causes pooling to retain variable utterance length.
                latents = quantizer.decode(codes.transpose(0, 1))
                if isinstance(latents, (tuple, list)):
                    latents = latents[0]
                if isinstance(latents, torch.Tensor) and latents.ndim == 3:
                    return latents.squeeze(0).transpose(0, 1).float().cpu().numpy()
        raise RuntimeError("Higgs adapter could not locate a continuous quantizer decode path in this pinned model revision")

    return Embedder(name=spec.name, fn=embed, spec=spec)


def build_embedder(
    name: str,
    *,
    ckpt: Optional[str] = None,
    random_seed: int = RANDOM_BASELINE_SEED,
) -> Embedder:
    if name == "ours":
        return _our_encoder_embedder("ours", random_init=False, ckpt=ckpt)
    if name == "ours_random":
        return _our_encoder_embedder(
            "ours_random",
            random_init=True,
            ckpt=ckpt,
            random_seed=random_seed,
        )
    if name == "mimi":
        return _mimi_embedder()
    if name == "wavlm":
        return _hf_hidden_state_embedder("wavlm", model_spec("wavlm").repo)
    if name == "mms":
        return _hf_hidden_state_embedder("mms", model_spec("mms").repo)
    if name == "whisper_tiny":
        return _whisper_embedder()
    if name == "ecapa":
        return _ecapa_embedder()
    if name == "xcodec2":
        return _xcodec2_embedder()
    if name == "higgs_audio_v2":
        return _higgs_embedder()
    if name == "emotion2vec":
        return _emotion2vec_embedder()
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_ORDER}")


# --------------------------------------------------------------------------- #
# Extraction + cache
# --------------------------------------------------------------------------- #


_UTMOS_MODEL = None
_UTMOS_IDENTITY: dict[str, Any] | None = None
UTMOS_PACKAGE_REVISION = "cc2700db57bb83ee13dc31ebe1b868c254e15d09"
UTMOS_CONFIG = "fusion_stage3"
UTMOS_FOLD = 0
UTMOS_SEED = 42


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash exact runtime weights so an unversioned upstream download is auditable."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"UTMOS state_dict entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _utmos_model():
    """Lazily load the UTMOSv2 model once (pretrained weights download on
    first use)."""
    global _UTMOS_IDENTITY, _UTMOS_MODEL
    if _UTMOS_MODEL is None:
        import utmosv2

        print("[utmos] loading UTMOSv2 pretrained model", flush=True)
        checkpoint_value = os.environ.get("UTMOSV2_CHECKPOINT")
        checkpoint_path = (
            Path(checkpoint_value).expanduser().resolve()
            if checkpoint_value
            else None
        )
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"UTMOSV2_CHECKPOINT does not exist: {checkpoint_path}"
            )
        _UTMOS_MODEL = utmosv2.create_model(
            pretrained=True,
            config=UTMOS_CONFIG,
            fold=UTMOS_FOLD,
            seed=UTMOS_SEED,
            checkpoint_path=checkpoint_path,
        )
        _UTMOS_MODEL.eval()
        _UTMOS_IDENTITY = {
            "package_revision": UTMOS_PACKAGE_REVISION,
            "config": UTMOS_CONFIG,
            "fold": UTMOS_FOLD,
            "seed": UTMOS_SEED,
            "checkpoint_source": (
                str(checkpoint_path)
                if checkpoint_path is not None
                else "UTMOSv2 package cache / sarulab-speech/UTMOSv2 resolve/main"
            ),
            "upstream_weight_revision": (
                "explicit local checkpoint"
                if checkpoint_path is not None
                else "not revision-pinned by upstream API; state_dict_sha256 is authoritative"
            ),
            "state_dict_sha256": _state_dict_sha256(_UTMOS_MODEL),
        }
        print(
            f"[utmos] runtime state_dict sha256={_UTMOS_IDENTITY['state_dict_sha256']}",
            flush=True,
        )
    return _UTMOS_MODEL


def utmos_runtime_identity() -> dict[str, Any]:
    _utmos_model()
    assert _UTMOS_IDENTITY is not None
    return dict(_UTMOS_IDENTITY)


def compute_utmos_scores(
    utts: List[Utterance], *, use_cache: bool = True
) -> np.ndarray:
    """Predicted MOS (naturalness/quality) per utterance, shape (N,).

    Cached to ``runs/eval/embeddings/utmos_mos.npz`` keyed by utterance ids.
    Used to color the cluster plots.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    model = _utmos_model()
    identity = utmos_runtime_identity()
    identity_json = json.dumps(identity, sort_keys=True)
    cache = EMB_DIR / "utmos_mos.npz"
    ids = np.array([u.id for u in utts])
    audio_fingerprint = _utterance_fingerprint(utts)
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_fingerprint = (
            str(data["audio_fingerprint"].item())
            if "audio_fingerprint" in data
            else ""
        )
        cached_identity = (
            str(data["model_identity"].item())
            if "model_identity" in data
            else ""
        )
        if (
            list(data["ids"]) == list(ids)
            and cached_fingerprint == audio_fingerprint
            and cached_identity == identity_json
        ):
            print(f"[utmos] using cached MOS ({len(data['mos'])})", flush=True)
            return data["mos"]

    mos: List[float] = []
    for i, u in enumerate(utts):
        # predict(data=..., sr=...) returns a scalar/array for one clip.
        out = model.predict(data=u.wav.numpy(), sr=TARGET_SR)
        mos.append(float(np.asarray(out).reshape(-1)[0]))
        if (i + 1) % 50 == 0:
            print(f"[utmos] {i + 1}/{len(utts)} scored", flush=True)
    mos_arr = np.asarray(mos, dtype=np.float32)
    print(f"[utmos] done: MOS range [{mos_arr.min():.2f}, {mos_arr.max():.2f}]", flush=True)
    np.savez(
        cache,
        mos=mos_arr,
        ids=ids,
        audio_fingerprint=audio_fingerprint,
        model_identity=identity_json,
    )
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


def extract_pools(
    name: str,
    utts: List[Utterance],
    *,
    pools: List[str],
    ckpt: Optional[str] = None,
    use_cache: bool = True,
    random_seed: int = RANDOM_BASELINE_SEED,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract one model once and derive every requested pooling mode.

    Each pooled result keeps its own cache file. If any modes are missing, the
    model runs once per utterance and fills all missing modes in the same pass.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    pools = list(dict.fromkeys(pools))
    if not pools:
        raise ValueError("at least one pooling mode is required")
    for pool in pools:
        if pool not in {"mean", "meanstd"}:
            raise ValueError(f"unknown pool mode {pool!r}")

    ids = np.array([u.id for u in utts])
    speakers = np.array([u.speaker for u in utts])
    audio_fingerprint = _utterance_fingerprint(utts)
    spec = model_spec(name)
    resolved_ckpt = ckpt
    if name in {"ours", "ours_random"}:
        resolved_ckpt = _resolve_our_ckpt(ckpt)

    ckpt_identity = ""
    if resolved_ckpt:
        checkpoint_path = Path(resolved_ckpt)
        ckpt_identity = (
            f"{checkpoint_path.resolve()}:{checkpoint_path.stat().st_size}:"
            f"{checkpoint_path.stat().st_mtime_ns}"
            if checkpoint_path.is_file()
            else resolved_ckpt
        )

    payloads: Dict[str, Dict[str, Any]] = {}
    caches: Dict[str, Path] = {}
    results: Dict[str, Dict[str, np.ndarray]] = {}
    missing: List[str] = []
    for pool in pools:
        payload: Dict[str, Any] = {
            "name": name,
            "repo": spec.repo,
            "revision": spec.revision,
            "feature_layer": spec.feature_layer,
            "pool": pool,
            "checkpoint": ckpt_identity,
            "ids": ids.tolist(),
            "audio_fingerprint": audio_fingerprint,
            "target_sr": TARGET_SR,
        }
        if name == "ours_random":
            payload["random_seed"] = random_seed
        cache_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache = EMB_DIR / f"{name}.{pool}.{cache_hash}.npz"
        payloads[pool] = payload
        caches[pool] = cache

        if use_cache and cache.exists():
            data = np.load(cache, allow_pickle=True)
            if list(data["ids"]) == list(ids):
                print(
                    f"[{name}] using cached embeddings "
                    f"({data['X'].shape}, pool={pool})",
                    flush=True,
                )
                results[pool] = {
                    "X": data["X"],
                    "speakers": speakers,
                    "ids": data["ids"],
                    "spec": spec,
                }
                continue
        missing.append(pool)

    if not missing:
        return results

    emb = build_embedder(
        name,
        ckpt=resolved_ckpt,
        random_seed=random_seed,
    )
    vecs: Dict[str, List[np.ndarray]] = {pool: [] for pool in missing}
    for i, u in enumerate(utts):
        frames = emb.fn(u.wav)
        for pool in missing:
            vec = _pool(frames, pool)
            previous = vecs[pool]
            if previous and vec.shape != previous[0].shape:
                raise ValueError(
                    f"{name} produced inconsistent pooled feature shapes: "
                    f"first={previous[0].shape}, utterance {u.id}={vec.shape}. "
                    "Check the adapter's time/feature axis handling."
                )
            previous.append(vec)
        if (i + 1) % 50 == 0:
            print(f"[{name}] {i + 1}/{len(utts)} embedded", flush=True)

    for pool in missing:
        X = np.stack(vecs[pool], axis=0).astype(np.float32)
        print(f"[{name}] done: {X.shape} (pool={pool})", flush=True)
        np.savez(
            caches[pool],
            X=X,
            speakers=speakers,
            ids=ids,
            metadata=json.dumps(payloads[pool], sort_keys=True),
        )
        results[pool] = {
            "X": X,
            "speakers": speakers,
            "ids": ids,
            "spec": spec,
        }
    return results


def extract(
    name: str,
    utts: List[Utterance],
    *,
    ckpt: Optional[str] = None,
    pool: str = "mean",
    use_cache: bool = True,
    random_seed: int = RANDOM_BASELINE_SEED,
) -> Dict[str, np.ndarray]:
    """Return pooled embeddings for one model and one pooling mode."""
    return extract_pools(
        name,
        utts,
        pools=[pool],
        ckpt=ckpt,
        use_cache=use_cache,
        random_seed=random_seed,
    )[pool]
