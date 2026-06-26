# %% [markdown]
# CLAE Kaggle eval — 51-minute budget.
#
# Axis 1 (embeddings): yours, Mimi, WavLM-base, ECAPA-TDNN (~6M peer), mel+PCA (0-param floor).
# Probes (utterance-level, mean+std pool over time): gender, speaker, age.
# CKA: yours vs Mimi / WavLM. z_rank: effective rank per model (collapse check).
# UTMOSv2: perceptual quality on decoded clips (runs LAST — most cuttable).
#
# Everything is cached to disk as it is computed, and the cheap/safe items run
# first, so a mid-run timeout still leaves you with completed results.

# %% [markdown]
# ## Dependencies
# Kaggle base image already has torch, torchaudio, numpy, pandas, scikit-learn, librosa.
# Add these in a first cell (internet ON):
#
# ```bash
# pip install -q transformers>=4.44 speechbrain soundfile
# pip install -q jiwer                       # only if you later add the ASR oracle
# pip install -q git+https://github.com/sarulab-speech/UTMOSv2.git   # UTMOSv2
# ```
# Also attach: (a) the Common Voice bn dataset, (b) your repo as a Kaggle dataset
# OR clone it (set REPO_URL), (c) HF token in secrets if the ckpt repo is private.

# %% CONFIG --------------------------------------------------------------------
import os, sys, time, glob, pathlib, math, json

CV_DIR     = "/kaggle/input/datasets/sajidullah03/common-voice-24-bn"  # full CV-bn mount (auto-resolves tsv+clips)
REPO_ROOT  = "/kaggle/working/continuous-latent-autoencoder"    # your model code (cloned below)
REPO_URL   = "https://github.com/Aryan3212/continuous-latent-autoencoder.git"
REPO_BRANCH= "simplification"
CONFIG_PATH= "configs/kaggle_3m_gan.yaml"                       # config matching the ckpt
HF_REPO    = "aryan3212/clae-bengali-encoder"
HF_FILE    = "last.pt"

OUT_DIR    = "/kaggle/working/eval_out"
N_UTTS     = 600          # utterances sampled for the embedding axis
MAX_SECS   = 8.0          # truncate long clips to bound forward-pass time
SR         = 16000
MIN_CLIPS_PER_SPK = 4     # speakers kept for the speaker-ID probe
N_UTMOS    = 150          # clips decoded + scored by UTMOSv2
SEED       = 0

import numpy as np, torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED); torch.manual_seed(SEED)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"device={DEVICE}  out={OUT_DIR}")

def stamp(msg):  # wall-clock budget tracker
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# %% Repo + checkpoint ---------------------------------------------------------
# HF_TOKEN lets hf_hub_download reach the (private) ckpt repo — set via Kaggle secrets.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    stamp(f"(no kaggle secret HF_TOKEN: {e!r} — fine if the ckpt repo is public)")

if REPO_URL and not pathlib.Path(REPO_ROOT).exists():
    os.system(f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} {REPO_ROOT}")
else:
    os.system(f"git -C {REPO_ROOT} pull origin {REPO_BRANCH}")
assert pathlib.Path(REPO_ROOT, "config.py").exists(), f"model code not found at {REPO_ROOT}"
sys.path.insert(0, REPO_ROOT)
from huggingface_hub import hf_hub_download
from config import load_config
from reconstruct_audio import load_model, reconstruct   # ModuleDict{frontend,encoder,decoder}

CKPT = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
cfg = load_config(str(pathlib.Path(REPO_ROOT, CONFIG_PATH)))
assert cfg.data.sample_rate == SR, f"config sr {cfg.data.sample_rate} != {SR}"
clae = load_model(cfg, CKPT, DEVICE)
CHUNK = int(round(cfg.data.segment_seconds * SR))
stamp(f"loaded CLAE ckpt {HF_REPO}/{HF_FILE}, segment={cfg.data.segment_seconds}s")

# %% Sample utterances + labels ------------------------------------------------
import pandas as pd
# CommonVoice layout under the mount: <...>/test.tsv next to a clips/ dir. Resolve both.
tsvs = glob.glob(f"{CV_DIR}/**/validated.tsv", recursive=True)  # denser gender/age labels than test.tsv
assert tsvs, f"no validated.tsv under {CV_DIR} — check `ls /kaggle/input`"
CV_TSV = tsvs[0]
CV_CLIPS = str(pathlib.Path(CV_TSV).parent / "clips")
assert pathlib.Path(CV_CLIPS).is_dir(), f"no clips/ dir next to {CV_TSV}"
stamp(f"tsv={CV_TSV}  clips={CV_CLIPS}")

df = pd.read_csv(CV_TSV, sep="\t", low_memory=False)
df["abspath"] = df["path"].map(lambda p: str(pathlib.Path(CV_CLIPS, p)))
df = df[df["abspath"].map(os.path.exists)].reset_index(drop=True)
assert len(df) > 0, f"no clips found under {CV_CLIPS}"

df = df.sample(n=min(N_UTTS, len(df)), random_state=SEED).reset_index(drop=True)
stamp(f"{len(df)} utterances; gender labeled={df['gender'].notna().sum()}, "
      f"age labeled={df['age'].notna().sum()}, unique speakers={df['client_id'].nunique()}")

# %% Audio loader --------------------------------------------------------------
import librosa
def load_wav(path):
    w, _ = librosa.load(path, sr=SR, mono=True)
    if len(w) > int(MAX_SECS * SR):
        w = w[: int(MAX_SECS * SR)]
    return w.astype(np.float32)

stamp("decoding audio to memory ...")
WAVS = [load_wav(p) for p in df["abspath"]]
stamp(f"loaded {len(WAVS)} waveforms")

# %% Pooling + feature helpers -------------------------------------------------
def meanstd_pool_TD(x):           # x: (T, D) frames -> (2D,) mean||std
    return np.concatenate([x.mean(0), x.std(0)]).astype(np.float32)

def save_feats(name, X):
    np.save(f"{OUT_DIR}/feat_{name}.npy", X)
    stamp(f"  cached feat_{name}.npy  shape={X.shape}")

# %% Extractor: mel + PCA (0-param floor) -------------------------------------
def extract_mel(wavs):
    feats = []
    for w in wavs:
        m = librosa.feature.melspectrogram(y=w, sr=SR, n_mels=80, hop_length=320)
        feats.append(meanstd_pool_TD(np.log(m + 1e-6).T))     # (T,80) -> (160,)
    return np.stack(feats)

# %% Extractor: CLAE encoder ---------------------------------------------------
@torch.no_grad()
def extract_clae(wavs):
    # The encoder only ever saw 3s (cfg.data.segment_seconds) segments and uses
    # unmasked global attention, so a single pass over a longer clip is OOD.
    # Mirror reconstruct_live: split into independent CHUNK windows, encode each,
    # concat all frames, then pool — same in-distribution path as decoding.
    import torch.nn.functional as F
    feats = []
    for w in wavs:
        x = torch.from_numpy(w).view(1, 1, -1).to(DEVICE)
        S = x.size(-1)
        n = max(1, math.ceil(S / CHUNK))
        x = F.pad(x, (0, n * CHUNK - S)).view(n, 1, CHUNK)    # (n_windows, 1, CHUNK)
        z = clae["encoder"](clae["frontend"](x))              # (n, d, Tc)
        frames = z.permute(0, 2, 1).reshape(-1, z.size(1))    # (n*Tc, d)
        feats.append(meanstd_pool_TD(frames.cpu().numpy()))   # -> (2d,)
    return np.stack(feats)

# %% Extractor: WavLM-base (ceiling) ------------------------------------------
@torch.no_grad()
def extract_wavlm(wavs):
    from transformers import AutoModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    m = AutoModel.from_pretrained("microsoft/wavlm-base").to(DEVICE).eval()
    feats = []
    for w in wavs:
        iv = fe(w, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        h = m(iv).last_hidden_state[0]                         # (T, 768)
        feats.append(meanstd_pool_TD(h.cpu().numpy()))
    del m; torch.cuda.empty_cache()
    return np.stack(feats)

# %% Extractor: Mimi (ceiling, 12.5Hz match) ----------------------------------
@torch.no_grad()
def extract_mimi(wavs):
    from transformers import MimiModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    m = MimiModel.from_pretrained("kyutai/mimi").to(DEVICE).eval()
    feats = []
    for w in wavs:
        iv = fe(raw_audio=w, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        codes = m.encode(iv).audio_codes                      # (1, n_q, T)
        emb = m.quantizer.decode(codes)[0]                    # (dim, T) continuous
        feats.append(meanstd_pool_TD(emb.T.cpu().numpy()))
    del m; torch.cuda.empty_cache()
    return np.stack(feats)

# %% Extractor: ECAPA-TDNN (~6M supervised peer; already pooled) --------------
@torch.no_grad()
def extract_ecapa(wavs):
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(DEVICE)})
    feats = []
    for w in wavs:
        e = m.encode_batch(torch.from_numpy(w).unsqueeze(0).to(DEVICE))  # (1,1,192)
        feats.append(e.squeeze().cpu().numpy().astype(np.float32))
    return np.stack(feats)

# %% Run extractors (cheapest/safest first; ECAPA = the one risky load) --------
EXTRACTORS = [
    ("mel_pca", extract_mel),
    ("clae",    extract_clae),
    ("mimi",    extract_mimi),
    ("wavlm",   extract_wavlm),
    ("ecapa",   extract_ecapa),   # if this fights you, comment it out — mel_pca is the fallback peer
]
FEATS = {}
for name, fn in EXTRACTORS:
    t0 = time.time()
    try:
        X = fn(WAVS)
        if name == "mel_pca":
            from sklearn.decomposition import PCA
            X = PCA(n_components=min(128, X.shape[0], X.shape[1]),
                    random_state=SEED).fit_transform(X)
        FEATS[name] = X
        save_feats(name, X)
        stamp(f"{name}: {time.time()-t0:.0f}s")
    except Exception as e:                       # one model failing must not sink the run
        stamp(f"!! {name} FAILED: {e!r} — skipping")

# %% z_rank (effective rank / participation ratio) ----------------------------
def effective_rank(X):
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float((ev.sum() ** 2) / (ev ** 2).sum())          # participation ratio

zrank = {n: effective_rank(X) for n, X in FEATS.items()}
stamp(f"z_rank (eff. dim): {json.dumps({k: round(v,1) for k,v in zrank.items()})}")

# %% Probes: gender / age / speaker -------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def probe(X, y, mask, label):
    Xm, ym = X[mask], np.asarray(y)[mask]
    if len(np.unique(ym)) < 2 or len(ym) < 20:
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    return float(cross_val_score(clf, Xm, ym, cv=5,
                                 scoring="balanced_accuracy").mean())

gender_mask = df["gender"].notna().to_numpy()
age_mask    = df["age"].notna().to_numpy()
# speaker probe: keep speakers with enough clips, classify among them
spk_counts  = df["client_id"].value_counts()
keep_spk    = set(spk_counts[spk_counts >= MIN_CLIPS_PER_SPK].index)
spk_mask    = df["client_id"].isin(keep_spk).to_numpy()
stamp(f"probe coverage: gender={gender_mask.sum()} age={age_mask.sum()} "
      f"speaker={spk_mask.sum()} over {len(keep_spk)} speakers")

results = {}
for name, X in FEATS.items():
    results[name] = {
        "gender":  probe(X, df["gender"].values,    gender_mask, "gender"),
        "age":     probe(X, df["age"].values,       age_mask,    "age"),
        "speaker": probe(X, df["client_id"].values, spk_mask,    "speaker"),
        "z_rank":  round(zrank[name], 1),
    }

# %% CKA: yours vs Mimi / WavLM -----------------------------------------------
def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))

cka = {}
if "clae" in FEATS:
    for ref in ("mimi", "wavlm", "ecapa"):
        if ref in FEATS:
            cka[f"clae~{ref}"] = round(linear_cka(FEATS["clae"], FEATS[ref]), 3)
stamp(f"CKA(clae, *): {json.dumps(cka)}")

# %% Summary table -------------------------------------------------------------
summary = pd.DataFrame(results).T
print("\n=== PROBE balanced-accuracy + effective rank ===")
print(summary.to_string())
print("\n=== CKA ===")
print(json.dumps(cka, indent=2))
summary.to_csv(f"{OUT_DIR}/probe_summary.csv")
json.dump({"probes": results, "cka": cka}, open(f"{OUT_DIR}/results.json", "w"), indent=2)
stamp(f"saved probe_summary.csv + results.json to {OUT_DIR}")

# %% UTMOSv2 on decoded clips (LAST — most cuttable) --------------------------
# If the clock is short, you can stop here: everything above is already saved.
import soundfile as sf
@torch.no_grad()
def decode_clip(w):
    x = torch.from_numpy(w).view(1, 1, -1).to(DEVICE)
    x_hat, _ = reconstruct(clae, x, CHUNK)
    return x_hat[0, 0].cpu().numpy()

dec_dir = pathlib.Path(OUT_DIR, "decoded"); dec_dir.mkdir(exist_ok=True)
for i, w in enumerate(WAVS[:N_UTMOS]):
    sf.write(str(dec_dir / f"{i:04d}.wav"), decode_clip(w), SR)
stamp(f"decoded {min(N_UTMOS, len(WAVS))} clips -> {dec_dir}")

try:
    import utmosv2
    mos_model = utmosv2.create_model(pretrained=True)
    preds = mos_model.predict(input_dir=str(dec_dir))         # list of {file_path, predicted_mos}
    scores = [p["predicted_mos"] for p in preds]
    utmos_mean = float(np.mean(scores))
    json.dump({"utmosv2_mean": utmos_mean, "n": len(scores)},
              open(f"{OUT_DIR}/utmosv2.json", "w"), indent=2)
    stamp(f"UTMOSv2 mean = {utmos_mean:.3f} over {len(scores)} clips "
          f"(robotic-but-legible expectation ~2.0-2.8; track as trend)")
except Exception as e:
    stamp(f"!! UTMOSv2 skipped: {e!r}")

stamp("DONE")
