The continuous audio langauge model is not publicly released only the paper has been released so we have to re-implement.


4 METHOD
4.1 OUR VAE-GAN
Most autoregressive audio models are built upon RVQ-GAN architectures (Zeghidour et al., 2021;
Défossez et al., 2024a; Kumar et al., 2023; Guo et al., 2025). Following the approach of Evans
et al. (2024), we instead adopt a VAE-GAN framework, replacing the RVQ bottleneck with a VAE
bottleneck to regularize the latent space and enforce a Gaussian prior. Our VAE is fully causal and
draws from the architecture of Mimi (Défossez et al., 2024b), using Transformers in addition to
convolutions in the encoder and decoder, which have been shown to improve performance.
While training the model with adversarial losses and VAE regularization without any reconstruction
losses improves the quality of the model for speech, it degrades the reconstruction quality for music.
Semantic distillation is performed for the speech VAE similarly as in Mimi, with WavLM (Chen
5
Preprint
et al., 2021b) as teacher. There is no semantic distillation for the music model and we let this for
future work as semantic content is harder to define for music. The loss is:
LVAE = λtLt(x,ˆ
x) + λfLf(x,ˆ
x) + λadvLadv(ˆ x) + λfeatLfeat(x,ˆ
x) + λKLLKL + λdistillLdistill (2)
where Lt and Lf are the temporal and frequential reconstruction losses, Ladv is the adversarial loss,
Lfeat is the feature matching loss, LKL is the KL regularization applied to the VAE bottleneck, and
Ldistill is the WavLM distillation loss applied for the speech VAE

B OUR MUSIC VAE
Table 7: Music compression models. At least 96 VAE latent dimensions are required to outperform
the 32-RVQ codec on reconstruction metrics. EnCodec has been retrained on our dataset.
MODEL TYPE DIMS / RVQ FRAME RATE (HZ) BITRATE (KBPS) VISQOL (↑) SISNR (↑)
ENCODEC COPET ET AL. (2023) 4 RVQ 50 2.2 2.41 5.62
VQ-VAE (INSPIRED FROM MIMI) 32 RVQ 25 8.8 3.63 9.61
VAE 32 DIMS 25 – 2.23 5.51
VAE 96 DIMS 25 – 3.65 9.76
VAE 128 DIMS 25 – 4.01 10.3
Our variational autoencoder (VAE) and codec architecture is adapted from the Mimi codec (Défossez
et al., 2024b), originally designed for 24kHz speech at 12.5Hz. We trained it to compress 32kHz
mono music with a 25Hz frame rate. We experiment with bottleneck sizes of 96 and 128 dimensions.
For comparison, MusicGen’s EnCodec model (Copet et al., 2023) also operates at 32kHz but uses
a 4-level RVQ at 50Hz. In Tab. 7, we report reconstruction metrics (audio ViSQOL Chinen et al.
(2020) and SISNR), showing that a 32-dim VAE matches MusicGen’s codec, and that at least 9

Table 13: VAE hyperparameters
Music VAE Speech continuation VAE
General
Sample rate Frame rate Latent dimension 32kHz 25Hz 128 24kHz
12.5Hz
32
Architecture
Convolutions ratios Num transformer encoder layers Num transformer decoder layers Transformer context 8, 8, 5, 4 6, 5, 4, 4, 4
4 8
4 8
30s 10s
Training parameters
Batch size Audio sample length KL loss weight Reconstruction loss Distillation loss weight LR Schedule Learning rate 64 64
12s 12s
0.01 0.01
✓ ✗
✗ 25
cosine cosine
8·10−4 8·10−4


Pocket TTS Technical blog

Architecture
The core components of Pocket TTS, allowing for its small size but high performance, come from the research described in our recent paper, Continuous Audio Language Models.

As described in our codec tutorial, the standard way to model audio for text-to-speech applications is to use a neural audio codec to convert audio to discrete tokens, then autoregressively predict continuations of these token sequences with a transformer, and finally decode those back into audio.

In our previous text-to-speech release, we use a so-called RQ-transformer to get audio tokens from the backbone. But when attempting to shrink the model, this ends up becoming a computational bottleneck as it is very challenging to make the RQ-transformer smaller without sacrificing quality.

With Pocket TTS, we remove this bottleneck by entirely avoiding discrete tokens and instead having the model predict sequences of continuous latents directly. This may need like a simple change, but a number of tricks and optimizations is needed to make this work. We provide a technical overview here; for more details, please refer to the Continuous Audio Language Models paper. A v3 of the paper with more details on Pocket TTS is coming soon.

Diagram of the Pocket TTS architecture
Neural audio codec
Our codec is based on Mimi, the neural audio codec we designed for Moshi and later used in our delayed streams modeling paper. The primary difference is that Mimi compresses the audio into discrete tokens, whereas here we use continuous latents, regularized to follow a normal distribution as it is done in standard VAE training. Like in Mimi, to enforce semanticity of the representations, we distill WavLM into the inner latent representation of our codec with a cosine similarity loss. Mimi applies this loss only to the first RVQ level, but here, since there is no RVQ, we apply the distillation loss to the entire latent representation.

Generative Model
We train the model to predict continuations for 
(
x
1
,
…
,
x
S
−
1
)
(x 
1
 ,…,x 
S−1
 ), the sequence of continuous latent vectors produced by the codec’s encoder.

We build on the Masked Autoregressive (MAR) framework by employing a causal transformer backbone 
T
θ
T 
θ
​
  that outputs 
(
z
2
,
…
,
z
S
)
(z 
2
 ,…,z 
S
 ), and conditions an MLP sampler to build the next continuous latents 
(
x
2
,
…
,
x
S
)
(x 
2
 ,…,x 
S
 ). In MAR, the sampler is a diffusion model, but here we use a Lagrangian Self-Distillation (LSD) loss to natively enable 1-step sampling. At step 
i
i, the sampler outputs the prediction for the next latent 
x
i
+
1
x 
i+1
 , which, at inference time, is autoregressively fed back to the model.

Voice and text conditioning
To condition the model with the text to say and the voice to say it with, we prefix the generated audio with a few second voice prompt followed by the text to say. The audio is encoded using the neural audio codec encoder, and the text is embedded using a SentencePiece tokenizer.

Model size breakdown
In total, the generative model (causal transformer + MLP head) has 90M parameters and the codec’s decoder has 10M, adding up to 100M parameters in total. There is also the 18M-parameter encoder of the codec, which is only used once to encode a given voice sample. Afterwards, we can keep the embedding in memory to generate different audio from the same voice.

Data
The model is trained purely on publicly released data. Specifically, the dataset is composed of AMI, EARNINGS22, GIGASpeech, SPGISpeech, TED-LIUM, VoxPopuli, LibriHeavy, and Emilia. These datasets are all in English and add up to 88k hours of audio.

Scientific contributions
We employ several strategies to train this new model with continuous latent in an efficient manner:

Head batch multiplier
The training is bottlenecked by the transformer backbone that generates the conditioning variable 
z
s
z 
s
 . To address this, we introduce the Head Batch Multiplier, which amortizes this cost by reusing 
z
s
z 
s
  multiple times per training step. Specifically, for each input sequence, we compute 
z
s
z 
s
  once and use it across 
N
N loss computations, each with independently sampled LSD noise levels 
s
s, 
t
t and gaussian noise 
ϵ
ϵ. This not only improves efficiency but also stabilizes training by averaging the loss over multiple samples. We use 
N
=
8
N=8.

Gaussian Temperature Sampling
Sampling strategies, such as temperature sampling, can have a significant positive impact on generation quality in the discrete setup, particularly for speech. To replicate this behavior in the continuous domain, we introduce a dedicated sampling heuristic that results in comparable gains.

Specifically, we reduce the variance of the Gaussian noise passed to the LSD head. Applying a temperature of 
τ
τ is mathematically equivalent to multiplying the standard deviation by 
τ
τ
​
 . We found that a temperature of 0.7 brought good results in practice.

Latent Classifier-Free Guidance
Similarly, Classifier-Free Guidance (CFG) is known to improve the generation quality of conditioned generative models. It can be – and is generally – applied for diffusion and flow matching models on the sampling trajectory as well as on the logits of autoregressive language models.

However, CFG cannot be applied on the trajectory of 1-step flow models. Intuitively, this is because the output space cannot be used for interpolation/extrapolation. For instance, if we had a flow model that predicts a waveform directly, we would be interpolating between two waveforms, which just amounts to layering the sounds over each other. Here we predict latents for a neural audio codec, but the same intuition holds.

Instead, we apply the CFG on the outputs of the causal transformer backbone. Formally, given 
C
C a conditioning and 
α
α the CFG coefficient, we compute for every 
s
s of the sequence 
z
C
F
G
s
=
z
∅
s
+
α
(
z
C
s
−
z
∅
s
)
z 
CFG
s
​
 =z 
∅
s
​
 +α(z 
C
s
​
 −z 
∅
s
​
 ) , where 
z
C
s
z 
C
s
​
  is the output of the conditioned forward pass, and 
z
∅
s
z 
∅
s
​
  that of the unconditioned forward pass. We then generate 
x
s
x 
s
  with the LSD head conditioned on 
z
C
F
G
s
z 
CFG
s
​
 .

We call this method Latent CFG, as it operates on the latent variable 
z
s
z 
s
  instead of the model output. It is somewhat surprising that this works at all, because the modified latents could be completely out-of-distribution for the Flow Head, but we find that it significantly improves performance. In practice, we use 
α
=
1.5
α=1.5.

We discovered latent CFG independently, we also note that a similar idea also appears in the video-to-audio literature with SoundReactor.

Distillation
Once we have trained a model with a set CFG coefficient 
α
α, we can use this model as a teacher to distill it into a model that generates with this coefficient without doubling the batch size. The distilled model has a frozen copy of the MLP head of the teacher. The distillation objective for the student model is to output 
z
d
i
s
t
i
l
l
s
z 
distill
s
​
  out of its backbone that matches the 
z
C
F
G
s
z 
CFG
s
​
  of the guided teacher with an L2 loss. We observe that the student can remain accurate even with fewer layers than the teacher, which enabled us to have a teacher with 24 layers and a student with a mere 6.