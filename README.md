# DALL·E 2 implementation

This project implements the main learning pipeline behind DALL·E 2 in a form
that can be trained and inspected on a single GPU. It connects three models:

```text
text prompt
    │
    ▼
CLIP text encoder ──► text embedding
                           │
                           ▼
                    diffusion prior
                           │
                           ▼
                 predicted image embedding
                           │
                           ▼
              conditional diffusion decoder
                           │
                           ▼
                       RGB image
```

This is an implementation inspired by DALL·E 2, not the original
production model. It generates low-resolution $64\times64$ natural images,
uses one pixel-space diffusion decoder, and has no super-resolution cascade or
classifier-free guidance. Its purpose is to make the complete CLIP → prior →
decoder workflow understandable and runnable.

## The three training stages

### 1. CLIP image-text representation

CLIP maps a paired image $I$ and caption $C$ into a shared semantic space:

$$
z_I=f_{\mathrm{image}}(I),
\qquad
z_T=f_{\mathrm{text}}(C).
$$

The custom CLIP implementation uses a Vision Transformer, a text Transformer,
and a symmetric contrastive loss over the image-text similarity matrix. There
are two CLIP modes:

- The default mode trains the local CLIP implementation from scratch and uses
  256-dimensional embeddings.
- `--using-pre-CLIP` loads and freezes a local
  `openai/clip-vit-base-patch32` checkpoint and uses its 512-dimensional
  embedding space. This is the recommended mode for stronger prompt semantics.

The pretrained model is expected by default at:

```text
~/scratch/llms_model/clip-vit-base-patch32
```

### 2. Diffusion prior

Text and image embeddings are related, but they are not interchangeable. A
caption can also describe many valid images. The prior therefore learns the
conditional distribution of CLIP image embeddings given text.

It adds noise to the ground-truth image embedding $z_I$:

$$
z_t=\sqrt{\bar\alpha_t}\,z_I
    +\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

and its Transformer predicts the clean image embedding:

$$
L_{\mathrm{prior}}
=\left\|\widehat z_I(z_t,t,C)-z_I\right\|_2^2.
$$

At inference time, the prior starts from Gaussian noise and performs a full
reverse-diffusion trajectory. `--prior-candidates N` samples $N$ possible image
embeddings and keeps the candidate with the highest cosine similarity to the
prompt's CLIP text embedding. Increasing it may improve prompt alignment, but
also increases inference time.

### 3. Conditional diffusion decoder

The decoder is a U-Net that turns an image embedding into pixels. A clean
training image $x_0$ is noised at a randomly chosen timestep:

$$
x_t=\sqrt{\bar\alpha_t}\,x_0
    +\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

The decoder predicts the added noise while conditioned on the timestep, text,
and CLIP image embedding:

$$
L_{\mathrm{decoder}}
=\mathbb E\left[
\left\|\epsilon-\epsilon_\theta(x_t,t,z_I,C)\right\|_2^2
\right].
$$

During decoder training, $z_I$ comes from the real image through the frozen
CLIP image encoder. During inference, it comes from the trained diffusion
prior and is held fixed throughout the pixel-denoising trajectory. The image
embedding conditions residual blocks; text and image tokens condition the
attention blocks.

The default U-Net uses channel widths `32 → 64 → 128 → 256`.
`--large-UNet` changes them to `64 → 128 → 256 → 512`, increases the
conditioning width, and is recommended when an A100 has sufficient memory.

## Datasets

| Option | Contents | Image format | Purpose |
|---|---|---:|---|
| no option / `--dataset flickr30k` | Flickr30k | RGB, 64×64 | Default natural-image dataset |
| `--dataset flickr8k` | Flickr8k | RGB, 64×64 | Smaller demonstration |
| `--dataset all` | Flickr8k + Flickr30k | RGB, 64×64 | Recommended natural-image combination |
| `--dataset fashion_mnist` | FashionMNIST only | grayscale, 32×32 | Fast pipeline demonstration |

`all` does **not** include FashionMNIST. The Flickr loaders read local Parquet
files, accept several common caption-column layouts, and select captions for
the image-text pairs. For the downloaded Flickr30k distribution whose shards
are all named `test-*.parquet`, the loader creates a deterministic 95%/5%
training-validation split.

Expected server layout:

```text
reinforcement_learning/
├── datasets/
│   ├── flickr8k/data/*.parquet
│   └── flickr30k/data/*.parquet
└── multip_modal/dalle2-fixed/
```

The compact custom text path uses UTF-8 byte tokens. In pretrained-CLIP mode,
the adapter reconstructs the text and retokenizes it with the official CLIP
tokenizer for CLIP encoding.

## Recommended A100 workflow

The following command trains on Flickr8k and Flickr30k, uses frozen pretrained
CLIP, and selects the larger decoder U-Net:

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/dalle2-fixed

TRAIN_JOB_ID=$(sbatch --parsable \
  submit-dalle2-train.sh \
  --dataset all \
  --using-pre-CLIP \
  --large-UNet \
  --decoder-lr 2e-4)

sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
  submit-dalle2-infer.sh \
  --run-name all-preclip-large-lr2e-4 \
  --prior-candidates 8 \
  --prompt "two children are playing outside" \
  --num-images 4 \
  --output generated_images/two-children.png
```

The `afterok` dependency starts inference only when all training stages finish
successfully. The training launcher runs the stages sequentially:

1. skip CLIP training and load the frozen pretrained CLIP;
2. train the diffusion prior;
3. train the pixel diffusion decoder.

To exercise the full from-scratch workflow, omit `--using-pre-CLIP`. Stage 1
will then train the custom CLIP before the prior and decoder.

## Configuration and experiment directories

`config.py` is the single source of truth for `CLIPConfig`, `PriorConfig`,
`DecoderConfig`, dataset presets, architecture choices, and command-line
overrides. Configuration is resolved in this order:

1. dataclass defaults;
2. dataset, pretrained-CLIP, and large-U-Net presets;
3. explicit command-line arguments;
4. validation and experiment-directory resolution.

Inspect all available options with:

```bash
python config.py --help
```

Unless `--run-name` is supplied, the experiment name records the dataset, CLIP
mode, U-Net size, and decoder learning rate. The example above produces:

```text
trained_models/all-preclip-large-lr2e-4/
├── effective_config.json
├── prior.pt
├── prior.pt.complete
├── decoder.pt
└── decoder.pt.complete
```

A custom-CLIP run also contains `clip.pt` and `clip.pt.complete`.
`effective_config.json` records the exact resolved architecture and training
settings. Inference with `--run-name` reloads this manifest, so dataset and
architecture flags should not be repeated. This prevents accidentally loading
a large-U-Net checkpoint into the base architecture or mixing incompatible
CLIP embedding spaces.

When changing parameters not encoded in the automatic name, such as epoch
counts or prior depth, provide a descriptive name explicitly:

```bash
sbatch submit-dalle2-train.sh \
  --dataset all \
  --using-pre-CLIP \
  --large-UNet \
  --prior-epochs 50 \
  --decoder-epochs 100 \
  --decoder-lr 1e-4 \
  --run-name all-preclip-large-prior50-decoder100-lr1e-4
```

## Checkpoint safety and resuming

- Normal execution refuses to overwrite an existing stage checkpoint.
- `--resume` skips stages that have both a checkpoint and its `.complete`
  marker, then starts at the first incomplete stage.
- `--overwrite` explicitly permits replacement of existing stage weights.

Resume is stage-level, not epoch-level. If a job stops during an epoch, a best
checkpoint may exist without a `.complete` marker; `--resume` restarts that
stage from epoch zero. Use the same model and dataset arguments, or the same
explicit `--run-name`, when resubmitting.

## Important files

| File | Role |
|---|---|
| `config.py` | Shared configuration, CLI, run manifests, and checkpoint safety |
| `dalle2_dataset.py` | FashionMNIST, Flickr8k, Flickr30k, and combined loaders |
| `model/clip.py` | Custom CLIP and frozen pretrained-CLIP adapter |
| `model/prior.py` | Transformer diffusion prior in CLIP embedding space |
| `model/decoder.py` | Conditional U-Net and reverse pixel diffusion |
| `train_clip.py` | Stage 1 training |
| `train_prior.py` | Stage 2 training |
| `train_decoder.py` | Stage 3 training |
| `infer.py` | Prompt-to-image generation and PNG-grid saving |
| `submit-dalle2-train.sh` | Sequential Slurm training job |
| `submit-dalle2-infer.sh` | Slurm inference job |
| `run_training_testing.txt` | Additional ready-to-run command examples |
| `understand_dalle2.md` | Deeper theory and implementation notes |

## Expectations and limitations

Flickr8k and Flickr30k are useful for learning and debugging this architecture,
but they are tiny compared with production text-to-image corpora. A pretrained
CLIP improves semantic conditioning; it does not give the decoder visual
knowledge that is absent from its training images. Consequently, outputs may
capture broad colours or layouts while missing fine objects, anatomy, text,
and unusual concepts.

For the most useful results from this project, use `--dataset all`, frozen
pretrained CLIP, the large U-Net, prompts resembling the Flickr caption domain,
and several prior candidates. Lower validation loss confirms the training
objective is improving, but visual sample quality remains the decisive test.
