# Resemble Enhance

[![PyPI](https://img.shields.io/pypi/v/resemble-enhance.svg)](https://pypi.org/project/resemble-enhance/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-Space-yellow)](https://huggingface.co/spaces/ResembleAI/resemble-enhance)
[![License](https://img.shields.io/github/license/resemble-ai/Resemble-Enhance.svg)](https://github.com/resemble-ai/resemble-enhance/blob/main/LICENSE)
[![Webpage](https://img.shields.io/badge/Webpage-Online-brightgreen)](https://www.resemble.ai/enhance/)

https://github.com/resemble-ai/resemble-enhance/assets/660224/bc3ec943-e795-4646-b119-cce327c810f1

Resemble Enhance is an AI-powered tool that aims to improve the overall quality of speech by performing denoising and enhancement. It consists of two modules: a denoiser, which separates speech from a noisy audio, and an enhancer, which further boosts the perceptual audio quality by restoring audio distortions and extending the audio bandwidth. The two models are trained on high-quality 44.1kHz speech data that guarantees the enhancement of your speech with high quality.

## Usage

### Installation

Install the stable version:

```bash
pip install resemble-enhance --upgrade
```

Or try the latest pre-release version:

```bash
pip install resemble-enhance --upgrade --pre
```

### Enhance

```
resemble-enhance in_dir out_dir
```

### Inference Without DeepSpeed (Windows-friendly)

- No additional flags are required to skip DeepSpeed for inference. The CLI below works without installing DeepSpeed.
- The model weights download automatically on first run.

Common flags:

- `--device {cuda|cpu}`: choose GPU or CPU. Example: `--device cuda`
- `--nfe <int>`: number of function evaluations (speed/quality tradeoff). Lower is faster. Example: `--nfe 16`
- `--solver {midpoint|rk4|euler}`: ODE solver for the enhancer. Example: `--solver midpoint`
- `--tau <0..1>`: CFM prior temperature. Example: `--tau 0.5`
- `--lambd <0..1>`: denoise strength blending during enhancement. Example: `--lambd 1.0`
- `--denoise_only`: only run the denoiser (no enhancement)
- `--run_dir <path>`: use a specific run directory with checkpoints and `hparams.yaml`

Examples:

```
# GPU inference (recommended if available)
resemble-enhance in_dir out_dir --device cuda --nfe 90 --solver midpoint --tau 0.5

# Faster test run (lower quality but quicker)
resemble-enhance in_dir out_dir --device cuda --nfe 16 --solver midpoint

# CPU fallback
resemble-enhance in_dir out_dir --device cpu --nfe 16 --solver euler

# Denoise only
resemble-enhance in_dir out_dir --denoise_only --device cuda
```
python.exe -m resemble_enhance.enhancer input_audio output_audio --denoise_only --device cuda


### Denoise only

```
resemble-enhance in_dir out_dir --denoise_only
```

### Web Demo

We provide a web demo built with Gradio, you can try it out [here](https://huggingface.co/spaces/ResembleAI/resemble-enhance), or also run it locally:

```
python app.py
```

Notes

- Inference does not require DeepSpeed. Training does. If you are on Windows, use the commands above without installing DeepSpeed.
- The first run downloads the default model to `resemble_enhance/model_repo/enhancer_stage2`.
- For custom checkpoints, pass `--run_dir <your_run_dir>` containing `hparams.yaml` and `ds/G/default/mp_rank_00_model_states.pt`.

## Train your own model

### Data Preparation

You need to prepare a foreground speech dataset and a background non-speech dataset. In addition, you need to prepare a RIR dataset ([examples](https://github.com/RoyJames/room-impulse-responses)).

```bash
data
├── fg
│   ├── 00001.wav
│   └── ...
├── bg
│   ├── 00001.wav
│   └── ...
└── rir
    ├── 00001.npy
    └── ...
```

### Training

#### Denoiser Warmup

Though the denoiser is trained jointly with the enhancer, it is recommended for a warmup training first.

```bash
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser
```

#### Enhancer

Then, you can train the enhancer in two stages. The first stage is to train the autoencoder and vocoder. And the second stage is to train the latent conditional flow matching (CFM) model.

##### Stage 1

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
```

##### Stage 2

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```

## Blog

Learn more on our [website](https://www.resemble.ai/enhance/)!
