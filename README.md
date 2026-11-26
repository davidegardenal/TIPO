<div align="center">

<h1>Taking Apart and Putting Together:<br>Tracking Objects Through Physical Transformations</h1>

[Davide Gardenal](https://github.com/davidegardenal)<sup>1</sup>, [Mingxuan Liu](https://oatmealliu.github.io)<sup>1</sup>, [Flavio Vella](https://hicrest.unitn.it/who-we-are/people/vella/)<sup>1</sup>, [Quentin Guimard](https://mardgui.github.io)<sup>2</sup>, [Elisa Ricci](https://eliricci.eu/)<sup>1,3</sup>

<sup>1</sup>University of Trento &nbsp;&nbsp; <sup>2</sup>AMIAD, Pôle Recherche, Palaiseau &nbsp;&nbsp; <sup>3</sup>Fondazione Bruno Kessler

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge)](https://davidegardenal.github.io/tipo-tracking/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.27398-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2611.27398)
[![VISTA](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VISTA-ffcc4d?style=for-the-badge)](https://huggingface.co/datasets/davidegarde/VISTA)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=for-the-badge)](LICENSE)

<img src="https://davidegardenal.github.io/tipo-tracking/assets/method.svg" alt="TIPO pipeline" width="90%">

</div>

**TIPO** (*Tracking Instances via Proposals and Overlaps*) is a training-free framework that tracks objects while they are **assembled** and **disassembled**. On top of SAM 2 / SAM 3 it follows a *generate → evaluate → judge* loop: it proposes splits and merges from disjoint mask components, scores them by mask overlap, and asks an MLLM to verify and name the best one. The output is a mask track for every object and a **state graph** of the transformations.

---

## Contents

- [Installation](#installation)
- [Data](#data)
- [Running TIPO](#running-tipo)
- [Evaluation](#evaluation)
- [Results](#results)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

**1. Clone** the repository with its submodules (SAM 2 and SAM 3 live in `deps/`):

```bash
git clone --recurse-submodules https://github.com/davidegardenal/TIPO.git
cd TIPO
```

**2. Build the environment** with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

> [!NOTE]
> The environment uses **CUDA 12.6** by default. To use another version, change the PyTorch index (`pytorch-cu126`) in `pyproject.toml`. The SAM 2 CUDA extension is built for `TORCH_CUDA_ARCH_LIST = "8.0"` (A100); set it to your GPU's architecture in the same file.

**3. Download the checkpoints**

*SAM 2.1 Hiera-L*: put it in `deps/sam2/checkpoints/`:

```bash
wget -P deps/sam2/checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

*SAM 3*: the model is gated, so first [request access](https://huggingface.co/facebook/sam3) on Hugging Face. Then log in and download it to the Hugging Face cache, where TIPO looks for it:

```bash
uv run hf auth login
uv run hf download facebook/sam3
```

> [!TIP]
> If you skip the download, SAM 3 is fetched on the first run.

## Data

Set `data_root` in [`configs/datasets.yaml`](configs/datasets.yaml) to the folder that holds the datasets. In the same config you can add your DAVIS-like dataset.

```
<data_root>/
├── VISTA/                 # https://huggingface.co/datasets/davidegarde/VISTA
│   ├── JPEGImages/
│   ├── Annotations/
│   ├── ImageSets/
│   ├── Transformations/
│   └── metadata.json
├── VOST/
└── VOST-TAS/vost_tas.json
```

## Running TIPO

```bash
uv run main.py -c configs/SAM3_qwen35bfp8.yaml \
    -m configs/mllm_labeler/qwen35bfp8.yaml -d vista
```

| Flag | Meaning |
| :-- | :-- |
| `-c` | Run config: tracker, thresholds, output folder |
| `-m` | Semantic MLLM config, to name objects and transformations |
| `--no-semantic` | Skip naming, even if `-m` is given |
| `-d` / `-s` | Dataset (`vista`, `vost`) and split (default `val`) |
| `--export_video` | Also save an overlay video (`.mp4`) per sequence |
| `--export_graph` | Also save the state graph (`.svg`) per sequence |

Any config value can be overridden from the command line, e.g. `SAMPLING_K=16`. Predictions are written as one JSON per sequence under the config's `paths.outdir`.

### Configs

All configs are in `configs/`. `SAM2` / `SAM3` in the name picks the tracker.

| Config | Transformations | MLLM judge |
| :-- | :-: | :-: |
| `only_SAM2` / `only_SAM3` | ✗ | ✗ |
| `default_SAM2` / `default_SAM3` | ✓ | ✗ |
| `SAM2_qwen35bfp8` / `SAM3_qwen35bfp8` | ✓ | Qwen3.5-35B-A3B |

### MLLM server

The judge and the labeler talk to any **OpenAI-compatible endpoint**. For open models, serve them with [vLLM](https://docs.vllm.ai/) and point `MODEL_PATH`, `BASE_URL` and `PORT` in the config at it. `--served-model-name` must match `MODEL_PATH`:

```bash
vllm serve <model-path> --served-model-name Qwen3.5-35B-A3B-FP8 --port 8000 \
    --limit-mm-per-prompt '{"image": 5}' --allowed-local-media-path $DATASET_FOLDER
```

> [!NOTE]
> The MLLM Judge sends the frames as file paths so you need to allow for local media. Remote MLLM Judges are not supported as of now.

For OpenAI models (`configs/mllm_labeler/gpt54*.yaml`), leave `API_KEY` empty and export `OPENAI_API_KEY`.

### Tools

*Semantic labeling*: if you ran `main.py` with `--no-semantic`, objects and transformations in the predictions have no names yet. This script fills them in with the semantic MLLM:

```bash
uv run scripts/semantic_label_predictions.py -c <config> -m <mllm-config> -d vista
```

*Artifacts*: export the overlay videos and state graphs from existing predictions:

```bash
uv run scripts/export_artifacts.py -c <config> -d vista --video --graph
```

## Evaluation

Use the same `-c` / `-m` / `-d` flags as the run you want to evaluate.

*VISTA*: `eval.py` evaluates the predictions on VISTA, or on any dataset annotated with the same transformation structure:

```bash
uv run eval.py -c configs/SAM3_qwen35bfp8.yaml \
    -m configs/mllm_labeler/qwen35bfp8.yaml -d vista
```

*VOST-TAS*: `eval_vost_tas.py` evaluates the predictions on VOST, using the VOST-TAS annotations:

```bash
uv run eval_vost_tas.py -c configs/SAM3_qwen35bfp8.yaml \
    -m configs/mllm_labeler/qwen35bfp8.yaml -d vost
```

*Semantic evaluation*: with `-m`, the object names and transformation descriptions are also evaluated, using Gemma 4 as an LLM judge. The judge is not configurable yet: it expects a vLLM server on `localhost:8000` that serves the model as `Gemma-4-31B-FP8`:

```bash
vllm serve <gemma-4-31b-path> --served-model-name Gemma-4-31B-FP8 --port 8000
```

To evaluate only tracking and state graphs, drop `-m`, so no LLM judge is needed.

## Results

Tracking and state-graph results on VISTA, as mean ± std over seeds 0, 1 and 2. Best per tracker in **bold**. See the paper for the full comparison.

<div align="center">
<table>
<thead>
<tr><th rowspan="2" align="left">Tracker</th><th rowspan="2" align="left">Method</th><th colspan="2">Tracking</th><th colspan="3">State graph</th><th rowspan="2">FPS&nbsp;↑</th></tr>
<tr><th>𝒥&nbsp;↑</th><th>𝒥<sub>tr</sub>&nbsp;↑</th><th>𝒮&nbsp;↑</th><th>𝒯<sub>F1</sub>&nbsp;↑</th><th>ℋ<sub>F1</sub>&nbsp;↑</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">SAM&nbsp;2.1</td><td>TIPO w/o Judge</td><td align="center">90.4&nbsp;<sub>±0.6</sub></td><td align="center">87.2&nbsp;<sub>±0.9</sub></td><td align="center">76.8&nbsp;<sub>±4.9</sub></td><td align="center">36.7&nbsp;<sub>±3.3</sub></td><td align="center">19.7&nbsp;<sub>±4.7</sub></td><td align="center"><b>3.0</b>&nbsp;<sub>±0.1</sub></td></tr>
<tr><td>TIPO</td><td align="center"><b>90.7</b>&nbsp;<sub>±0.3</sub></td><td align="center"><b>88.6</b>&nbsp;<sub>±0.2</sub></td><td align="center"><b>78.9</b>&nbsp;<sub>±2.1</sub></td><td align="center"><b>44.7</b>&nbsp;<sub>±1.6</sub></td><td align="center"><b>27.1</b>&nbsp;<sub>±2.2</sub></td><td align="center">1.8&nbsp;<sub>±0.0</sub></td></tr>
<tr><td rowspan="2">SAM&nbsp;3</td><td>TIPO w/o Judge</td><td align="center"><b>91.5</b>&nbsp;<sub>±1.0</sub></td><td align="center"><b>88.0</b>&nbsp;<sub>±1.4</sub></td><td align="center">77.4&nbsp;<sub>±3.4</sub></td><td align="center">33.4&nbsp;<sub>±1.8</sub></td><td align="center">16.8&nbsp;<sub>±0.0</sub></td><td align="center"><b>3.9</b>&nbsp;<sub>±0.1</sub></td></tr>
<tr><td>TIPO</td><td align="center">91.4&nbsp;<sub>±0.2</sub></td><td align="center">87.8&nbsp;<sub>±0.5</sub></td><td align="center"><b>80.2</b>&nbsp;<sub>±2.8</sub></td><td align="center"><b>42.5</b>&nbsp;<sub>±1.5</sub></td><td align="center"><b>28.3</b>&nbsp;<sub>±1.5</sub></td><td align="center">2.4&nbsp;<sub>±0.1</sub></td></tr>
</tbody>
</table>
</div>

Configs: TIPO w/o Judge uses [`default_SAM2`](configs/default_SAM2.yaml) / [`default_SAM3`](configs/default_SAM3.yaml), TIPO uses [`SAM2_qwen35bfp8`](configs/SAM2_qwen35bfp8.yaml) / [`SAM3_qwen35bfp8`](configs/SAM3_qwen35bfp8.yaml).

To run a single seed, pass `--seed` and give each run its own output folder:

```bash
uv run main.py -c configs/SAM3_qwen35bfp8.yaml \
    -m configs/mllm_labeler/qwen35bfp8.yaml -d vista \
    --seed 1 paths.outdir=out/ours-SAM3_qwen35bfp8-seed1/
```

## Development

<details>
<summary><b>Full <a href="https://pyrefly.org">pyrefly</a> support for SAM 2 and SAM 3</b></summary>

<br>

Add a `pyrefly.toml` to both `deps/sam2` and `deps/sam3`:

```toml
python-interpreter-path = "../../.venv/bin/python"
```

Then fix a typing error in SAM 3 by applying this patch from `deps/sam3`:

```diff
--- a/sam3/model/sam3_video_base.py
+++ b/sam3/model/sam3_video_base.py
@@ -24,6 +24,7 @@ from sam3.model.data_misc import BatchedDatapoint
 from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores, mask_to_box
 from sam3.perflib.masks_ops import mask_iou
 from sam3.train.masks_ops import mask_iom, rle_encode
+from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
 from torch import nn, Tensor
 
 logger = get_logger(__name__)
@@ -260,7 +261,7 @@ class Sam3VideoBase(nn.Module):
     def __init__(
         self,
         detector: nn.Module,
-        tracker: nn.Module,
+        tracker: Sam3TrackerPredictor,
         # prob threshold for detection outputs -- only keep detections above this threshold
         # enters NMS and det-to-track matching
         score_threshold_detection=0.5,
```

</details>

## Citation

```bibtex
@article{gardenal2026takingapartputtingtogether,
  title   = {Taking Apart and Putting Together: Tracking Objects Through Physical Transformations},
  author  = {Gardenal, Davide and Liu, Mingxuan and Vella, Flavio and Guimard, Quentin and Ricci, Elisa},
  journal = {arXiv preprint arXiv:2611.27398},
  year    = {2026}
}
```

## Acknowledgements

We thank the authors of the work this project builds on:

- [SAM 2](https://github.com/facebookresearch/sam2) and [SAM 3](https://github.com/facebookresearch/sam3), the trackers at the core of TIPO.
- [VOST](https://www.vostdataset.org/) and [TubeletGraph](https://github.com/YihongSun/TubeletGraph), whose VOST-TAS annotations we evaluate on.

## License

TIPO is released under the [CC BY 4.0](LICENSE) license. SAM 2 and SAM 3 in `deps/` keep their own licenses ([Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) and the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)).