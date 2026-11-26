import csv
import io
import json
import os
import os.path as osp
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from glob import glob
from typing import Any, TypedDict

import numpy as np
import openai
import torch
import tqdm
from PIL import Image
from pycocotools import mask as mask_utils


def check_device(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but is not available. Run on a GPU machine or pass `--device cpu`."
        )

def iou(mask_i: torch.Tensor, mask_j: torch.Tensor) -> torch.Tensor:
    return (mask_i & mask_j).sum() / (mask_i | mask_j).sum()

def inter_over_a(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    return (mask_a & mask_b).sum() / mask_a.sum()

@contextmanager
def tqdm_disabled(active: bool = True):
    if active:
        original_init = tqdm.tqdm.__init__
        
        def silent_init(self, *args, **init_kwargs):
            init_kwargs['disable'] = True
            original_init(self, *args, **init_kwargs)
            
        tqdm.tqdm.__init__ = silent_init
    
        try:
            yield
        finally:
            tqdm.tqdm.__init__ = original_init
    else: yield

def get_sequence_fps_map(dataset_conf: Any, sequences: list[str]) -> dict[str, float]:
    """
    Resolve a per-sequence FPS map.

    - If `dataset_conf.fps` is set, we use it for every sequence.
    - Otherwise, we read `<data_dir>/metadata.json`, which maps sequence name -> { "fps": <float> }.
    """
    dataset_fps = getattr(dataset_conf, "fps", None)
    if dataset_fps is not None:
        return {seq: float(dataset_fps) for seq in sequences}

    data_dir = getattr(dataset_conf, "data_dir", None)
    if not data_dir:
        raise ValueError("dataset_conf.data_dir is required when dataset_conf.fps is null/None.")

    metadata_path = osp.join(data_dir, "metadata.json")
    if not osp.isfile(metadata_path):
        raise FileNotFoundError(f"Missing metadata.json for FPS resolution: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {seq: float(metadata[seq]['fps']) for seq in sequences}

## EVAL UTILS ##

class RLE(TypedDict):
    """COCO run-length encoded mask (same shape as pycocotools' ``_EncodedRLE``)."""
    size: list[int]
    counts: str | bytes

def decode_rle(rle_mask: RLE) -> np.ndarray:
    counts = rle_mask["counts"]
    rle: RLE = {
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
        "size": rle_mask["size"],
    }
    return mask_utils.decode(rle).astype(bool)

def _ensure_rle_bytes(rle: dict[str, Any]) -> RLE:
    """Normalize RLE so that ``counts`` is bytes (required by pycocotools C ops)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"counts": counts, "size": rle["size"]}

def load_pred_rle(pred_json_path: str) -> tuple[dict[int, dict[int, RLE]], list[int], dict[str, Any]]:
    """Load prediction masks as frame -> track_id -> RLE without dense tensors."""
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    frame_indices = sorted(int(k) for k in pred_data["prediction"])
    if len(frame_indices) == 0:
        raise ValueError(f"Prediction JSON has no frames: {pred_json_path}")

    pred_rle: dict[int, dict[int, RLE]] = {}
    for frame_idx in frame_indices:
        frame_pred = pred_data["prediction"][str(frame_idx)]
        frame_rle: dict[int, RLE] = {}
        for track_id_str, rle in frame_pred.items():
            frame_rle[int(track_id_str)] = _ensure_rle_bytes(rle)
        pred_rle[frame_idx] = frame_rle

    return pred_rle, frame_indices, pred_data

def load_gt_rle(annotation_dir: str, sequence: str) -> tuple[dict[int, dict[int, RLE]], list[str]]:
    """Load GT label-map PNGs as frame -> obj_id -> RLE."""
    seq_anno_dir = osp.join(annotation_dir, sequence)
    anno_paths = sorted(glob(osp.join(seq_anno_dir, "*.png")))

    gt_rle: dict[int, dict[int, RLE]] = {}
    for frame_idx, path in enumerate(anno_paths):
        label_map = np.array(Image.open(path), dtype=np.int64)
        frame_rle: dict[int, RLE] = {}
        for obj_id in np.unique(label_map):
            if obj_id == 0:
                continue
            binary = (label_map == obj_id).astype(np.uint8)
            frame_rle[int(obj_id)] = mask_utils.encode(np.asfortranarray(binary))
        gt_rle[frame_idx] = frame_rle

    return gt_rle, anno_paths

def load_metadata(data_dir: str, sequence: str) -> dict:
    meta_path = osp.join(data_dir, 'metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)[sequence]
    return metadata

def build_pred_transform(pred_data: dict[str, Any]) -> tuple[list, dict[int, str]]:
    out = []
    obj_names = {}
    for event in pred_data.get("transformations", []):
        if len(event['beforeObjects']) == 1 and len(event['afterObjects']) == 1: continue # Keep only meaningful transformations
        beforeObjects = []
        for bo in event['beforeObjects']: 
            beforeObjects.append(bo['obj_id'])
            obj_names[bo['obj_id']] = bo['name']
        event['beforeObjects'] = beforeObjects
        afterObjects = []
        for bo in event['afterObjects']:
            afterObjects.append(bo['obj_id'])
            obj_names[bo['obj_id']] = bo['name']
        event['afterObjects'] = afterObjects
        
        out.append(event)

    # Reorder transformations based on frame idx
    out.sort(key=lambda x: x['frameIdx'])
    return out, obj_names

def build_gt_transform(trans_dir: str, sequence: str) -> tuple[list, dict[int, str]]:
    transform_path = osp.join(trans_dir, f"{sequence}.json")
    if not osp.isfile(transform_path):
        return [], {}
    
    with open(transform_path, "r", encoding="utf-8") as f:
        raw_events = json.load(f)
    out = []
    obj_names = {}
    for event in raw_events:
        del event['id']
        beforeObjects = []
        for bo in event['beforeObjects']: 
            beforeObjects.append(bo['obj_id']+1)
            obj_names[bo['obj_id']+1] = bo['name']
        event['beforeObjects'] = beforeObjects
        afterObjects = []
        for bo in event['afterObjects']:
            afterObjects.append(bo['obj_id']+1)
            obj_names[bo['obj_id']+1] = bo['name']
        event['afterObjects'] = afterObjects
        out.append(event)

    # Reorder transformations based on frame idx
    out.sort(key=lambda x: x['frameIdx'])
    return out, obj_names

def read_sequences(split_dir: str, split: str) -> list[str]:
    imagesets = osp.join(split_dir, f"{split}.txt")
    with open(imagesets, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def list_sequence_prediction_candidates(pred_dir: str, sequence: str) -> list[str]:
    """Prediction JSONs of *sequence*: ``<sequence>.json`` (unified tracking) and
    ``<sequence>_<prompt_id>.json`` (one per prompted object)."""
    if not osp.isdir(pred_dir):
        return []
    pattern = re.compile(rf"{re.escape(sequence)}(_\d+)?\.json")
    candidates = [osp.join(pred_dir, f) for f in os.listdir(pred_dir) if pattern.fullmatch(f)]
    return sorted(candidates)

## EVAL LOGGING UTILS ##
@dataclass(frozen=True)
class EvalTerm:
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"
    YELLOW: str = "\033[33m"
    CYAN: str = "\033[36m"


EVAL_TERM = EvalTerm()


def print_eval_banner(title: str, width: int = 72) -> None:
    bar = "═" * width
    t = EVAL_TERM
    print(f"{t.CYAN}{t.BOLD}{bar}{t.RESET}")
    print(f"{t.CYAN}{t.BOLD} {title}{t.RESET}")
    print(f"{t.CYAN}{t.BOLD}{bar}{t.RESET}")


def _format_eval_metric_cell(key: str, value: float) -> str:
    if key == "T_lag":
        return f"{value:.2f}"
    return f"{value:.4f}"


EVAL_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "J": r"$\mathcal{J}$",
    "J_tr": r"$\mathcal{J}_{tr}$",
    "R": r"$\mathcal{R}$",
    "T_rec": r"$\mathcal{T}_{r}$",
    "T_pre": r"$\mathcal{T}_{p}$",
    "F_1t": r"$\mathcal{T}_{F_1}$",
    "T_lag": r"$\mathcal{T}_{lag}$",
    "H_st": r"$\mathcal{H}_{st}$",
    "H": r"$\mathcal{H}$",
}

def _eval_metric_column_order(mean_metrics: dict[str, float]) -> list[str]:
    preferred = ("J", "J_tr", "R", "T_rec", "T_pre", "F_1t", "T_lag")
    keys = [k for k in preferred if k in mean_metrics]
    keys.extend(sorted(k for k in mean_metrics if k not in preferred))
    return keys

def build_eval_markdown_report(
    rows: list[tuple[str, dict[str, float]]],
    mean_metrics: dict[str, float],
    *,
    dataset: str,
    split: str,
    pred_dir: str,
) -> str:
    if not rows:
        return ""
    metric_keys = _eval_metric_column_order(mean_metrics)
    header_labels = [EVAL_METRIC_DISPLAY_NAMES.get(k, k) for k in metric_keys]
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# TIPO evaluation results",
        "",
        f"- **Dataset:** `{dataset}`",
        f"- **Split:** `{split}`",
        f"- **Predictions:** `{pred_dir}`",
        f"- **Generated:** {ts}",
        "",
        "| Sequence | " + " | ".join(header_labels) + " |",
        "| :--- | " + " | ".join(":---:" for _ in metric_keys) + " |",
    ]
    for seq, metrics in rows:
        cells = [_format_eval_metric_cell(k, float(metrics[k])) for k in metric_keys]
        lines.append("| `" + seq.replace("`", "'") + "` | " + " | ".join(cells) + " |")

    avg_cells = [_format_eval_metric_cell(k, float(mean_metrics[k])) for k in metric_keys]
    lines.append("| **Average (dataset)** | " + " | ".join(avg_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build_eval_csv_report(
    rows: list[tuple[str, dict[str, float]]],
    mean_metrics: dict[str, float],
) -> str:
    """CSV table only: header with translated metric names, per-sequence rows, then dataset average."""
    if not rows:
        return ""
    metric_keys = _eval_metric_column_order(mean_metrics)
    header_labels = [EVAL_METRIC_DISPLAY_NAMES.get(k, k) for k in metric_keys]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Sequence", *header_labels])
    for seq, metrics in rows:
        writer.writerow([seq, *[_format_eval_metric_cell(k, float(metrics[k])) for k in metric_keys]])
    writer.writerow(
        ["Average (dataset)", *[_format_eval_metric_cell(k, float(mean_metrics[k])) for k in metric_keys]]
    )
    return buf.getvalue()


def print_eval_results_table(
    rows: list[tuple[str, dict[str, float]]],
    mean_metrics: dict[str, float],
    *,
    seq_col_max: int = 40,
) -> None:
    """Fixed-width Unicode box table for the terminal."""
    if not rows:
        return

    t = EVAL_TERM
    metric_keys = list(mean_metrics.keys())
    headers = ["Sequence"] + metric_keys

    def _trunc_seq(name: str) -> str:
        if len(name) <= seq_col_max:
            return name
        return name[: max(1, seq_col_max - 1)] + "…"

    data_rows: list[list[str]] = []
    for seq, metrics in rows:
        data_rows.append(
            [_trunc_seq(seq)] + [_format_eval_metric_cell(k, float(metrics[k])) for k in metric_keys]
        )
    avg_row = ["Average (dataset)"] + [
        _format_eval_metric_cell(k, float(mean_metrics[k])) for k in metric_keys
    ]

    widths = [len(h) for h in headers]
    for r in data_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    for i, cell in enumerate(avg_row):
        widths[i] = max(widths[i], len(cell))

    thin_h = "─"
    top = "╭" + "┬".join(thin_h * (w + 2) for w in widths) + "╮"
    mid = "├" + "┼".join(thin_h * (w + 2) for w in widths) + "┤"
    bot = "╰" + "┴".join(thin_h * (w + 2) for w in widths) + "╯"

    def _row_cells(cells: list[str], *, is_header: bool = False) -> str:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i == 0:
                padded = cell.ljust(w)
            else:
                padded = cell.rjust(w)
            if is_header:
                padded = f"{t.BOLD}{padded}{t.RESET}"
            parts.append(f" {padded} ")
        inner = "│".join(parts)
        return "│" + inner + "│"

    print(top)
    print(_row_cells(headers, is_header=True))
    print(mid)
    for r in data_rows:
        print(_row_cells(r))
    print(mid)
    print(_row_cells(avg_row))
    print(bot)

def _empty_rle(size: list[int]) -> RLE:
    return mask_utils.encode(np.zeros((size[0], size[1]), dtype=np.uint8, order='F'))

# VOST-TAS eval utils
def parse_vost_video_and_obj_id(seq_name: str, anno_dir: str) -> tuple[str, int]:
    """Parse a VOST-TAS sequence name into (vost_video, object_id).

    The convention is ``<vost_video>_<obj_id>`` where ``<vost_video>`` is
    itself a directory under *anno_dir*.  Since video names can contain
    trailing numbers (e.g. ``556_cut_tomato``), we resolve ambiguity by
    checking which candidate directory actually exists.
    """
    parts = seq_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        candidate = parts[0]
        if osp.isdir(osp.join(anno_dir, candidate)):
            return candidate, int(parts[1])

    # Fallback: try progressively longer prefixes
    tokens = seq_name.split("_")
    for i in range(len(tokens) - 1, 0, -1):
        candidate = "_".join(tokens[:i])
        suffix = "_".join(tokens[i:])
        if suffix.isdigit() and osp.isdir(osp.join(anno_dir, candidate)):
            return candidate, int(suffix)

    raise ValueError(
        f"Cannot parse VOST video + object ID from '{seq_name}' "
        f"(no matching directory in {anno_dir})"
    )

def gt_rle_for_object(
    gt_rle: dict[int, dict[int, RLE]],
    obj_id: int,
    frame_indices: list[int],
) -> dict[int, dict[int, RLE]]:
    """Slice full GT to one object for J/J_tr (same dict structure, one key per frame)."""
    result: dict[int, dict[int, RLE]] = {}
    for f in frame_indices:
        objs = gt_rle.get(f, {})
        if obj_id in objs:
            result[f] = {obj_id: objs[obj_id]}
        else:
            result[f] = {}
    return result

def collapse_predictions_to_labelmap(
    objects: dict,
    num_frames: int,
    img_size: tuple[int, int],
) -> list[np.ndarray]:
    """Collapse per-object prediction JSONs into per-frame label-map arrays.

    Replicates the json_to_png logic from eval.py: objects are written
    in sorted obj_id order; overlapping pixels go to the last-written object.
    """
    obj_predictions = {oid: path for oid, (_, path) in objects.items()}
    label_maps = [np.zeros(img_size, dtype=np.uint8) for _ in range(num_frames)]

    for obj_id in sorted(obj_predictions.keys()):
        pred_rle, _, _ = load_pred_rle(obj_predictions[obj_id])

        for frame_idx, tracks in pred_rle.items():
            if frame_idx >= num_frames:
                continue
            masks = list(tracks.values())
            if not masks:
                continue
            bmask = np.zeros(img_size, dtype=bool)
            for rle in masks:
                bmask |= decode_rle(rle)
            label_maps[frame_idx][bmask] = obj_id

    return label_maps

def extract_object_mask_rle_from_collapsed(
    label_maps: list[np.ndarray],
    obj_id: int,
    frame_indices: list[int],
) -> dict[int, dict[int, RLE]]:
    """Extract one object's mask from collapsed prediction label-maps as RLE dict.

    Returns ``frame_idx -> {obj_id: RLE}`` (or ``{}`` when absent), the
    format expected by ``eval_J_scores``.
    """
    result: dict[int, dict[int, RLE]] = {}
    for frame_idx in frame_indices:
        binary = (label_maps[frame_idx] == obj_id)
        if binary.any():
            result[frame_idx] = {
                obj_id: mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
            }
        else:
            result[frame_idx] = {}
    return result

def load_vost_tas_annotations(json_path: str) -> dict[str, dict[str, Any]]:
    """Load the full VOST-TAS annotation JSON, returning the annotations dict."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("annotations", {})

@cache
def judge_sem_acc(pred_txt: str, gt_txt: str, is_action: bool):
    if is_action:
        system_prompt = """
        You are a highly intelligent assistant that can analyze actions in text.
        """
        prompt_question = """
        Given the particular action description: “{0}”, is “{1}” similar to the verbs in this action? Please rate from -1 to 1, where -1 means completely unrelated, 0 means ambiguous, and 1 means “{1}” captures the meaning of “{0}” or is directly in it. Brief/general descriptions should still be considered as +1. Please answer with a single integer.'
        """
    else:
        system_prompt = """
        You are a highly intelligent assistant that can analyze actions and resulting objects in text.
        """
        prompt_question = """
            Given the object description “{0}”, is “{1}” similar to it? Please rate from -1 to 1, where -1 means completely unrelated, 0 means ambiguous, and 1 means “{1}” is similar. Over- or under-specified descriptions should still be considered as +1. Please answer with a single integer.'
        """

    client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key="none")
    response = client.chat.completions.create(
        model='/leonardo_scratch/fast/IscrC_EOVPS/models/Gemma-4-31B-FP8',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{
                "type": "text",
                "text": prompt_question.format(gt_txt.replace('_', ' '), pred_txt),
            }]}
        ],
        extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        temperature=0.0,
    )

    rsp = (response.choices[0].message.content or "").strip()
    outcome = int(rsp) > 0

    return outcome, rsp
