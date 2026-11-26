import argparse
import json
import os
import os.path as osp
import random
import time
from argparse import Namespace
from glob import glob
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import opts
import utils
from state_graph import StateGraph
from tracker.mllm_judge import MLLMJudge
from tracker.mllm_labeler import MLLMLabeler
from tracker.sam2_tracker import SAM2
from tracker.sam3_tracker import SAM3
from tracker.tipo import track_transformations
from utils import check_device

# Tracker class definition
SAM = SAM3

def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args_and_conf() -> tuple[Namespace, Any]:
    # Load config and args
    args, overrides = get_parser().parse_known_args()
    conf = opts.load_conf(args.config, args.semantic_config, overrides) # + CLI overrides
    if args.semantic_label and conf.get("SEMANTIC_MLLM") is None:
        print("WARNING: no semantic MLLM config (-m), semantic labeling is skipped.")
        args.semantic_label = False

    # Set SAM variant
    global SAM
    if conf.SAM_VERSION == 3: SAM = SAM3 # pyrefly: ignore
    elif conf.SAM_VERSION == 2: SAM = SAM2

    check_device(args.device)

    # Setup prediction dir 
    if args.semantic_label:
        args.pred_dir = osp.join(osp.abspath(conf.paths.outdir), conf.SEMANTIC_MLLM.out_path, f"{args.dataset}-{args.split}")
    else:
        args.pred_dir = osp.join(osp.abspath(conf.paths.outdir), f"{args.dataset}-{args.split}")
    os.makedirs(args.pred_dir, exist_ok=True)
    OmegaConf.save(conf, osp.join(conf.paths.outdir, "config.yaml"))
    print(OmegaConf.to_yaml(conf))

    # Video/Graph dir setup
    args.export_dir = osp.abspath(args.export_dir) if args.export_dir is not None else None
    if (args.export_video or args.export_graph) and args.export_dir is None:
        args.export_dir = osp.join(args.pred_dir, "artifacts")
    if args.export_dir is not None:
        os.makedirs(args.export_dir, exist_ok=True)

    return args, conf

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="""
        Run TIPO over a DAVIS-style dataset and export per-sequence JSON predictions.
        """
    )
    p.add_argument("-c", "--config", type=str, help='Path to the config file', required=True)
    p.add_argument("-m", "--mllm-config", type=str, default=None, dest="semantic_config", help="Path to the semantic MLLM config file")
    p.add_argument("-d", "--dataset", type=str, help="Name of the dataset (found in the config)", required=True)
    p.add_argument("-s", "--split", type=str, default="val", help="ImageSets split (val/test/train)")
    p.add_argument("--device", type=str, default="cuda", help="torch device, e.g. cuda or cpu")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for Python random, NumPy, and PyTorch (CUDA manual seeds and cuDNN deterministic mode on GPU). "
    )
    p.add_argument(
        "--no-semantic",
        action="store_false",
        dest="semantic_label",
        help="Do not fill object names and transformation descriptions with the semantic MLLM",
    )
    p.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Export optional inspection artifacts under <export_dir>/<seq>/ (default: <pred_dir>/artifacts when exporting).",
    )
    p.add_argument(
        "--export_video",
        action="store_true",
        help="Export an inspection video with masks overlaid (requires --export_dir).",
    )
    p.add_argument(
        "--export_graph",
        action="store_true",
        help="Export a compressed state graph SVG (requires --export_dir).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <seq>.json files if present.",
    )
    return p


def read_sequences(split_dir: str, split: str) -> list[str]:
    imagesets = osp.join(split_dir, f"{split}.txt")
    with open(imagesets, "r") as f:
        return [x.strip() for x in f if x.strip()]

def get_track_prompts(dataset_conf, seq: str, unified_tracks: bool) -> list[tuple[int|None, dict[int, torch.Tensor]]]:
    anno_mask_path = osp.join(dataset_conf.anno_dir, seq)
    anno_paths = sorted(glob(osp.join(anno_mask_path, "*.png")))
    if len(anno_paths) == 0:
        raise FileNotFoundError(f"No annotation PNGs found in {anno_mask_path}")

    if unified_tracks:
        # Get prompt masks from sequence metadata
        metadata_path = osp.join(dataset_conf.data_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f: # crash if we do not have the metadata file
            prompt_masks = json.load(f).get(seq, {}).get("prompt_masks")
        if not prompt_masks: raise ValueError(f"Initial prompt metadata not found for sequence {seq} in {metadata_path}")

        # Create the prompt
        unif_prompt = {}
        for frame_key, obj_ids in prompt_masks.items():
            frame_idx = int(frame_key)
            frame_mask = torch.from_numpy(np.array(Image.open(anno_paths[frame_idx]))).to(torch.int32) - 1 # masks map with int ids
            frame_mask[~torch.isin(frame_mask, torch.tensor(obj_ids, dtype=torch.int32))] = -1 # remove unprompted ids
            unif_prompt[frame_idx] = frame_mask + 1 
        return [(None, unif_prompt)]

    else:
        # Use only frame 0 objects
        init_mask = torch.Tensor(np.array(Image.open(anno_paths[0])))
        init_objects = init_mask.unique()
        init_objects = init_objects[init_objects != 0]
        return [(int(obj), {0: init_mask == obj}) for obj in init_objects]

def export_artifacts(graph: StateGraph, seq: str, seq_name: str, args: Namespace):
    seq_out_dir = osp.join(args.export_dir, seq)
    os.makedirs(seq_out_dir, exist_ok=True)
    if args.export_video:
        graph.export_video(osp.join(seq_out_dir, f"{seq_name}.mp4"))
    if args.export_graph:
        graph.export_graph(osp.join(seq_out_dir, f"{seq_name}.svg"))

def track_sequence(
    tracker: SAM,
    mllm: MLLMJudge | None,
    semantic_mllm: MLLMLabeler | None,
    video_dir: str,
    seq: str,
    args: Namespace,
    conf: Any,
    seq_fps: float,
    unified_tracks: bool        # T -> VISTA-like, F -> VOST-like
):
    prompts = get_track_prompts(conf.datasets[args.dataset], seq, unified_tracks)

    # Prompt structure
    # [(None, {frame_idx: prompt_masks, ...})] -> VISTA unified tracking (using all the objects)
    # [(1, {...}), (2, {...}), ...] -> VOST independent tracking (one prompted object at the time)

    # Per-prompt tracking
    for prompt_id, prompt_objs in prompts:
        seq_name = f"{seq}" if prompt_id is None else f"{seq}_{prompt_id}" # if unified we use all the prompted objects so we have a single prompt
        
        # Sequence cache check
        out_json = osp.join(args.pred_dir, f"{seq_name}.json")
        if osp.exists(out_json) and not args.overwrite:
            print(f"{out_json} already found!")
            continue

        # TRACKING
        t0 = time.perf_counter()
        tracker.init_session(video_dir)
        graph = track_transformations(tracker, mllm, prompt_objs, conf, semantic_mllm, fps=float(seq_fps))
        tracking_elapsed = time.perf_counter() - t0
        
        # EXPORT
        graph.sequence = seq_name
        graph.tracking_elapsed_seconds = tracking_elapsed
        graph.export_predictions_json(out_json)
        if args.export_dir is not None:
            export_artifacts(graph, seq, seq_name, args)


def main() -> None:
    args, conf = get_args_and_conf()
    set_random_seeds(args.seed)
    dataset_conf = conf.datasets[args.dataset]
    
    mllm = MLLMJudge(conf.MLLM) if conf.ENABLE_MLLM else None
    tracker = SAM(video_path=None, device=args.device) # init tracker model

    semantic_mllm = MLLMLabeler(conf.SEMANTIC_MLLM) if args.semantic_label else None

    if mllm is not None: mllm.wait_for_server()
    if semantic_mllm is not None: semantic_mllm.wait_for_server()

    # Main sequence loop
    sequences = read_sequences(dataset_conf.split_dir, args.split)
    fps_map = utils.get_sequence_fps_map(dataset_conf, sequences)
    for seq in tqdm(sequences, desc=f"Exporting JSON ({args.split})"):
        video_dir = osp.join(dataset_conf.image_dir, seq)
        if mllm is not None: mllm.set_video(video_dir, fps_map[seq])
        track_sequence(tracker, mllm, semantic_mllm, video_dir, seq, args, conf, fps_map[seq], unified_tracks=dataset_conf.unified_tracking)


if __name__ == "__main__":
    main()
