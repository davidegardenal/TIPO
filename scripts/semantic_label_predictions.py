import argparse
import os
import os.path as osp
import sys

from tqdm import tqdm

# Allow running as a top-level script outside the package.
_repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import opts
import utils
from state_graph import StateGraph
from tracker.mllm_labeler import MLLMLabeler


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fill TODO object names and transformation descriptions in prediction JSONs via a semantic MLLM.",
    )
    p.add_argument("-c", "--config", type=str, required=True,
                    help="Path to the run config (dataset paths + prediction outdir).")
    p.add_argument("-m", "--mllm-config", type=str, required=True, dest="semantic_config",
                    help="Path to the semantic MLLM config (model, base URL, API key; "
                    "out_path is a subfolder under the main config paths.outdir).")
    p.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset (found in the config).")
    p.add_argument("-s", "--split", type=str, default="val", help="ImageSets split (val/test/train).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing filled JSON files.")
    return p


def main() -> None:
    args, overrides = get_parser().parse_known_args()
    conf = opts.load_conf(args.config, args.semantic_config, overrides)
    dataset_conf = conf.datasets[args.dataset]

    # Resolve prediction directory (same layout as main.py)
    pred_dir = osp.join(osp.abspath(conf.paths.outdir), f"{args.dataset}-{args.split}")
    if not osp.isdir(pred_dir):
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    # Resolve output directory: <main paths.outdir>/<MLLM out_path>/<dataset-split>/
    out_dir = osp.join(osp.abspath(conf.paths.outdir), conf.SEMANTIC_MLLM.out_path, f"{args.dataset}-{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    # Build MLLM client
    mllm = MLLMLabeler(conf.SEMANTIC_MLLM)
    mllm.wait_for_server()

    # Predictions are <seq>.json (unified tracking) or <seq>_<prompt_id>.json (one per prompted object)
    work_items = [
        (pred_json, osp.join(dataset_conf.image_dir, seq))
        for seq in utils.read_sequences(dataset_conf.split_dir, args.split)
        for pred_json in utils.list_sequence_prediction_candidates(pred_dir, seq)
    ]

    for pred_json_path, video_path in tqdm(work_items, desc="Semantic labeling"):
        bn = osp.basename(pred_json_path)
        out_path = osp.join(out_dir, bn)
        if osp.exists(out_path) and not args.overwrite:
            tqdm.write(f"[Skip] {bn}: already exists at {out_path}")
            continue

        if not osp.isdir(video_path):
            tqdm.write(f"[Skip] {bn}: video directory not found at {video_path}")
            continue

        graph = StateGraph.from_predictions_json(pred_json_path, video_path)
        graph.fill_semantic_labels(mllm)
        graph.export_predictions_json(out_path)
        tqdm.write(f"[OK] {bn} -> {out_path}")


if __name__ == "__main__":
    main()
