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


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export inspection artifacts (overlay video, state graph SVG) from existing prediction JSONs, "
            "under <pred_dir>/artifacts/<sequence>/ (same layout as main.py). "
            "Single-clip mode: pass --prediction-json and --sequence together."
        ),
    )
    p.add_argument("-c", "--config", type=str, required=True,
                   help="Path to the run config (dataset paths + prediction outdir).")
    p.add_argument("-m", "--mllm-config", type=str, default=None, dest="semantic_config",
                   help="Semantic MLLM config: read the labeled predictions under its out_path.")
    p.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset (found in the config).")
    p.add_argument("-s", "--split", type=str, default="val", help="ImageSets split (val/test/train).")
    p.add_argument("--video", action="store_true", help="Export the video with the masks overlaid (.mp4).")
    p.add_argument("--graph", action="store_true", help="Export the state graph (.svg).")
    p.add_argument("--alpha", type=float, default=0.5, help="Opacity of the video mask overlay.")
    p.add_argument("--fps", type=float, default=None, help="Video FPS for prediction JSONs without an `fps` field.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts.")
    p.add_argument("--prediction-json", type=str, default=None, metavar="PATH",
                   help="Export a single prediction JSON (writes under <dirname(json)>/artifacts/). Requires --sequence.")
    p.add_argument("--sequence", type=str, default=None,
                   help="Sequence (frame folder under the dataset image_dir) of --prediction-json.")
    return p


def export(pred_json: str, video_path: str, out_dir: str, args: argparse.Namespace) -> None:
    graph = StateGraph.from_predictions_json(pred_json, video_path)
    graph.fps = graph.fps or args.fps
    os.makedirs(out_dir, exist_ok=True)
    exporters = [
        (args.video, ".mp4", lambda p: graph.export_video(p, alpha=args.alpha)),
        (args.graph, ".svg", graph.export_graph),
    ]
    for enabled, ext, export_fn in exporters:
        if not enabled:
            continue
        out_path = osp.join(out_dir, f"{graph.sequence}{ext}")
        if osp.exists(out_path) and not args.overwrite:
            tqdm.write(f"[Skip] {osp.basename(out_path)}: already exists")
            continue
        export_fn(out_path)
        tqdm.write(f"[OK] {out_path}")


def main() -> None:
    parser = get_parser()
    args, overrides = parser.parse_known_args()
    if not (args.video or args.graph):
        parser.error("nothing to export: pass --video and/or --graph.")
    if (args.prediction_json is None) != (args.sequence is None):
        parser.error("--prediction-json and --sequence must be used together.")

    conf = opts.load_conf(args.config, args.semantic_config, overrides)
    dataset_conf = conf.datasets[args.dataset]

    if args.prediction_json:
        # Single clip
        pred_dir = osp.dirname(osp.abspath(args.prediction_json))
        work_items = [(osp.abspath(args.prediction_json), args.sequence)]
    else:
        # Predictions are <seq>.json (unified tracking) or <seq>_<prompt_id>.json (one per prompted object)
        sub_dir = conf.SEMANTIC_MLLM.out_path if args.semantic_config else ""
        pred_dir = osp.join(osp.abspath(conf.paths.outdir), sub_dir, f"{args.dataset}-{args.split}")
        if not osp.isdir(pred_dir):
            raise SystemExit(f"Prediction directory not found: {pred_dir}")
        work_items = [
            (pred_json, seq)
            for seq in utils.read_sequences(dataset_conf.split_dir, args.split)
            for pred_json in utils.list_sequence_prediction_candidates(pred_dir, seq)
        ]

    for pred_json, seq in tqdm(work_items, desc="Exporting artifacts"):
        video_path = osp.join(dataset_conf.image_dir, seq)
        if not osp.isdir(video_path):
            tqdm.write(f"[Skip] {osp.basename(pred_json)}: video directory not found at {video_path}")
            continue
        try:
            export(pred_json, video_path, osp.join(pred_dir, "artifacts", seq), args)
        except Exception as e:  # noqa: BLE001 -- keep exporting the other sequences if one fails
            tqdm.write(f"[Error] {osp.basename(pred_json)}: {e}")


if __name__ == "__main__":
    main()
