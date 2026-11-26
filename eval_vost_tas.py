import argparse
import os
import os.path as osp
import shutil
from argparse import Namespace
from collections import defaultdict
from glob import glob
from typing import Any

import numpy as np
from omegaconf import open_dict
from pycocotools import mask as mask_utils
from tqdm import tqdm

import opts
import utils
from eval import safe_lin_sum_assignment
from utils import RLE


def get_args_and_conf() -> tuple[Namespace, Any]:
    args = get_parser().parse_args()
    conf = opts.load_conf(args.config, args.mllm_config)
    with open_dict(conf): # eval_semantic is not part of the config files
        if args.mllm_config is not None:
            conf.paths.outdir = osp.join(conf.paths.outdir, conf.SEMANTIC_MLLM.out_path)
            conf.eval_semantic = True
        else:
            conf.eval_semantic = False
    return args, conf

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate predictions on the VOST-TAS dataset (J, J_tr)."
    )
    p.add_argument("-c", "--config", type=str, help="Path to the config file", required=True)
    p.add_argument("-m", "--mllm-config", type=str, required=False, dest="mllm_config",
                    help="Path to the semantic MLLM config")
    p.add_argument("-d", "--dataset", type=str, help="Name of the dataset (found in the config)", required=True)
    p.add_argument("-s", "--split", type=str, default="val", help="ImageSets split (val/test/train)")
    p.add_argument("--csv_out", type=str, default=None,
                    help="Also copy results.csv to this path (if it is an existing directory, saved as <csv_out>/results.csv)")
    return p

def print_banner(T, args, pred_dir, video_groups, tas_annotations):
    utils.print_eval_banner("TIPO evaluation (VOST-TAS)")
    print(f"{T.DIM}dataset{T.RESET}  {T.BOLD}{args.dataset}{T.RESET}   "
          f"{T.DIM}split{T.RESET}  {T.BOLD}{args.split}{T.RESET}")
    print(f"{T.DIM}pred_dir{T.RESET}  {pred_dir}")
    print(f"{T.DIM}videos{T.RESET}    {len(video_groups)}   "
          f"{T.DIM}sequences{T.RESET}  {sum(len(objs) for objs in video_groups.values())}")
    print(f"{T.DIM}VOST-TAS annotations{T.RESET}  {len(tas_annotations)}")
    print()


def eval_J_scores(
    pred_rle: dict[int, dict[int, RLE]],
    gt_rle: dict[int, dict[int, RLE]],
    frame_indices: list[int],
) -> tuple[float, float]:
    """Evaluate J and J_tr"""
    ious = np.empty(len(frame_indices))
    for i, frame_idx in enumerate(frame_indices):
        pred_masks = list(pred_rle.get(frame_idx, {}).values())
        gt_masks = list(gt_rle.get(frame_idx, {}).values())

        if not pred_masks and not gt_masks:
            ious[i] = 1.0
            continue
        if not pred_masks or not gt_masks:
            ious[i] = 0.0
            continue

        pred_merged = mask_utils.merge(pred_masks, intersect=False)
        gt_merged = mask_utils.merge(gt_masks, intersect=False)
        ious[i] = mask_utils.iou([pred_merged], [gt_merged], [0])[0][0]

    J_score = float(np.nanmean(ious))
    J_tr_score = float(np.nanmean(ious[int(len(ious) * 0.75):]))
    return J_score, J_tr_score

def eval_T_scores(gt_transform, pred_transform):
    # Build cost matrix
    C_mat = np.zeros((len(pred_transform), len(gt_transform)))
    for i, pt in enumerate(pred_transform):
        for j, gt in enumerate(gt_transform):
            C_mat[i,j] = 0 if pt['frameIdx'] >= gt['start_frame'] and pt['frameIdx'] <= gt['end_frame'] else np.inf

    p_row, g_col = safe_lin_sum_assignment(C_mat)

    tp = p_row.size
    fp = len(pred_transform) - tp
    fn = C_mat.shape[1] - g_col.size
    # NOTE: here we follow the same implementation of TubeletGraph where
    # we have 1. if the denominator is 0. This is different from our implementation on VISTA.
    T_rec = 1. if (tp + fn) == 0 else tp / (tp + fn)
    T_pre = 1. if (tp + fp) == 0 else tp / (tp + fp)
    return T_rec, T_pre, g_col, p_row, tp, fn, fp

def eval_H_scores(gt_transform, pred_transform, pred_obj_name, pred_rle, matched_gts, matched_pred, semantic_eval = False):

    def _get_iou_matrix(pred_rle_end, gt_rle_end):
        p_rles = [rle for obj, rle in pred_rle_end.items()]
        g_rles = [obj_dict['mask'] for obj_id, obj_dict in gt_rle_end.items()]
        iou_mat = np.asarray(mask_utils.iou(p_rles, g_rles, [0] * len(g_rles)))
        return (iou_mat <= 0.5).astype(np.float64)

    # Init counters
    H_matched = H_st_matched = 0
    H_st_matched = 0
    S_A_matched = S_O_matched = 0
    A_counter = O_counter = 0
    
    for m_gt, m_pred in zip(matched_gts, matched_pred):
        gt_t = gt_transform[m_gt]
        pred_t = pred_transform[m_pred]
        pred_rle_end = pred_rle[gt_t['end_frame']]
        iou_mat = _get_iou_matrix(pred_rle_end, gt_t['result_objects'])
        iou_mat[iou_mat == 1.] = np.inf # set invalid matches
        row, col = safe_lin_sum_assignment(iou_mat)
        # Check the semantic
        sem_matched = False
        if semantic_eval:
            
            act_match, _ = utils.judge_sem_acc(pred_t['description'], gt_t['action_desc'], is_action=True)
            obj_match = []
            gt_obj_names = [ob['desc'] for ob in gt_t['result_objects'].values()]
            p_obj_n = [pred_obj_name.get(k, "none") for k in pred_rle_end]
            for o_pred, o_gt in zip(row, col):
                obj_flag, _ = utils.judge_sem_acc(p_obj_n[o_pred], gt_obj_names[o_gt], is_action=False)
                obj_match.append(obj_flag)
            
            A_counter += 1
            O_counter += len(obj_match)
            S_A_matched += int(act_match)
            S_O_matched += sum(obj_match)

            if act_match and all(obj_match):
                sem_matched = True
                    
        if col.size == len(gt_t['result_objects']): # if all matched increase H_st_matched
            H_st_matched += 1
            if sem_matched: H_matched += 1
            

    H = H_matched /  len(gt_transform) if len(gt_transform) != 0 else np.nan
    H_st = H_st_matched / len(gt_transform) if len(gt_transform) != 0 else np.nan
    return H_st, H, H_st_matched, H_matched, [S_A_matched, A_counter], [S_O_matched, O_counter]

def evaluate_sequence(
    pred_rle_collapsed: dict[int, dict[int, RLE]],
    pred_rle: dict[int, dict[int, RLE]],
    gt_rle: dict[int, dict[int, RLE]],
    obj_id: int,
    frame_indices: list[int],
    tas_anno: dict[str, Any] | None,
    pred_transform: list,
    pred_obj_name: dict[int, str],
    semantic_eval: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate a single VOST-TAS sequence.

    J / J_tr use *pred_rle_collapsed* vs single-object GT sliced from *gt_rle*.
    *pred_rle* and full *gt_rle* are exposed for future metrics (no overlap
    collapse on predictions).

    *tas_anno* is the VOST-TAS annotation dict for this sequence (or None if
    the sequence has no VOST-TAS entry).
    """
    gt_single = utils.gt_rle_for_object(gt_rle, obj_id, frame_indices)
    J, J_tr = eval_J_scores(pred_rle_collapsed, gt_single, frame_indices)

    if tas_anno is not None:
        T_r, T_p, matched_gts, matched_pred, tp, fn, fp = eval_T_scores(tas_anno['transformations'], pred_transform)
        H_st, H, H_st_matched, H_matched, S_A_list, S_O_list = eval_H_scores(tas_anno['transformations'], pred_transform, pred_obj_name, pred_rle, matched_gts, matched_pred, semantic_eval=semantic_eval)
       
        S_A_matched, A_counter = S_A_list[0], S_A_list[1]
        S_O_matched, O_counter = S_O_list[0], S_O_list[1]
        S_A = S_A_matched / A_counter if A_counter != 0 else np.nan
        S_O = S_O_matched / O_counter if O_counter != 0 else np.nan
    else:
        T_r = T_p = H_st = H = S_A = S_O = np.nan
        H_st_matched = H_matched = 0
        tp = fn = fp = 0
        S_A_matched = A_counter = 0
        S_O_matched = O_counter = 0

    return {
        "J": J, "J_tr": J_tr,
        "T_rec": T_r, "T_pre": T_p,
        "H_st": H_st, "H": H,
        "S_O": S_O, 'S_A': S_A,
    }, {
        'TP': tp, 'FN': fn, 'FP': fp,
        'H_st_matched': H_st_matched,
        'H_matched': H_matched,
        'S_A_matched': S_A_matched, 'A_counter': A_counter,
        'S_O_matched': S_O_matched, 'O_counter': O_counter,
        'num_t': len(tas_anno['transformations']) if tas_anno is not None else 0,
    }


def main() -> None:
    T = utils.EVAL_TERM
    args, conf = get_args_and_conf()
    dataset_conf = conf.datasets[args.dataset]
    pred_dir = osp.join(osp.abspath(conf.paths.outdir), f"{args.dataset}-{args.split}")
    anno_dir: str = dataset_conf.anno_dir

    # VOST-TAS loading
    tas_annotations = utils.load_vost_tas_annotations(dataset_conf["anno_tas"])

    # VOST loading
    pred_jsons = sorted(glob(osp.join(pred_dir, "*.json")))
    video_groups = defaultdict(dict) # vost_video -> {obj_id: (seq_name, pred_json_path)}
    for pj in pred_jsons:
        seq = osp.splitext(osp.basename(pj))[0]
        vost_video, obj_id = utils.parse_vost_video_and_obj_id(seq, anno_dir)
        video_groups[vost_video][obj_id] = (seq, pj)

    print_banner(T, args, pred_dir, video_groups, tas_annotations)

    aggregate: dict[str, list[float]] = defaultdict(list)
    extra_agg = defaultdict(int)
    result_rows: list[tuple[str, dict[str, float]]] = []

    progress = tqdm(
        sorted(video_groups.keys()),
        desc="Evaluating",
        unit="video",
        dynamic_ncols=True,
        leave=True,
        colour="cyan",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for vost_video in progress:
        progress.set_postfix_str(
            vost_video[:28] + ("…" if len(vost_video) > 28 else ""), refresh=False,
        )

        objects = video_groups[vost_video]
        gt_rle, _ = utils.load_gt_rle(anno_dir, vost_video)
        num_frames = len(gt_rle)
        h, w = gt_rle[0][1]['size'] # Frame 0, obj_id 1
        img_size = (h, w)

        # Exclude first and last frames (TubeletGraph specific)
        frame_indices = list(range(1, num_frames - 1))

        # Collapse all per-object predictions into label-map arrays
        pred_labelmaps = utils.collapse_predictions_to_labelmap(objects, num_frames, img_size)

        # Evaluate each object
        for obj_id in sorted(objects):
            seq_name, pred_path = objects[obj_id]
            pred_rle_collapsed = utils.extract_object_mask_rle_from_collapsed(pred_labelmaps, obj_id, frame_indices)
            pred_rle, _, pred_data = utils.load_pred_rle(pred_path)
            pred_transform, pred_obj_name = utils.build_pred_transform(pred_data)
            tas_anno = tas_annotations.get(seq_name, None) # We don't have annotations for all videos

            if tas_anno is not None and tas_anno['anno_end'] != -1:
                pred_transform = [p for p in pred_transform if p['frameIdx'] <= tas_anno['anno_end']]

            metrics, extra_info = evaluate_sequence(
                pred_rle_collapsed=pred_rle_collapsed,
                pred_rle=pred_rle,
                gt_rle=gt_rle,
                obj_id=obj_id,
                frame_indices=frame_indices,
                tas_anno=tas_anno,
                pred_transform=pred_transform,
                pred_obj_name=pred_obj_name,
                semantic_eval=conf.eval_semantic,
            )
            
            for metric_name, value in metrics.items():
                if metric_name == "H" or metric_name == "H_st": continue # Skip because we don't want the mean
                aggregate[metric_name].append(float(value))
            
            # Add after aggregation to avoid duplication on multiple objects
            time_s = pred_data.get('tracking_elapsed_seconds', np.nan)
            metrics['Time (s)'] = time_s
            if len(result_rows) > 0 and time_s != result_rows[-1][1]['Time (s)']: # Same exact time -> timing is shared -> do not add
                aggregate['Time (s)'].append(time_s)

            result_rows.append((seq_name, metrics))
            for extra_name, value in extra_info.items():
                if extra_name in ['H_st_matched', 'H_matched','num_t', 'TP', 'FN', 'FP', 'S_A_matched', 'S_O_matched', 'A_counter', 'O_counter']:
                    extra_agg[extra_name] += value

    progress.close()

    
    # --- Reporting ---
    if len(aggregate) == 0:
        print(f"\n{T.YELLOW}No sequences were evaluated (check predictions and GT paths).{T.RESET}\n")
        return

    mean_metrics = {name: float(np.nanmean(values)) for name, values in aggregate.items()}
    # Final H_st, H, T_pre, T_rec aggregate
    mean_metrics['H_st'] = extra_agg['H_st_matched'] / extra_agg['num_t']
    mean_metrics['H'] = extra_agg['H_matched'] / extra_agg['num_t']
    mean_metrics['T_pre'] =  extra_agg['TP'] / (extra_agg['TP'] + extra_agg['FP']) if (extra_agg['TP'] + extra_agg['FP']) > 0 else 1.
    mean_metrics['T_rec'] = extra_agg['TP'] / (extra_agg['TP'] + extra_agg['FN']) if (extra_agg['TP'] + extra_agg['FN']) > 0 else 1.
    mean_metrics['S_A'] = extra_agg['S_A_matched'] / extra_agg['A_counter'] if extra_agg['A_counter'] != 0 else 0.
    mean_metrics['S_O'] = extra_agg['S_O_matched'] / extra_agg['O_counter'] if extra_agg['A_counter'] != 0 else 0.
    mean_metrics['Time (s)'] = sum(aggregate['Time (s)']) # We use the total time

    eval_dir = osp.join(pred_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Per-sequence reports
    results_md_path = osp.join(eval_dir, "results.md")
    md_text = utils.build_eval_markdown_report(
        result_rows,
        mean_metrics,
        dataset=args.dataset,
        split=args.split,
        pred_dir=pred_dir,
    )
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    results_csv_path = osp.join(eval_dir, "results.csv")
    csv_text = utils.build_eval_csv_report(result_rows, mean_metrics)
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    report_paths = [results_md_path, results_csv_path]

    # Optional copy of the csv
    if args.csv_out is not None:
        csv_out = osp.abspath(args.csv_out)
        if osp.isdir(csv_out):
            csv_out = osp.join(csv_out, "results.csv")
        os.makedirs(osp.dirname(csv_out), exist_ok=True)
        shutil.copyfile(results_csv_path, csv_out)
        report_paths.append(csv_out)

    utils.print_eval_banner("Results (VOST-TAS)")
    utils.print_eval_results_table(result_rows, mean_metrics)
    print(f"{T.DIM}Reports written to{T.RESET}")
    for path in report_paths:
        print(f"  {T.BOLD}{path}{T.RESET}")
    print()


if __name__ == "__main__":
    main()
