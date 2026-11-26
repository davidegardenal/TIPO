import argparse
import os
import os.path as osp
import shutil
from argparse import Namespace
from collections import defaultdict
from typing import Any

import numpy as np
from omegaconf import OmegaConf, open_dict
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import opts
import utils
from utils import RLE

IOU_TH = 0.5  # IoU threshold for matching objects in frames (used when computing transformations matching cost matrix)

def safe_lin_sum_assignment(cost_matrix, penalty=1e12):
    """
    An in-place wrapper for scipy.optimize.linear_sum_assignment that safely 
    handles +inf values by overwriting them with a large penalty, filtering 
    them out, and returning only the valid assignments.
    
    WARNING: This will modify the input `cost_matrix` if it contains +inf.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    inf_mask = np.isinf(cost_matrix)
    cost_matrix[inf_mask] = penalty
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    valid_rows = []
    valid_cols = []
    for r, c in zip(row_ind, col_ind):
        if not inf_mask[r, c]:  # Check against our saved mask
            valid_rows.append(r)
            valid_cols.append(c)
            
    return np.array(valid_rows, dtype=int), np.array(valid_cols, dtype=int)


def get_args_and_conf() -> tuple[Namespace, Any]:
    args, overrides = get_parser().parse_known_args()
    conf = opts.load_conf(args.config, args.mllm_config, overrides) # + CLI overrides

    with open_dict(conf): # eval_semantic is not part of the config files
        if args.mllm_config is not None:
            conf.paths.outdir = osp.join(conf.paths.outdir, conf.SEMANTIC_MLLM.out_path)
            conf.eval_semantic = True
        else:
            conf.eval_semantic = False

    print(OmegaConf.to_yaml(conf))
    return args, conf

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Skeleton evaluator: load GT/prediction masks and transformations for "
            "VISTA-style data, then print dummy per-sequence reports."
        )
    )
    p.add_argument("-c", "--config", type=str, help="Path to the config file", required=True)
    p.add_argument("-m", "--mllm-config", type=str, required=False, dest="mllm_config",
                    help="Path to the semantic MLLM config")
    p.add_argument("-d", "--dataset", type=str, help="Name of the dataset (found in the config)", required=True)
    p.add_argument("-s", "--split", type=str, default="val", help="ImageSets split (val/test/train)")
    p.add_argument("--csv_out", type=str, default=None,
                    help="Also copy results.csv to this path (if it is an existing directory, saved as <csv_out>/results.csv)")
    return p

def eval_J_scores(
    pred_rle: dict[int, dict[int, RLE]],
    gt_rle: dict[int, dict[int, RLE]],
    frame_indices: list[int],
) -> tuple[float, float]:
    """
        Evaluate the J scores (J and J_tr).
        This is computed by calculating the average IoU for each frame without considering the object ids.
        The objective of this metric is to evaluate the total tracking quality and precision.
    """
    ious = np.empty(len(frame_indices))
    for i, frame_idx in enumerate(frame_indices):
        pred_masks = list(pred_rle.get(frame_idx, {}).values())
        gt_masks = list(gt_rle.get(frame_idx, {}).values())

        if not pred_masks and not gt_masks:
            ious[i] = float('nan')
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

def eval_R_score(
    pred_rle: dict[int, dict[int, RLE]],
    gt_rle: dict[int, dict[int, RLE]],
    gt_obj_ids: list[int],
    frame_indices: list[int],
) -> tuple[dict[int, float], float]:
    """
        Evaluate the R score.
        This is computed by calculating the area recalled by the prediction to all the single objects in the video, then the score is the average per-object.
        The objective is to understand how much the objects are recalled by the tracker without being affected by total area bias.
    """
    obj_ids = [oid for oid in gt_obj_ids if oid != 0]
    recalls: dict[int, float] = {}

    for obj_id in obj_ids:
        frame_recalls = np.empty(len(frame_indices))
        for i, frame_idx in enumerate(frame_indices):
            gt_obj = gt_rle.get(frame_idx, {}).get(obj_id)
            if gt_obj is None:
                frame_recalls[i] = float('nan')
                continue
            gt_area = float(mask_utils.area(gt_obj))
            if gt_area == 0:
                frame_recalls[i] = float('nan')
                continue

            pred_masks = list(pred_rle.get(frame_idx, {}).values())
            if not pred_masks:
                frame_recalls[i] = 0.0
                continue

            pred_merged = mask_utils.merge(pred_masks, intersect=False)
            inter_rle = mask_utils.merge([pred_merged, gt_obj], intersect=True)
            frame_recalls[i] = float(mask_utils.area(inter_rle)) / gt_area

        recalls[obj_id] = 0. if np.isnan(frame_recalls).all() else float(np.nanmean(frame_recalls))

    mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0
    return recalls, mean_recall

def get_t_cost_matching_matrix(
    pred_transforms: list,
    gt_transforms: list,
    pred_rle: dict[int, dict[int, RLE]],
    gt_rle: dict[int, dict[int, RLE]],
    img_size: list[int],
):
    """Get the cost matrix to perform bipartite matching between transformations."""
    empty = utils._empty_rle(img_size)

    def _get_iou_matrix(pred_parts, gt_parts, frame_idx, iou_th):
        pred_frame = pred_rle.get(frame_idx, {})
        gt_frame = gt_rle.get(frame_idx, {})
        p_rles = [pred_frame.get(pid, empty) for pid in pred_parts]
        g_rles = [gt_frame.get(gpid, empty) for gpid in gt_parts]
        iou_mat = np.asarray(mask_utils.iou(p_rles, g_rles, [0] * len(g_rles)))
        return (iou_mat <= iou_th).astype(np.float64)  # 0 = valid match, 1 = no match (cost)

    def _is_ancestor(p, f, start_frame):
        # Assume that transf is frame ordered, return True if 'f' is ancestor of 'p'
        anc_list = [f]
        for t in pred_transforms:
            if t['frameIdx'] > start_frame: continue
            if p in anc_list: return True
            new_anc_list = []
            for a in anc_list:
                if a in t['afterObjects']: new_anc_list.extend(t['beforeObjects'])
                else: new_anc_list.append(a)
            anc_list = new_anc_list
        return False

    def _get_ancestor_matrix(obj_past, obj_future, start_frame):
        anc_matrix = np.zeros((len(obj_past), len(obj_future)))
        for f_idx, obj_f in enumerate(obj_future):
            for p_idx, obj_p in enumerate(obj_past):
                anc_matrix[p_idx][f_idx] = _is_ancestor(obj_p, obj_f, start_frame)
        return anc_matrix

    def _struct_check(gt_trans, pred_trans) -> tuple[bool, list]:
        if len(gt_trans['beforeObjects']) != len(pred_trans['beforeObjects']) or len(gt_trans['afterObjects']) != len(pred_trans['afterObjects']): return False, []

        if pred_trans['frameIdx'] <= gt_trans['frameIdx']:
            iou_matrix_parts = _get_iou_matrix(pred_trans['beforeObjects'], gt_trans['beforeObjects'], frame_idx=pred_trans['frameIdx'] - 1, iou_th=IOU_TH)
            pred_objs_future = [oid for oid, rle in pred_rle.get(pred_trans['frameIdx'], {}).items() if mask_utils.area(rle) > 0]
            if len(pred_objs_future) < len(gt_trans['afterObjects']): return False, []
            iou_matrix_next = _get_iou_matrix(pred_objs_future, gt_trans['afterObjects'], frame_idx=gt_trans['frameIdx'], iou_th=IOU_TH)
            anc_mat = _get_ancestor_matrix(pred_trans['afterObjects'], pred_objs_future, start_frame=gt_trans['frameIdx'])
            for i in range(iou_matrix_next.shape[0]):
                if anc_mat[:, i].sum() == 0: iou_matrix_next[i, :] = 1
        else:
            iou_matrix_next = _get_iou_matrix(pred_trans['afterObjects'], gt_trans['afterObjects'], frame_idx=pred_trans['frameIdx'], iou_th=IOU_TH)
            pred_objs_past = [oid for oid, rle in pred_rle.get(gt_trans['frameIdx'] - 1, {}).items() if mask_utils.area(rle) > 0]
            if len(pred_objs_past) < len(gt_trans['beforeObjects']): return False, []
            iou_matrix_parts = _get_iou_matrix(pred_objs_past, gt_trans['beforeObjects'], frame_idx=gt_trans['frameIdx'] - 1, iou_th=IOU_TH)
            anc_mat = _get_ancestor_matrix(pred_objs_past, pred_trans['beforeObjects'], start_frame=pred_trans['frameIdx'])
            for i in range(iou_matrix_parts.shape[0]):
                if anc_mat[i, :].sum() == 0: iou_matrix_parts[i, :] = 1

        p_row, p_col = safe_lin_sum_assignment(iou_matrix_parts)
        n_row, n_col = safe_lin_sum_assignment(iou_matrix_next)
        return iou_matrix_parts[p_row, p_col].sum() == 0 and iou_matrix_next[n_row, n_col].sum() == 0, [p_row, p_col, n_row, n_col, anc_mat, pred_trans['frameIdx'] <= gt_trans['frameIdx']]


    # - Time-window check
    ##  A predicted transformation can match only to the gt transformations temporally right before or after. The cost for valid transformation is the minimum distance in frames
    # - Transformation type check (assembly cannot match with disassembly and vice-versa)
    # - Transformation structure check -> struct_check
    ##  We check if the transformation has the same structure (same number of objects in and out). Moreover we check if the objects have a match with IoU > 0.5 between prediction and GT
    cost_mat = np.zeros((len(pred_transforms), len(gt_transforms)))
    obj_matching = [[[] for _ in range(len(gt_transforms))] for _ in range(len(pred_transforms))] # No append, just init
    for i, gt_t in enumerate(gt_transforms):
        gt_asmbl = len(gt_t['beforeObjects']) > 1
        for j, pred_t in enumerate(pred_transforms):
            pred_asml = len(pred_t['beforeObjects']) > 1
            inside_window = pred_t['frameIdx'] >= gt_t['frameIdx'] and (i == len(gt_transforms)-1 or pred_t['frameIdx'] <= gt_transforms[i+1]['frameIdx']) or \
                            pred_t['frameIdx'] <= gt_t['frameIdx'] and (i == 0 or pred_t['frameIdx'] >= gt_transforms[i-1]['frameIdx'])

            cost_mat[j][i] = abs(gt_t['frameIdx'] - pred_t['frameIdx'])
            cost_mat[j][i] = cost_mat[j][i] if gt_asmbl == pred_asml else np.inf
            cost_mat[j][i] = cost_mat[j][i] if inside_window else np.inf
            
            if not np.isinf(cost_mat[j][i]):
                struct_ok, obj_mats = _struct_check(gt_t, pred_t)
                cost_mat[j][i] = cost_mat[j][i] if struct_ok else np.inf
                obj_matching[j][i] = obj_mats

    return cost_mat, obj_matching

def eval_S_score(t_row, t_col, pred_transforms, gt_transforms, obj_matching, pred_obj_names, gt_obj_names):
    act_scores = []
    obj_scores = []
    complete_match = []
    for p_idx, gt_idx in zip(t_row, t_col):
        act_flag, _ = utils.judge_sem_acc(pred_transforms[p_idx]['description'], gt_transforms[gt_idx]['description'], is_action=True)
        
        p_row, p_col = obj_matching[p_idx][gt_idx][0], obj_matching[p_idx][gt_idx][1]
        n_row, n_col = obj_matching[p_idx][gt_idx][2], obj_matching[p_idx][gt_idx][3]

        # Map the ancestors objects that were matched previously
        anc_mat = obj_matching[p_idx][gt_idx][-2]
        if obj_matching[p_idx][gt_idx][-1]: # n_row are from future objects
            n_row = [anc_mat[:, nr].argmax() for nr in n_row]
        else: # p_row are from past objects
            p_row = [anc_mat[pr, :].argmax() for pr in p_row]

        # Object match
        obj_resp = []
        for p_obj_idx, gt_obj_idx in zip(p_row, p_col):
            obj_resp.append(utils.judge_sem_acc(pred_obj_names[pred_transforms[p_idx]['beforeObjects'][p_obj_idx]], gt_obj_names[gt_transforms[gt_idx]['beforeObjects'][gt_obj_idx]], is_action=False)[0])
        
        for p_obj_idx, gt_obj_idx in zip(n_row, n_col):
            obj_resp.append(utils.judge_sem_acc(pred_obj_names[pred_transforms[p_idx]['afterObjects'][p_obj_idx]], gt_obj_names[gt_transforms[gt_idx]['afterObjects'][gt_obj_idx]], is_action=False)[0])

        # Update scores
        act_scores.append(act_flag)
        obj_scores.extend(obj_resp)
        complete_match.append(all(obj_resp) and act_flag)

    S_A = sum(act_scores) / len(act_scores) if len(act_scores) != 0 else np.nan
    S_O = sum(obj_scores) / len(obj_scores) if len(obj_scores) != 0 else np.nan
    return S_A, S_O, sum(complete_match)


def evaluate_sequence(
    pred_rle: dict[int, dict[int, RLE]],
    pred_transforms: list,
    pred_obj_names: dict[int, str],
    gt_rle: dict[int, dict[int, RLE]],
    gt_transforms: list,
    gt_obj_names: dict[int, str],
    frame_indices: list[int],
    img_size: list[int],
    fps: float,
    semantic_eval: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:

    # Transformation matching
    C_matrix, obj_matching = get_t_cost_matching_matrix(pred_transforms, gt_transforms, pred_rle, gt_rle, img_size)
    t_row, t_col = safe_lin_sum_assignment(C_matrix)
    tp = t_row.size
    fp = C_matrix.shape[0] - tp
    fn = C_matrix.shape[1] - t_col.size
    
    # Semantic matching
    if semantic_eval:
        S_A, S_O, tp_complete_sem = eval_S_score(t_row, t_col, pred_transforms, gt_transforms, obj_matching, pred_obj_names, gt_obj_names)
        S = 2 / (1 / S_A + 1 / S_O) if S_A != 0 and S_O != 0 else 0.
        H_F1 = (2 * tp_complete_sem) / (2 * tp_complete_sem + fp + fn) if tp_complete_sem + fp + fn != 0 else 0.
    else:
        S_A = S_O = S = H_F1 = np.nan

    # Metrics
    J, J_tr = eval_J_scores(pred_rle, gt_rle, frame_indices)
    R_per_obj, R = eval_R_score(pred_rle, gt_rle, list(gt_obj_names.keys()), frame_indices)
    T_rec = 0. if (tp + fn) == 0 else tp / (tp + fn)
    T_pre = 0. if (tp + fp) == 0 else tp / (tp + fp)
    F_1t = (2 * tp) / (2 * tp + fp + fn)
    T_lag = float(np.nan) if C_matrix[t_row, t_col].size == 0 or np.isnan(C_matrix[t_row, t_col]).all() else float(C_matrix[t_row, t_col].mean())
    T_lag = T_lag / fps # convert to seconds

    return {
        "J": J, "J_tr": J_tr,
        "R": R,
        "T_rec": T_rec, "T_pre": T_pre, "F_1t": F_1t,
        "S_A": S_A, "S_O": S_O, "S": S,
        "H_F1": H_F1,
        "T_lag": T_lag,
    }, \
    { # extra results in case we want some global stats
        "R_per_obj": R_per_obj, 
        "TP": tp, "FP": fp, "FN": fn
    }


def main() -> None:
    T = utils.EVAL_TERM
    args, conf = get_args_and_conf()
    dataset_conf = conf.datasets[args.dataset]
    pred_dir = osp.join(osp.abspath(conf.paths.outdir), f"{args.dataset}-{args.split}")

    sequences = utils.read_sequences(dataset_conf.split_dir, args.split)
    aggregate: dict[str, list[float]] = defaultdict(list)
    result_rows: list[tuple[str, dict[str, float]]] = []

    utils.print_eval_banner("TIPO evaluation")
    print(f"{T.DIM}dataset{T.RESET}  {T.BOLD}{args.dataset}{T.RESET}   "
          f"{T.DIM}split{T.RESET}  {T.BOLD}{args.split}{T.RESET}")
    print(f"{T.DIM}pred_dir{T.RESET}  {pred_dir}")
    print(f"{T.DIM}clips{T.RESET}     {len(sequences)}\n")

    progress = tqdm(
        sequences,
        desc="Evaluating",
        unit="clip",
        dynamic_ncols=True,
        leave=True,
        colour="cyan",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for seq in progress:
        progress.set_postfix_str(seq[:28] + ("…" if len(seq) > 28 else ""), refresh=False)

        pred_candidates = utils.list_sequence_prediction_candidates(pred_dir, seq)
        if len(pred_candidates) == 0: 
            tqdm.write(f"{T.YELLOW}[Skip]{T.RESET} `{seq}` — no prediction JSON in `{pred_dir}`")
            continue

        pred_json_path = pred_candidates[0] # Take the first sorted
        if len(pred_candidates) > 1:
            tqdm.write(
                f"{T.DIM}[Info]{T.RESET} `{seq}` — {len(pred_candidates)} JSON files; "
                f"using `{osp.basename(pred_json_path)}`"
            )
        
        # Loading GT + predictions as RLE
        gt_rle, _anno_paths = utils.load_gt_rle(dataset_conf.anno_dir, seq)
        pred_rle, frame_indices, pred_data = utils.load_pred_rle(pred_json_path)
        pred_transform, pred_obj_name = utils.build_pred_transform(pred_data)
        gt_transform, gt_obj_names = utils.build_gt_transform(dataset_conf.trans_dir, seq)
        metadata = utils.load_metadata(dataset_conf.data_dir, seq)

        # Determine image size from the first available RLE
        first_rle = next(iter(next(iter(pred_rle.values())).values()))
        img_size = first_rle['size']

        # EVAL
        metrics, _extra_results = evaluate_sequence(
            pred_rle=pred_rle,
            pred_transforms=pred_transform,
            pred_obj_names=pred_obj_name,
            gt_rle=gt_rle,
            gt_transforms=gt_transform,
            gt_obj_names=gt_obj_names,
            frame_indices=frame_indices,
            img_size=img_size,
            fps=metadata['fps'],
            semantic_eval=conf.eval_semantic,
        )

        metrics['Time (s)'] = pred_data.get('tracking_elapsed_seconds', np.nan)
        for metric_name, value in metrics.items():
            aggregate[metric_name].append(float(value))
        result_rows.append((seq, metrics))

    progress.close()

    # LOGGING
    if len(aggregate) == 0:
        print(f"\n{T.YELLOW}No sequences were evaluated (check predictions and GT paths).{T.RESET}\n")
        return

    mean_metrics = {name: float(np.nanmean(values)) for name, values in aggregate.items()}
    mean_metrics['Time (s)'] = sum(aggregate['Time (s)']) # We use the total time

    eval_dir = osp.join(pred_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
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

    utils.print_eval_banner("Results")
    utils.print_eval_results_table(result_rows, mean_metrics)
    print(f"{T.DIM}Reports written to{T.RESET}")
    for path in report_paths:
        print(f"  {T.BOLD}{path}{T.RESET}")
    print()


if __name__ == "__main__":
    main()
