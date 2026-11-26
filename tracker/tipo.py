import functools
import math

# CUDA optimized CC labeling
import cupy as cp
import numpy as np
import torch
from cucim.skimage.measure import label
from more_itertools import set_partitions
from omegaconf import OmegaConf

from state_graph import Object, State, StateGraph, Transformation
from tracker.mllm_judge import MLLMJudge
from tracker.mllm_labeler import MLLMLabeler
from tracker.sam2_tracker import SAM2
from tracker.sam3_tracker import SAM3
from utils import inter_over_a

# Defaults
NEXT_OBJECT_ID = 0

TRANSFORMATIONS_ENABLED = True
K_SAMPLES = 8                      # Number of masked sampled per component
MIN_IOU = 0.8                      # SAM IoU threshold for sampled masks ensemble
STATE_PROB_TH = 0.                # Threshold for the probability of a disassembly to be considered
STATE_ASMBL_PROB_TH = 0.          # Threshold for the probability of an assembly to be considered
ASMBL_IOA_TH = 0.05
TRANSF_TH_DIS = 0.8                # Threshold for the probability of a state to trigger a disassembly
TRANSF_TH_ASMBL = 0.8              # Threshold for the probability of a state to trigger an assembly
MAX_OBJECTS = 7

SAM = SAM3

def set_global_conf(conf: OmegaConf) -> None:
    global K_SAMPLES, TRANSF_TH_DIS, TRANSF_TH_ASMBL, MIN_IOU, SAM, NEXT_OBJECT_ID, TRANSFORMATIONS_ENABLED, STATE_PROB_TH, STATE_ASMBL_PROB_TH
    K_SAMPLES = conf.SAMPLING_K # pyrefly: ignore
    MIN_IOU = conf.SAMPLING_MIN_IOU  # pyrefly: ignore
    TRANSF_TH_DIS = conf.TRANSF_TH_DIS  # pyrefly: ignore
    TRANSF_TH_ASMBL = conf.TRANSF_TH_ASMBL # pyrefly: ignore
    TRANSFORMATIONS_ENABLED = TRANSF_TH_DIS <= 1. and TRANSF_TH_ASMBL <= 1. # Disable the transformations if the THs are impossible to satisfy

    # These THs are used but they don't influence the results (just the speed)
    STATE_PROB_TH = TRANSF_TH_DIS
    STATE_ASMBL_PROB_TH = TRANSF_TH_ASMBL

    if conf.SAM_VERSION == 3: SAM = SAM3 # pyrefly: ignore
    else: SAM = SAM2
    NEXT_OBJECT_ID = 0

def get_connected_components(mask: torch.Tensor, min_size: float = 2e-4) -> torch.Tensor:
    labeled_mask = label(cp.asarray(mask), background=0)
    labeled_mask = torch.as_tensor(labeled_mask, device=mask.device)

    labels = labeled_mask.unique()
    for l in labels:
        l_mask = labeled_mask == l
        if l_mask.sum() < mask.shape[-1] * mask.shape[-2] * min_size:
            # Delete the component
            labeled_mask[labeled_mask == l] = 0

    return labeled_mask

def eval_disassembly(state: State, prob_th: float = 0.):

    @functools.cache
    def recall_component_set(comps: tuple[int], obj_id: int) -> list[float]:
        mask = state.objects[obj_id].mask
        assert isinstance(mask, torch.Tensor)

        comp_mask = torch.isin(mask, torch.tensor(comps, device=mask.device))
        sampled_mask = torch.zeros_like(mask, device=mask.device)
        inner_recall: list[float] = []
        outer_recall: list[float] = []
        for c_id in comps:
            mask_c_id = mask == c_id
            sampled_mask_c_id = state.objects[obj_id].get_sampled_mask(c_id)
            sampled_mask = sampled_mask | sampled_mask_c_id
            # Inner
            num_inner = mask_c_id & sampled_mask_c_id
            inner_recall.append(float(num_inner.sum() / mask_c_id.sum()))
            # Outer
            if len(comps) > 1:
                not_mask_c_id = (comp_mask & (~mask_c_id))
                num_outer = not_mask_c_id & (sampled_mask_c_id & (~mask_c_id))
                outer_recall.append(float(num_outer.sum() / not_mask_c_id.sum()))
            else: outer_recall.append(1.)

        # Extern
        if len(state.objects[obj_id].component_ids) != len(comps):
            ext_mask = (mask != 0) & (~comp_mask)
            extern_recall = float((ext_mask & sampled_mask).sum() / ext_mask.sum())
        else: extern_recall = 0.

        return [(sum([inn * out for inn, out in zip(inner_recall, outer_recall)]) / len(inner_recall)) * (1. - extern_recall)]

    def eval_partition(part: list[list[int]], obj_id: int) -> float:
        """ 
        Args:
            part:   The list of component ids that identify the disassembly
            obj_id: The id of the object under consideration

        Returns:
            Probability of the given partition to reflect the real state of the object
        """
        prob_tmp = []
        for cset in part:
            prob_tmp.extend(recall_component_set(tuple(cset), obj_id))
        
        return math.prod(prob_tmp)
    
    for obj_id, obj in state.objects.items():
        states_prob = []
        if len(obj.component_ids) == 1 or len(obj.component_ids) > MAX_OBJECTS: # No possible disassembly OR we have a degenerate mask (adds a bound to the compute)
            state.objects[obj_id].hypo_states = [[obj.component_ids]]
            states_prob.append(1.)
        else:
            state.objects[obj_id].hypo_states = []
            for part in set_partitions(obj.component_ids): # For each possible disassembly
                likelihood_part = eval_partition(part, obj_id)
                if likelihood_part > prob_th: 
                    state.objects[obj_id].hypo_states.append(part)
                    states_prob.append(likelihood_part)
        state.objects[obj_id].states_prob = np.array(states_prob)
    return state

def eval_assembly(state: State, ioa_th: float = 0.1, prob_th: float = 0.05):
    # Check the overlap between all possible pair of objects and create a set of objects overlapping
    # NOTE: this can be probably batched but I'm lazy :)
    obj_overlapping = set()
    obj_id_to_idx = {}
    obj_idx_to_id = {}
    overlap_mat = torch.zeros((len(state.objects), len(state.objects)))
    for idx_i, obj_i in enumerate(state.objects.keys()):
        obj_id_to_idx[obj_i] = idx_i
        obj_idx_to_id[idx_i] = obj_i
        for idx_j, obj_j in enumerate(state.objects.keys()):
            ioa_ij = 1.
            if idx_i != idx_j:
                ioa_ij = inter_over_a(state.objects[obj_i].sampled_masks.sum(dim=0) > 0, state.objects[obj_j].sampled_masks.sum(dim=0) > 0)
                if ioa_ij > ioa_th:
                    obj_overlapping.add(obj_i)
                    obj_overlapping.add(obj_j)

            overlap_mat[idx_i][idx_j] = ioa_ij
    
    # Helper functions
    @functools.cache
    def recall_object_set(comps: tuple[int]) -> float:
        # Sampled objects overlap
        comps_idx = [obj_id_to_idx[c_id] for c_id in comps]
        comp_ioa_mat = overlap_mat[comps_idx][:, comps_idx]

        # Extern
        if len(obj_overlapping) != len(comps):
            sampled_mask = torch.zeros_like(state.objects[comps[0]].mask, device=state.objects[comps[0]].mask.device) # pyrefly: ignore
            for c_id in comps:
                sampled_mask_obj = state.objects[c_id].sampled_masks.sum(dim=0) > 0 # Sampled mask of the current object
                sampled_mask = sampled_mask | sampled_mask_obj # Incremental mask to get the complete *sampled* mask of the component
            ext_mask_l = []
            for obj_id in obj_overlapping:
                if obj_id not in comps:
                    ext_mask_l.append(state.objects[obj_id].sampled_masks)
            ext_mask = torch.cat(ext_mask_l, dim=0).sum(dim=0) > 0  # Sampled mask of all the others object outside the component
            extern_recall = float((ext_mask & sampled_mask).sum() / ext_mask.sum())
        else: extern_recall = 0.

        return float((comp_ioa_mat.prod(dim=0).mean(dim=0)) * (1. - extern_recall))
    # NOTE: theoretically this can be batched if we create an index for the partitions (but I'm not sure)
    def eval_partition(part: list[list[int]]):
        prob_tmp = []
        for cset in part:
            prob_tmp.append(recall_object_set(tuple(cset)))
        return math.prod(prob_tmp)
    # -------------------------------------------------------
    
    if len(obj_overlapping) > MAX_OBJECTS: # we have a degenerate case (adds a bound to the compute)
        # instead of skipping we consider the top 7 objects with highest maximum overlap
        # Note that since we don't use a symmetric overlap function we could be deleting valid assemblies in some extreme cases
        l_obj_idx_overlapping = torch.tensor([obj_id_to_idx[o] for o in obj_overlapping])
        top_7_objs_idx = torch.topk(overlap_mat[l_obj_idx_overlapping].max(dim=1).values, k=7).indices
        obj_overlapping = {obj_idx_to_id[o_i] for o_i in l_obj_idx_overlapping[top_7_objs_idx].numpy().tolist()}
    
    if len(obj_overlapping) != 0:
        state.hypo_asmbl_states = []
        asmbl_p = []
        left_out_objs = [[lo] for lo in (set(state.objects.keys()) - obj_overlapping)]
        # Possible assembly so we create hypothesis states and evaluate them
        for part in set_partitions(obj_overlapping):
            likelihood_part = eval_partition(part)
            if likelihood_part > prob_th: 
                state.hypo_asmbl_states.append(part + left_out_objs)
                asmbl_p.append(likelihood_part)
        state.asmbl_states_prob = np.array(asmbl_p)
    else: # No possible assembly
        state.hypo_asmbl_states = [[[ob] for ob in state.objects]]
        state.asmbl_states_prob = np.array([1.,])

    return state

def get_compatible_assembly(state: State, past_to_new_obj: dict) -> int:
    def is_asmbl_valid(indx: int) -> bool: # check that the objects in the partitions with len > 1 are not in past_to_new_obj keys
        for asm in state.hypo_asmbl_states[indx]:
            if len(asm) > 1 and any(o in past_to_new_obj for o in asm): return False
        return True

    # Visit the states from the most to the least probable (stable: ties keep the lowest index, like argmax)
    for state_idx in np.argsort(-np.asarray(state.asmbl_states_prob), kind="stable"):
        if is_asmbl_valid(int(state_idx)): return int(state_idx)
    return -1

def get_next_state_w_trans(frame_idx:int, state: State, tracker: SAM, mllm: MLLMJudge | None, transf_th_dis: float = 0.8, transf_th_asmbl: float = 0.5) -> tuple[State, list[Transformation]]:
    # The evaluation is composed by:
    #   - Object-wise disassembly evaluation
    #   - State-wise assembly evaluation
    # For disassembly: each object has a set of possible states with corresponding probability -> pick the one with highest prob about a specific threshold 
    # For assembly: pick the state with the highest probability that doesn't involve an object selected for disassembly

    global NEXT_OBJECT_ID
    next_state = State.create([])
    transf_list = []
    past_to_new_obj = {}
    if frame_idx != 0 and TRANSFORMATIONS_ENABLED: # The first frame is conditioned so no possible transformation
        # DISASSEMBLY
        for obj_id, obj in state.objects.items():
            if len(obj.states_prob) != 0:
                top_state_idx = torch.tensor(obj.states_prob).argmax()
                if obj.states_prob[top_state_idx] > transf_th_dis and len(obj.hypo_states[top_state_idx]) > 1:
                    obj_masks: dict[tuple, torch.Tensor] = {}
                    for part in obj.hypo_states[top_state_idx]:
                        obj_masks[tuple(part)] = torch.isin(obj.mask, torch.tensor(part, device=obj.mask.device)) # Get the correct mask for the object # pyrefly: ignore
                    if mllm is None or (mllm is not None and mllm.is_disassembled(frame_idx, obj_masks)): # MLLM GATING
                        # Create new objects based on the new state for this object
                        new_objects_ids = []
                        for part in obj.hypo_states[top_state_idx]:
                            obj_m = obj_masks[tuple(part)]
                            next_state.objects[NEXT_OBJECT_ID] = Object.create(NEXT_OBJECT_ID, parent_ids=[obj.id], name="TODO", mask=obj_m)
                            new_objects_ids.append(NEXT_OBJECT_ID)
                            tracker.add_object(frame_idx, parent_obj_id=obj_id, obj_id=NEXT_OBJECT_ID, mask=obj_m[0])
                            NEXT_OBJECT_ID += 1

                        tracker.remove_object(obj_id)
                        transf_list.append(Transformation.create(parts={obj.id}, next_ids=set(new_objects_ids),description="TODO"))
                        past_to_new_obj[obj.id] = new_objects_ids

        # ASSEMBLY
        if len(state.asmbl_states_prob) != 0:
            top_state_idx = get_compatible_assembly(state, past_to_new_obj)
            if top_state_idx != -1 and state.asmbl_states_prob[top_state_idx] >= transf_th_asmbl and len(state.hypo_asmbl_states[top_state_idx]) < len(state.objects):
                for asm in state.hypo_asmbl_states[top_state_idx]:
                    if len(asm) > 1:
                        obj_masks: dict[int, torch.Tensor] = {o: state.objects[o].mask for o in asm} # Get cumulative mask of all the objects # pyrefly: ignore
                        if mllm is None or (mllm is not None and mllm.is_assembled(frame_idx, obj_masks)): # MLLM GATING
                            for o in asm:
                                tracker.remove_object(o) # Remove old objects from the tracking
                                past_to_new_obj[o] = [NEXT_OBJECT_ID]
                            
                            obj_mask = (torch.cat(list(obj_masks.values()), dim=0).sum(dim=0) > 0).unsqueeze(0)
                            next_state.objects[NEXT_OBJECT_ID] = Object.create(NEXT_OBJECT_ID, parent_ids=asm, name="TODO", mask=obj_mask)
                            # No need to add any conditioning of the parent objects
                            tracker.add_object(frame_idx, parent_obj_id=-1, obj_id=NEXT_OBJECT_ID, mask=obj_mask[0])
                            transf_list.append(Transformation.create(parts=set(asm), next_ids={NEXT_OBJECT_ID},description="TODO"))
                            NEXT_OBJECT_ID += 1

    # Add the `none` transforms for the objects left
    for obj_id, obj in state.objects.items():
        if obj_id not in past_to_new_obj:
            next_state.objects[obj_id] = obj
            transf_list.append(Transformation.create(parts={obj.id}, next_ids={obj.id},description="none"))
    # Copy the assembly information
    next_state.hypo_asmbl_states = state.hypo_asmbl_states
    next_state.asmbl_states_prob = state.asmbl_states_prob
    next_state.asmbl_belief = state.asmbl_belief

    return next_state, transf_list

def add_init_objects(tracker: SAM, init_masks: dict[int, torch.Tensor], frame_idx: int) -> State:
    global NEXT_OBJECT_ID
    init_ids = init_masks[frame_idx].unique()
    init_obj_list = []
    for init_id in init_ids:
        if int(init_id) == 0: continue # Background
        init_obj_list.append(Object.create(id=NEXT_OBJECT_ID, parent_ids=[-1], name="TODO", mask=init_masks[frame_idx] == init_id))
        tracker.add_object(frame_idx=frame_idx, parent_obj_id=-1, obj_id=NEXT_OBJECT_ID, mask=init_masks[frame_idx] == init_id)
        NEXT_OBJECT_ID += 1
    
    return State.create(init_obj_list)

def track_transformations(
    tracker: SAM,
    mllm: MLLMJudge | None,
    init_masks: dict[int, torch.Tensor],
    conf: OmegaConf,
    semantic_mllm: MLLMLabeler | None,
    fps: float | None = None,
) -> StateGraph:
    """Track the prompted objects and their transformations."""
    set_global_conf(conf)

    graph = StateGraph(tracker.video_path)
    graph.fps = fps
    graph.states[0] = add_init_objects(tracker, init_masks, frame_idx=0)
    
    # Tracking loop
    for frame_idx, obj_ids, obj_id_to_idx, masks in tracker.track_iter():

        curr_state = State.create()
        # Transformation eval
        for obj in obj_ids:
            m_idx = obj_id_to_idx[obj]
            if TRANSFORMATIONS_ENABLED: cc_mask = get_connected_components(masks[m_idx])
            else: cc_mask = masks[m_idx]

            # Take the info from the previous object
            f_idx = frame_idx - 1 if frame_idx > 0 else 0 # if it is the first frame then take for it directly
            prev_obj = graph.states[f_idx].objects.get(obj, Object.create(id=obj, parent_ids=[-1], name='TODO', mask=torch.empty(1,)))
            curr_state.objects[obj] = Object.create(id=prev_obj.id, parent_ids=prev_obj.parent_ids, name=prev_obj.name, mask=cc_mask)
            if TRANSFORMATIONS_ENABLED:
                if cc_mask.sum() > 0: # if the object is not fully occluded
                    curr_state.objects[obj].sampled_masks = tracker.sample_sam_point_mask(frame_idx, cc_mask[0], curr_state.objects[obj].component_ids, K_samples=K_SAMPLES, min_IoU=MIN_IOU) # pyrefly: ignore
                else:
                    curr_state.objects[obj].sampled_masks = torch.zeros_like(cc_mask, device=cc_mask.device)

        if TRANSFORMATIONS_ENABLED:
            curr_state = eval_disassembly(curr_state, prob_th=STATE_PROB_TH)
            curr_state = eval_assembly(curr_state, ioa_th=ASMBL_IOA_TH, prob_th=STATE_ASMBL_PROB_TH)

        # Make a decision based on the estimated prob
        next_state, transformations = get_next_state_w_trans(frame_idx, curr_state, tracker, mllm, transf_th_dis=TRANSF_TH_DIS, transf_th_asmbl=TRANSF_TH_ASMBL)
        next_state.to('cpu')

        if frame_idx in init_masks and frame_idx != 0: # Add the new init objects
            next_init_state = add_init_objects(tracker, init_masks, frame_idx=frame_idx)
            next_state.objects.update(next_init_state.objects) # add the new init objects to the state

        next_state.clean()
        graph.states[frame_idx] = next_state
        graph.transformations[frame_idx] = transformations

        # Cleanup (without disrupting the memory encoding)
        tracker.del_old_output(frame_idx, delete_last_n=25)

    # Semantic labeling (optional)
    if semantic_mllm is not None:
        graph.fill_semantic_labels(semantic_mllm)

    return graph