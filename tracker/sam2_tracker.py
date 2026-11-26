import os
from collections import OrderedDict
from typing import Any

import sam2
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from utils import tqdm_disabled


class SAM2:
    def __init__(self, video_path: str|None, device: str|torch.device):

        sam2_root = os.path.dirname(os.path.dirname(sam2.__file__))
        model_weights = os.path.join(sam2_root, "checkpoints", "sam2.1_hiera_large.pt")
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, model_weights, device=device)
        assert not self.predictor.clear_non_cond_mem_around_input, "`clear_non_cond_mem_around_input` not supported"
        self.device = device

        self._shared_cached_features: dict = {}
        self.inference_states: dict[int, dict] = {} # start_frame_idx → inference_state dict
        self._obj_to_state_frame: dict[int, int] = {}
        if video_path is not None:
            self.init_session(video_path)
    
    def _create_inference_state(
        self,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    ):
        """Initialize an inference state."""
        compute_device = self.predictor.device 
        inference_state: dict[str, Any] = {}

        inference_state["images"] = self._video_meta['images']
        inference_state["num_frames"] = self._video_meta['num_frames']
        inference_state["video_height"] = self._video_meta['video_height']
        inference_state["video_width"] = self._video_meta['video_width']
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        inference_state["cached_features"] = self._shared_cached_features
        return inference_state

    def init_session(self, video_path: str, disable_tqdm: bool = True):
        self.video_path = video_path
        self.frame_names = sorted(os.listdir(video_path))
        with tqdm_disabled(active=disable_tqdm):
            self._session_template = self.predictor.init_state(
                video_path=video_path,
                offload_state_to_cpu=True,
                offload_video_to_cpu=True,
            )
        self._video_meta: dict[str, Any] = {
            'images': self._session_template['images'],
            'video_height': self._session_template['video_height'],
            'video_width': self._session_template['video_width'],
            'num_frames': self._session_template['num_frames'],
        }

        self._shared_cached_features: dict = {}
        self.inference_states: dict[int, dict] = {} # start_frame_idx → inference_state dict
        self._obj_to_state_frame: dict[int, int] = {}

    def __normalize_coords(self, coords, og_H, og_W, targ_H, targ_W):
        coords[:, 0] = coords[:, 0] * (targ_H / og_H)
        coords[:, 1] = coords[:, 1] * (targ_W / og_W)
        return coords

    def add_object(self, frame_idx: int, parent_obj_id: int, obj_id: int, mask: torch.Tensor):
        """Add an object to the tracker.

        If an inference state already exists for *frame_idx* the object is
        added there; otherwise a brand-new state is created.
        """
        if frame_idx not in self.inference_states:
            self.inference_states[frame_idx] = self._create_inference_state()

        state = self.inference_states[frame_idx]
        # Parent original mask
        if parent_obj_id != -1:
            parent_state_key = self._obj_to_state_frame[parent_obj_id]
            parent_state_idx = self.inference_states[parent_state_key]['obj_id_to_idx'][parent_obj_id]
            parent_out = self.inference_states[parent_state_key]['output_dict_per_obj'][parent_state_idx]
            parent_frame = max(parent_out['cond_frame_outputs'].keys())
            parent_mask = (parent_out['cond_frame_outputs'][parent_frame]['pred_masks'] > 0.)[0, 0]
            self.predictor.add_new_mask(inference_state=state, frame_idx=parent_frame, obj_id=obj_id, mask=parent_mask)
        # Current mask
        self.predictor.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )
        
        self._obj_to_state_frame[obj_id] = frame_idx

    def remove_object(self, obj_id: int):
        """Remove an object from the tracker.

        If the owning inference state becomes empty it is also removed.
        """
        frame = self._obj_to_state_frame.pop(obj_id)
        state = self.inference_states[frame]
        self.predictor.remove_object(state, obj_id=obj_id, need_output=False)

        # If no objects left in this state, drop it entirely
        if len(state['obj_ids']) == 0:
            del self.inference_states[frame]

    def del_old_output(self, frame_idx: int, delete_last_n: int = 5):
        """Remove old non-conditioning outputs to save memory."""
        oldest_allowed_idx = frame_idx - delete_last_n
        for s in self.inference_states.values():
            all_frame_idxs = list(s["output_dict_per_obj"][0]['non_cond_frame_outputs'].keys())
            old_frame_idxs = [idx for idx in all_frame_idxs if idx < oldest_allowed_idx]
            for old_idx in old_frame_idxs:
                for obj_id in s["output_dict_per_obj"]:
                    s["output_dict_per_obj"][obj_id]['non_cond_frame_outputs'].pop(old_idx, None)

    @torch.inference_mode()
    def sample_sam_point_mask(self, frame_idx: int, cc_mask: torch.Tensor, cc_ids: list[int], n_points: int = 1, K_samples: int = 4, min_IoU: float = 0.8) -> torch.Tensor:
        back_features = self._shared_cached_features[frame_idx][1]
        N_comp = len(cc_ids)
        video_H = self._video_meta["video_height"]
        video_W = self._video_meta["video_width"]
        back_H, back_W = self._video_meta['images'].shape[2:] # type: ignore
        
        # Sample points for each component
        point_coords = torch.zeros(N_comp, K_samples, n_points, 2, dtype=torch.float32, device=self.device)  # [B, S, P, 2]
        for i, l in enumerate(cc_ids):
            coords = torch.nonzero(cc_mask == l)
            rand_point = coords[torch.randint(low=0, high=coords.size(0), size=(n_points * K_samples,))].to(torch.float32)
            point_coords[i] = self.__normalize_coords(rand_point, video_H, video_W, back_H, back_W).reshape(K_samples, n_points, 2)
        point_coords = point_coords[..., [1, 0]]  # From (y, x) to (x, y) coordinates
        point_coords = point_coords.reshape(N_comp * K_samples, n_points, 2)  # [B * S, P, 2]
        point_inputs = {'point_coords': point_coords, 'point_labels': torch.ones(N_comp * K_samples, n_points, device=self.device, dtype=torch.int32)}

        # Prompt SAM with points to get the propagated mask
        sam2_out = self.predictor._forward_sam_heads(
            backbone_features=back_features['vision_features'].repeat(N_comp * K_samples, 1, 1, 1),
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=[fpn_f.repeat(N_comp * K_samples, 1, 1, 1) for fpn_f in back_features['backbone_fpn'][:2]],
            multimask_output=True,
        )
        point_masks, ious = sam2_out[1], sam2_out[2]
        point_masks = point_masks.reshape(N_comp, K_samples * 3, 1, *point_masks.shape[2:])
        ious = ious.reshape(N_comp, K_samples * 3)
        
        # Ensemble strong masks (or the best available if none is good)
        good_masks = ious >= min_IoU
        no_good_mask = good_masks.sum(dim=1) == 0
        max_iou_idx = torch.topk(ious, k=max(1, int(K_samples*3 / 8)), dim=1, sorted=False).indices # Take top-(k/8) masks if th is not met
        row_indices = torch.arange(len(no_good_mask), device=no_good_mask.device).unsqueeze(-1)
        rows_to_update = row_indices[no_good_mask]
        cols_to_update = max_iou_idx[no_good_mask]
        good_masks[rows_to_update, cols_to_update] = True
        
        max_iou_masks = ((point_masks > 0.) * good_masks.reshape(*good_masks.shape, 1, 1, 1)).sum(dim=1).to(torch.bfloat16)

        # Resize to match the size of the video
        video_res_masks = torch.nn.functional.interpolate(
            max_iou_masks,
            size=(video_H, video_W),
            mode="bilinear",
            align_corners=False,
        ) > 0.

        return video_res_masks

        
    
    @torch.inference_mode()
    def track_iter(
        self,
        start_frame_idx=0,
        tqdm_disable=False
    ):
        assert self.inference_states != {} 
    
        for frame_idx in tqdm(
            range(start_frame_idx, self._video_meta['num_frames']), 
             desc="Tracking", disable=tqdm_disable, leave=None
        ):

            obj_ids_all = []
            obj_ids_to_idx = {}
            video_res_masks_all = []
            base_idx = 0
            # Multi-session tracking
            for inference_state in self.inference_states.values():
                self.predictor.propagate_in_video_preflight(inference_state)

                batch_size = self.predictor._get_obj_num(inference_state)
                pred_masks_per_obj = [torch.empty(1,)] * batch_size
                for obj_idx in range(batch_size):
                    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                    # We skip those frames already in consolidated outputs (these are frames
                    # that received input clicks or mask). Note that we cannot directly run
                    # batched forward on them via `_run_single_frame_inference` because the
                    # number of clicks on each object might be different.
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        storage_key = "cond_frame_outputs"
                        current_out = obj_output_dict[storage_key][frame_idx]
                        device = inference_state["device"]
                        pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    else:
                        storage_key = "non_cond_frame_outputs"
                        inference_state['cached_features'] = self._shared_cached_features # link features
                        current_out, pred_masks = self.predictor._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=False,
                            run_mem_encoder=True,
                        )
                        obj_output_dict[storage_key][frame_idx] = current_out
                    
                    # Relink the features (useful when a cache miss was encountered)
                    self._shared_cached_features = inference_state['cached_features']
                    inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                        "reverse": False
                    }
                    pred_masks_per_obj[obj_idx] = pred_masks

                if len(pred_masks_per_obj) > 1:
                    all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                else:
                    all_pred_masks = pred_masks_per_obj[0]
                # Resize the output mask to the original video resolution (we directly use
                # the mask scores on GPU for output to avoid any CPU conversion in between)
                _, video_res_masks = self.predictor._get_orig_video_res_output(
                    inference_state, all_pred_masks
                )

                video_res_masks_all.append(video_res_masks)
                obj_ids_all.extend(inference_state["obj_ids"])
                # Update the mapping
                for id in inference_state["obj_ids"]:
                        obj_ids_to_idx[id] = inference_state['obj_id_to_idx'][id] + base_idx
                base_idx += len(inference_state["obj_ids"])

            yield frame_idx, obj_ids_all, obj_ids_to_idx, torch.cat(video_res_masks_all, dim=0) >= 0.
