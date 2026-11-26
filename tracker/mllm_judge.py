import os
from typing import Any

import numpy as np
import torch

from tracker.base_mllm import BaseMLLM

PROMPT_DISASSEMBLY = """You are analyzing a video sequence to track objects. You are provided five images:
[Image 1] Context frame (5s before).
[Image 2] Context frame (3s before).
[Image 3] Context frame (1s before).
[Image 4] Current frame.
[Image 5] Current frame with colored outlines and labels highlighting candidate object regions.

The outlined and labeled regions in Image 5 are: {labeled_regions}
All the candidate object regions in Image 5 currently tracked together. 
Task: Determine if these outlined regions have separated into physically distinct objects (e.g they are no more physically attached or touching) OR if they are still a single object (e.g they are still physically attached).

You must respond ONLY with a valid JSON object using the following schema:
{{
  "analysis": "Provide exactly ONE short sentence comparing the context frame to the current frame to identify occlusion or breakage.",
  "all_distinct": "YES or NO. Answer YES only if confident they are distinct objects. Answer NO if they are parts of the same object."
}}
"""

PROMPT_ASSEMBLY = """You are analyzing a video sequence to track objects. You are provided five images:
[Image 1] Context frame (5s before).
[Image 2] Context frame (3s before).
[Image 3] Context frame (1s before).
[Image 4] Current frame.
[Image 5] Current frame with colored outlines and labels highlighting regions belonging to different tracked objects.

The outlined and labeled regions in Image 5 are: {labeled_regions}
Task: Determine if these objects have been assembled (e.g they are physically attached/joined together) OR if they remain distinct, independent objects (e.g they are not physically attached).

You must respond ONLY with a valid JSON object using the following schema:
{{
  "analysis": "Provide exactly ONE short sentence comparing the context frame to the current frame to verify if physical attachment or merging occurred.",
  "all_assembled": "YES or NO. Answer YES only if confident ALL the objects have assembled into one. Answer NO if they are still separate objects or if uncertain."
}}
"""

OVERLAY_ALPHA = 0.3
CONTEXT_FRAME_SEC = [5, 3, 1] # seconds before the current frame


class MLLMJudge(BaseMLLM):
    """MLLM Judge deciding whether a candidate assembly/disassembly really happened."""

    def __init__(self, conf):
        super().__init__(
            model_path=conf.MODEL_PATH,
            base_url=conf.BASE_URL,
            port=conf.PORT,
            api_key=conf.API_KEY,
            max_tokens=conf.MAX_TOKENS,
            temperature=conf.TEMPERATURE,
            top_p=conf.TOP_P,
            presence_penalty=conf.PRESENCE_PENALTY,
            top_k=conf.TOP_K,
            enable_thinking=conf.ENABLE_THINKING,
            reasoning_effort=conf.REASONING_EFFORT,
        )
        # Stop right after the JSON answer: only the final YES/NO is read
        self.gen_kwargs["stop"] = ["}", "\n}"]
        self._video_path = ""
        self._frame_names = []
        self._video_fps = 0.

    def set_video(self, video_path: str, fps: float):
        self._video_path = video_path
        self._video_fps = fps
        self._frame_names = sorted(os.listdir(video_path))

    def _frame_path(self, frame_idx: int) -> str:
        return os.path.join(self._video_path, self._frame_names[frame_idx])

    def _check_transformation(self, current_frame_idx: int, obj_masks: dict[Any, torch.Tensor], prompt: str):
        context_frames_idx = [self.context_frame_idx(current_frame_idx, sec, self._video_fps) for sec in CONTEXT_FRAME_SEC]
        context_frames = [self.image_block(self._frame_path(idx)) for idx in context_frames_idx]
        current_frame = self.image_block(self._frame_path(current_frame_idx))

        masks_np: dict[Any, np.ndarray] = {key: t[0].cpu().numpy() for key, t in obj_masks.items()}
        frame = self.read_frame(self._video_path, self._frame_names[current_frame_idx])
        overlay_img, labels = self.annotate_masks(frame, masks_np, fill_alpha=OVERLAY_ALPHA)
        overlay_frame = self.image_block(overlay_img)

        labeled_regions = ", ".join(f"Region {l['label']} ({l['color']})" for l in labels)

        content = context_frames + [
            current_frame,
            overlay_frame,
            {"type": "text", "text": prompt.format(labeled_regions=labeled_regions)}
        ]
        answer = self._chat(content).upper()
        return "YES" in answer[-10:]

    def is_disassembled(self, current_frame_idx: int, obj_masks: dict[Any, torch.Tensor]) -> bool:
        return self._check_transformation(current_frame_idx, obj_masks, PROMPT_DISASSEMBLY)

    def is_assembled(self, current_frame_idx: int, obj_masks: dict[Any, torch.Tensor]) -> bool:
        return self._check_transformation(current_frame_idx, obj_masks, PROMPT_ASSEMBLY)
