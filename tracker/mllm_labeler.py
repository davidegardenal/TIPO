import json
from typing import Any

import numpy as np

from tracker.base_mllm import BaseMLLM

PROMPT_INITIAL_OBJECTS = """You are analyzing {frame_ref} of a video sequence where physical objects will undergo transformations (assembly, disassembly, etc.).

You are provided with one image:
[Image 1] {frame_image}, with colored contour outlines and letter labels highlighting the tracked objects.

The labeled regions are:
{labeled_regions}

Task: For each labeled region, provide a concise name for the physical object it represents.

You must respond ONLY with a valid JSON object using the following schema:
{{
  "summary": "One sentence describing the scene and the objects visible in the frame.",
  "objects": [
    {{"label": "A", "name": "concise object name"}},
    {{"label": "B", "name": "concise object name"}}
  ]
}}
"""

PROMPT_DISASSEMBLY = """You are analyzing a video sequence where a physical object has been DISASSEMBLED into multiple parts.

You are provided with two images:
[Image 1] Context frame (~5 seconds before the disassembly).
[Image 2] Current frame right after the disassembly, with colored contour outlines and letter labels highlighting the resulting parts.

The object that was disassembled:
{before_objects}

The resulting parts after disassembly (labeled in Image 2):
{labeled_regions}

Task:
1. Name each resulting part (labeled region) based on what you see in Image 2 and the context from Image 1.
2. Describe the specific disassembly action with only VERBS, you can include multiples if needed (e.g. "unscrew", "pull apart", "detach", "remove").

You must respond ONLY with a valid JSON object using the following schema:
{{
  "summary": "One sentence describing what happened.",
  "objects": [
    {{"label": "A", "name": "concise part name"}},
    {{"label": "B", "name": "concise part name"}}
  ],
  "action_name": "concise verb phrase for the disassembly action"
}}
"""

PROMPT_ASSEMBLY = """You are analyzing a video sequence where multiple physical objects have been ASSEMBLED into a single object.

You are provided with two images:
[Image 1] Context frame (~5 seconds before the assembly).
[Image 2] Current frame right after the assembly, with a colored contour outline and letter label highlighting the assembled result.

The individual parts that were assembled together:
{before_objects}

The assembled result (labeled in Image 2):
{labeled_regions}

Task:
1. Name the assembled result (labeled region) based on what you see in Image 2 and knowing it was formed from the parts listed above.
2. Describe the specific assembly action with only VERBS, you can include multiples if needed (e.g. "snap", "screw", "insert").

You must respond ONLY with a valid JSON object using the following schema:
{{
  "summary": "One sentence describing what happened.",
  "objects": [
    {{"label": "A", "name": "concise name for the assembled object"}}
  ],
  "action_name": "concise verb phrase for the assembly action"
}}
"""

CONTEXT_FRAME_SEC = 5


class MLLMLabeler(BaseMLLM):
    """MLLM Labeler labeling objects and actions in the state graph"""

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

    def label_transformation(
        self,
        video_path: str,
        frame_names: list[str],
        fps: float,
        frame_idx: int,
        obj_masks: dict[int, np.ndarray],
        before_names: dict[int, str] | None = None,
        transformation_type: str = "disassembly",
    ) -> tuple[dict[int, str], str | None]:
        # Build contour overlay on the current frame
        current_frame = self.read_frame(video_path, frame_names[frame_idx])
        annotated, labels = self.annotate_masks(current_frame, obj_masks)

        # Context frame ~5 s before
        context_idx = self.context_frame_idx(frame_idx, CONTEXT_FRAME_SEC, fps)
        context_frame = self.read_frame(video_path, frame_names[context_idx])

        # Build the text listing the input objects (before transformation)
        if before_names:
            before_parts = [f"- \"{name}\" (object ID {oid})" for oid, name in before_names.items()]
        else:
            before_parts = ["- (unknown)"]
        before_objects = "\n".join(before_parts)

        # Build the text listing labeled output regions
        labeled_regions = "\n".join(
            f"Region {entry['label']} ({entry['color']}): object ID {entry['obj_id']}" for entry in labels
        )

        # Select prompt based on transformation type
        prompt_template = PROMPT_ASSEMBLY if transformation_type == "assembly" else PROMPT_DISASSEMBLY

        # Construct message content (images as base64 data URLs only)
        content = [
            self.image_block(context_frame),
            self.image_block(annotated),
            {"type": "text", "text": prompt_template.format(
                before_objects=before_objects,
                labeled_regions=labeled_regions,
            )},
        ]
        result = self._parse_response(self._chat(content))
        return self._names_by_obj_id(result, labels), result.get("action_name")

    def label_initial_objects(
        self,
        video_path: str,
        frame_names: list[str],
        frame_idx: int,
        obj_masks: dict[int, np.ndarray],
    ) -> dict[int, str]:
        frame = self.read_frame(video_path, frame_names[frame_idx])
        annotated, labels = self.annotate_masks(frame, obj_masks)

        labeled_regions = "\n".join(f"Region {entry['label']} ({entry['color']})" for entry in labels)

        content = [
            self.image_block(annotated),
            {"type": "text", "text": PROMPT_INITIAL_OBJECTS.format(
                frame_ref="the first frame" if frame_idx == 0 else "a frame",
                frame_image="The first frame of the video" if frame_idx == 0
                else "A frame of the video where the labeled objects first appear",
                labeled_regions=labeled_regions,
            )},
        ]
        return self._names_by_obj_id(self._parse_response(self._chat(content)), labels)

    @staticmethod
    def _names_by_obj_id(result: dict[str, Any], labels: list[dict]) -> dict[int, str]:
        # Map the MLLM answer from region letter to object IDs
        letter_to_oid = {entry["label"]: entry["obj_id"] for entry in labels}
        return {
            letter_to_oid[o["label"]]: o["name"]
            for o in result.get("objects", [])
            if o.get("label") in letter_to_oid and o.get("name")
        }

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            first_nl = text.index("\n")
            text = text[first_nl + 1:]
        text = text.removesuffix("```")
        text = text.strip()
        return json.loads(text)
