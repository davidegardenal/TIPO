import base64
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from openai import APIError, OpenAI

logger = logging.getLogger(__name__)


OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "red":     (255,   0,   0),
    "blue":    (  0,   0, 255),
    "green":   (  0, 200,   0),
    "yellow":  (255, 220,   0),
    "magenta": (255,   0, 255),
    "orange":  (255, 165,   0),
    "purple":  (128,   0, 128),
}

CONTOUR_THICKNESS = 3


class BaseMLLM:
    """OpenAI-compatible multimodal LLM client with image utilities"""

    def __init__(
        self,
        model_path: str,
        base_url: str,
        port: int | None,
        api_key: str | None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model_path = model_path
        self.gen_kwargs = self.build_gen_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            top_k=top_k,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        # Empty key -> None, so the OpenAI client falls back to OPENAI_API_KEY
        self.client = OpenAI(base_url=self.build_api_url(base_url, port), api_key=api_key or None)

    def wait_for_server(self, timeout: float = 300, poll_interval: float = 5) -> None:
        """Block until the server responds or *timeout* seconds elapse."""
        name = type(self).__name__
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.client.models.list()
                logger.info("%s server is ready.", name)
                return
            except APIError: # connection / timeout / HTTP status errors while the server starts
                remaining = int(deadline - time.time())
                logger.info(
                    "%s server not reachable yet - retrying in %ds (timeout in %ds)…",
                    name, poll_interval, remaining,
                )
                time.sleep(poll_interval)
        raise TimeoutError(f"{name} server did not become available within {timeout}s.")

    def _chat(self, content: list[dict]) -> str:
        """Send a single user message and return the stripped text answer."""
        response = self.client.chat.completions.create(  # pyrefly: ignore
            model=self.model_path,
            messages=[{"role": "user", "content": content}],
            **self.gen_kwargs,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def build_api_url(base_url: str, port: int | None) -> str:
        url = base_url.rstrip("/")
        if port is not None:
            url = f"{url}:{port}"
        return f"{url}/v1"

    @staticmethod
    def build_gen_kwargs(
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        presence_penalty: float | None,
        top_k: int | None,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        """Build `chat.completions.create` kwargs.

        `top_k` and `enable_thinking` are sent only to vLLM servers in `extra_body`.
        """
        standard = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "reasoning_effort": reasoning_effort,
        }
        gen_kwargs: dict[str, Any] = {k: v for k, v in standard.items() if v is not None}

        extra_body: dict[str, Any] = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if extra_body:
            gen_kwargs["extra_body"] = extra_body
        return gen_kwargs

    ## Image utils

    @staticmethod
    def read_frame(video_path: str, frame_name: str) -> np.ndarray:
        """Read a BGR frame"""
        frame = cv2.imread(os.path.join(video_path, frame_name))
        if frame is None:
            raise RuntimeError(f"Could not read frame {frame_name}")
        return frame

    @staticmethod
    def context_frame_idx(frame_idx: int, seconds_before: float, fps: float) -> int:
        """Index of the frame *seconds_before* seconds before *frame_idx* (clamped to 0)."""
        return max(0, frame_idx - round(seconds_before * fps))

    @staticmethod
    def image_block(img: "str | np.ndarray") -> dict:
        """Build an OpenAI `image_url` content block.

        * str  -> treated as a local file path, sent as `file://` URI.
        * ndarray -> BGR image, JPEG-encoded and base64-wrapped.
        """
        if isinstance(img, str):
            url = f"file://{img}"
        elif isinstance(img, np.ndarray):
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                raise RuntimeError("Failed to JPEG-encode the image array.")
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            url = f"data:image/jpeg;base64,{b64}"
        else:
            raise TypeError(f"Expected str or np.ndarray, got {type(img)}")
        return {"type": "image_url", "image_url": {"url": url}}

    @staticmethod
    def annotate_masks(
        frame: np.ndarray,
        obj_masks: dict[Any, np.ndarray],
        fill_alpha: float = 0.0,
    ) -> tuple[np.ndarray, list[dict]]:
        """Draw colored contours and letter labels for each mask on a copy of *frame*.

        Letters follow the iteration order of *obj_masks* (`A` for the first key,
        `B` for the second, ...).
        With `fill_alpha > 0` the mask areas are also filled with their color and
        alpha-blended onto the annotated frame.

        Returns the annotated BGR image and a list of `{obj_id, label, color}` entries.
        """
        annotated = frame.copy()
        overlay = frame.copy()

        masks_np = {k: (m > 0).astype(np.uint8) for k, m in obj_masks.items()}

        # Union of all the other masks, per object, for label placement
        all_other = {k: np.zeros_like(m) for k, m in masks_np.items()}
        for k in masks_np:
            for other_k, other_m in masks_np.items():
                if other_k != k:
                    all_other[k] = all_other[k] | other_m

        color_entries = list(OVERLAY_COLORS.items())
        labels: list[dict] = []
        for i, (obj_id, mask) in enumerate(masks_np.items()):
            if mask.sum() == 0:
                continue

            color_name, color_rgb = color_entries[i % len(color_entries)]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            letter = chr(ord("A") + i)

            if fill_alpha > 0:
                overlay[mask != 0] = color_bgr

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color_bgr, thickness=CONTOUR_THICKNESS)

            # Farthest-from-others placement: maximize distance transform within
            # the exclusive area of this mask (pixels not covered by other masks).
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            exclusive = dist.copy()
            exclusive[all_other[obj_id] == 1] = 0
            if exclusive.max() > 0:
                idx = np.unravel_index(exclusive.argmax(), exclusive.shape)
            else:
                # Fallback to centroid when mask is fully covered by others.
                ys, xs = np.where(mask != 0)
                idx = (int(ys.mean()), int(xs.mean()))
            pos = (int(idx[1]), int(idx[0]))  # cv2 uses (x, y)

            cv2.putText(annotated, letter, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)  # dark outline
            cv2.putText(annotated, letter, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_bgr, 3)  # colored fill

            labels.append({"obj_id": obj_id, "label": letter, "color": color_name})

        if fill_alpha > 0:
            annotated = cv2.addWeighted(overlay, fill_alpha, annotated, 1.0 - fill_alpha, 0)
        return annotated, labels
