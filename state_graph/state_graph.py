from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils

if TYPE_CHECKING:
    from tracker.mllm_labeler import MLLMLabeler
    from utils import RLE

DEFAULT_FPS = 10

@dataclass
class Object:
    """A segmented object within a State.

    Attributes:
        id:            Unique integer identifier for this object.
        name:          Human-readable name.
        mask:          Instance mask stored as a ``torch.long`` tensor and
                       optionally converted in-place to a COCO RLE dictionary.
        component_ids: Non-zero IDs found in *mask* via ``torch.unique()``.
                       Background label ``0`` is automatically excluded.
        sampled_masks: Tensor holding the binary masks sampled with SAM
                       
    """

    id: int
    parent_ids: list[int]
    name: str
    mask: torch.Tensor | RLE
    component_ids: list[int] = field(init=False, default_factory=list)
    sampled_masks: torch.Tensor
    hypo_states: list[list[list[int]]] # (obj_state, partition, component_id)
    states_prob: np.ndarray

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.mask):
            raise TypeError(
                f"Object.mask must have dtype torch.long, got {type(self.mask)}"
            )
        if self.mask.dtype != torch.long:
            raise TypeError(
                f"Object.mask must have dtype torch.long, got {self.mask.dtype}"
            )
        unique_ids = torch.unique(self.mask).tolist()
        self.component_ids = [int(v) for v in unique_ids if int(v) != 0]

    @classmethod
    def create(cls, id: int, parent_ids: list[int], name: str, mask: torch.Tensor) -> Object:
        """Create a new :class:`Object`.

        ``component_ids`` is derived automatically from *mask* via
        ``torch.unique()``, with the background label ``0`` excluded.

        Args:
            id:   Unique identifier.
            name: Human-readable name.
            mask: Integer mask tensor (will be cast to ``torch.long``).

        Returns:
            A new :class:`Object` instance.
        """
        return cls(
            id=id, parent_ids=parent_ids,
            name=name, mask=mask.to(torch.long), 
            sampled_masks=torch.empty(()), 
            hypo_states=[[[id]]], # Only 1 possible state
            states_prob=np.array([1.,]))

    def get_sampled_mask(self, comp_id: int) -> torch.Tensor:
        return self.sampled_masks[self.component_ids.index(comp_id)]

    def get_binary_mask(self) -> np.ndarray:
        if torch.is_tensor(self.mask):
            bin_mask = self.mask != 0
            if torch.is_tensor(bin_mask):
                bin_mask = bin_mask.detach().cpu()
            return np.asarray(bin_mask).squeeze().astype(bool)

        return maskUtils.decode(self.mask).astype(bool)

    def get_rle_mask(self) -> RLE:
        if isinstance(self.mask, dict):
            return self.mask

        mask_np = self.get_binary_mask()
        rle = maskUtils.encode(np.asfortranarray(mask_np))
        return {
            "counts": rle["counts"].decode("ascii"), # pyrefly: ignore
            "size": [v for v in rle["size"]],
        }

    def encode_mask_to_rle(self) -> None:
        rle_mask = self.get_rle_mask()
        del self.mask
        self.mask = rle_mask

    def to(self, device: str|torch.device):
        if torch.is_tensor(self.mask):
            self.mask = self.mask.to(device)
        self.sampled_masks = self.sampled_masks.to(device)

    def __repr__(self) -> str:
        if torch.is_tensor(self.mask):
            mask_desc = f"mask_shape={tuple(self.mask.shape)}"
        else:
            mask_desc = f"mask_rle_size={self.mask['size']}"
        return (
            f"Object(id={self.id}, name={self.name!r}, "
            f"{mask_desc}, "
            f"component_ids={self.component_ids})"
        )

@dataclass
class State:
    """A node in the :class:`StateGraph`.

    Attributes:
        objects:     Dictionary mapping object ID → :class:`Object`.
    """

    hypo_asmbl_states: list[list[list[int]]] # (state, partition, obj_id)
    asmbl_states_prob: np.ndarray
    asmbl_belief: np.ndarray
    objects: dict[int, Object] = field(default_factory=dict)

    @classmethod
    def create(cls, objects: list[Object] | None = None) -> State:
        """Create a new :class:`State`.

        Args:
            objects: Optional list of :class:`Object` instances to include.

        Returns:
            A new :class:`State` instance.
        """
        obj_dict: dict[int, Object] = {}
        for obj in (objects or []):
            obj_dict[obj.id] = obj
        return cls(objects=obj_dict, hypo_asmbl_states=[], asmbl_states_prob=np.array([1.,]), asmbl_belief=np.array([1.,]))

    def to(self, device: str|torch.device):
        for obj_idx in self.objects:
            self.objects[obj_idx].to(device)

    def clean(self):
        for obj in self.objects.values():
            del obj.sampled_masks
            obj.sampled_masks = torch.empty(1,)
            obj.encode_mask_to_rle()

    def __repr__(self) -> str:
        return (
            f"State(objects=[{', '.join(str(k) for k in self.objects)}])"
        )

@dataclass
class Transformation:
    """A directed edge in the :class:`StateGraph`.

    Attributes:
        frame_index: Position of this transformation within the graph. Set
                     automatically by :class:`StateGraph` when inserted;
                     ``None`` until then.
        parts:       Set of object IDs that participate in the transformation.
        next_ids:    Set of object IDs produced / relevant after the transformation.
        description: Human-readable description of what the transformation does.
    """

    frame_index: int | None = field(init=False, default=None)
    parts: set[int] = field(default_factory=set)
    next_ids: set[int] = field(default_factory=set)
    description: str = ""

    @classmethod
    def create(
        cls,
        parts: set[int] | None = None,
        next_ids: set[int] | None = None,
        description: str = "",
    ) -> Transformation:
        """Create a new :class:`Transformation`.

        ``frame_index`` is **not** set here; it will be assigned automatically
        when this transformation is inserted into a :class:`StateGraph`.

        Args:
            parts:       Set of participating object IDs.
            next_ids:    Set of resulting / successor object IDs.
            description: Optional text description.

        Returns:
            A new :class:`Transformation` instance.
        """
        return cls(
            parts=set(parts) if parts is not None else set(),
            next_ids=set(next_ids) if next_ids is not None else set(),
            description=description,
        )

    @property
    def is_identity(self) -> bool:
        """True for the no-op edges that carry an object unchanged to the next frame."""
        return self.parts == self.next_ids

    def __repr__(self) -> str:
        return (
            f"Transformation(frame_index={self.frame_index}, "
            f"parts={self.parts}, next_ids={self.next_ids}, "
            f"description={self.description!r})"
        )

class StateGraph:
    """A graph whose nodes are :class:`State` objects and whose edges are
    :class:`Transformation` objects, both indexed by integer *frame indices*.

    States and transformations are stored in their respective dictionaries.
    Assign a ``frame_index`` to an item by inserting it directly into the
    appropriate collection::

        graph = StateGraph()
        graph.states[0] = State.create(...)
        graph.transformations[0] = Transformation.create(...)

    Attributes:
        states:          Dict[int, State]          - all states, keyed by frame index.
        transformations: Dict[int, Transformation] - all transformations, keyed by frame index.
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.states: dict[int, State] = {}
        self.transformations: dict[int, list[Transformation]] = {}
        # Sequence metadata, written to / read from the prediction JSON
        self.sequence: str | None = None
        self.fps: float | None = None
        self.tracking_elapsed_seconds: float | None = None

    @classmethod
    def from_predictions_json(cls, json_path: str, video_path: str) -> StateGraph:
        """Load a graph written by :meth:`export_predictions_json`, with its sequence metadata."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls(video_path)
        names: dict[int, str] = {}
        parents: dict[int, list[int]] = {}
        for t in data.get("transformations", []):
            before = [o["obj_id"] for o in t["beforeObjects"]]
            after = [o["obj_id"] for o in t["afterObjects"]]
            for o in t["beforeObjects"] + t["afterObjects"]:
                names[o["obj_id"]] = o["name"]
            transf = Transformation.create(parts=set(before), next_ids=set(after), description=t["description"])
            if not transf.is_identity:
                for oid in after:
                    parents[oid] = before
            graph.transformations.setdefault(int(t["frameIdx"]), []).append(transf)

        # Objects are created with a dummy tensor mask, then given their RLE mask to respect the construction contract
        with_rle = lambda obj, rle: setattr(obj, "mask", rle) or obj
        for f_idx, frame_pred in data["prediction"].items():
            graph.states[int(f_idx)] = State.create([
                with_rle(Object.create(id=int(oid), parent_ids=parents.get(int(oid), [-1]), name=names.get(int(oid), "TODO"), mask=torch.zeros(1)), rle)
                for oid, rle in frame_pred.items()
            ])

        graph.sequence = data["sequence"]
        graph.fps = data.get("fps")
        graph.tracking_elapsed_seconds = data.get("tracking_elapsed_seconds")
        return graph

    def fill_semantic_labels(self, labeler: MLLMLabeler) -> None:
        """Fill object names and transformation descriptions in place with *labeler* (needs ``self.fps``)."""
        if self.fps is None: self.fps = DEFAULT_FPS
        frame_names = sorted(os.listdir(self.video_path))
        names: dict[int, str] = {}

        # Discover prompted objects in the graph
        roots: dict[int, list[int]] = {}
        seen: set[int] = set()
        for f_idx in sorted(self.states):
            for oid, obj in self.states[f_idx].objects.items():
                if oid not in seen:
                    seen.add(oid)
                    if obj.parent_ids == [-1]:
                        roots.setdefault(f_idx, []).append(oid)

        # Fill semantic labels for each initial object and transformation
        for f_idx in sorted(set(roots) | set(self.transformations)):
            if f_idx in roots:
                names.update(labeler.label_initial_objects(
                    self.video_path, frame_names, f_idx, self._binary_masks(f_idx, roots[f_idx])))

            for t in self.transformations.get(f_idx, []):
                if t.is_identity: continue
                masks = self._binary_masks(f_idx, t.next_ids)
                if not masks: continue

                new_names, action = labeler.label_transformation(
                    self.video_path, frame_names, self.fps, f_idx, masks,
                    before_names={oid: names.get(oid, "TODO") for oid in t.parts},
                    transformation_type="assembly" if len(t.parts) > 1 and len(t.next_ids) == 1 else "disassembly",
                )
                for oid, name in new_names.items():
                    names.setdefault(oid, name)
                if action: t.description = action

        # Propagate object names
        for state in self.states.values():
            for oid, obj in state.objects.items():
                if oid in names:
                    obj.name = names[oid]

    def _binary_masks(self, frame_idx: int, obj_ids) -> dict[int, np.ndarray]:
        objects = self.states[frame_idx].objects
        return {oid: objects[oid].get_binary_mask() for oid in obj_ids if oid in objects}

    def __repr__(self) -> str:
        return (
            f"StateGraph("
            f"states={list(self.states.keys())}, "
            f"transformations={list(self.transformations.keys())})"
        )

    def overlay_masks(self, frame: np.ndarray, frame_idx: int, colors: dict[int, tuple[int, int, int]],
                      alpha: float) -> np.ndarray:
        """Paint the masks at *frame_idx* of the objects in *colors* (BGR) on *frame*, alpha-blended."""
        state = self.states.get(frame_idx)
        if state is None:
            return frame
        overlay = frame.copy()
        for oid, obj in state.objects.items():
            if oid in colors:
                mask = obj.get_binary_mask()
                if mask.shape == frame.shape[:2]:
                    overlay[mask] = colors[oid]
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def export_video(self, out_path: str, alpha: float = 0.5) -> None:
        """Render the video at ``self.fps`` with the object masks overlaid (same colours as
        :meth:`export_graph`) and save it as .mp4."""
        from .graph_svg import object_colors

        if self.fps is None:
            raise ValueError("StateGraph.fps is required to export a video")
        colors = object_colors(self)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for frame_idx, fname in enumerate(sorted(os.listdir(self.video_path))):
                frame = cv2.imread(os.path.join(self.video_path, fname))
                if frame is None:
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))  # pyrefly: ignore
                writer.write(self.overlay_masks(frame, frame_idx, colors, alpha))
        finally:
            if writer is not None:
                writer.release()

    def export_graph(self, out_path: str, thumbnails: bool = True):
        """Export the state graph as an SVG in the style of TIPO's paper figures.
        Args:
            out_path:   Destination path for the ``.svg`` file.
            thumbnails: If True, add a strip of key-frame thumbnails (one per column, with the object masks
                        painted in the node colours) on top of the graph. Needs ``video_path``.
        """
        from .graph_svg import render_svg

        render_svg(self, out_path, thumbnails=thumbnails)

    def export_predictions_json(self, out_path: str) -> None:
        """Write the graph as a prediction JSON (masks as COCO RLE), with its sequence metadata.

        Output JSON schema:
        {
          "sequence": "<sequence name>",
          "prediction": {
             "<frame_idx>": { "<obj_id>": {"counts": "...", "size": [H, W]}, ... },
             ...
          },
          "fps": <float>,                       // optional
          "tracking_elapsed_seconds": <float>,  // optional, wall time for init_session + track_transformations
          "transformations": [
             {"frameIdx": <int>,
              "beforeObjects": [{"obj_id": <int>, "name": "..."}, ...],
              "afterObjects":  [{"obj_id": <int>, "name": "..."}, ...],
              "description": "..."},
             ...
          ]
        }
        """
        names = {oid: obj.name for f_idx in sorted(self.states) for oid, obj in self.states[f_idx].objects.items()}
        objs = lambda ids: [{"obj_id": oid, "name": names[oid]} for oid in ids]

        out: dict = {
            "sequence": self.sequence,
            "prediction": {
                str(f_idx): {str(oid): obj.get_rle_mask() for oid, obj in self.states[f_idx].objects.items()}
                for f_idx in sorted(self.states)
            },
        }
        if self.fps is not None:
            out["fps"] = float(self.fps)
        if self.tracking_elapsed_seconds is not None:
            out["tracking_elapsed_seconds"] = float(self.tracking_elapsed_seconds)
        out["transformations"] = [
            {"frameIdx": f_idx, "beforeObjects": objs(t.parts), "afterObjects": objs(t.next_ids), "description": t.description}
            for f_idx in sorted(self.transformations) for t in self.transformations[f_idx]
        ]

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f)
