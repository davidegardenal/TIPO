"""
This code is not humanly readable but the result is really good,
Thanks Claude :)

Render a :class:`state_graph.StateGraph` as an SVG in the style of the TIPO paper figures.

Layout (left to right):
  - initial objects are stacked on the left, with their name right-aligned next to the node;
  - every transformation gets its own column: the inputs join in a dark square junction (labelled in italics with
    the action), and an arrow goes from the junction to each resulting object node (name centred below the node);
  - every object keeps one horizontal lane from its node to the junction that consumes it. Objects that survive
    until the end of the video are drawn with a dark outline and a bold name;
  - optionally, a strip of key-frame thumbnails (with the object masks painted in the node colours) sits on top,
    one per column, linked to the new nodes by a dashed line.

Lanes are assigned with a small search (order of the initial objects, side on which split results go) that
minimises the number of lines crossing a lane, then the total vertical travel. Pure Python: only numpy and cv2.
"""

from __future__ import annotations

import base64
import itertools
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .state_graph import StateGraph

# ---------------------------------------------------------------- style (taken from the paper figures)
PALETTE = ["#E69F00", "#68023F", "#6B5B95", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#D9C800", "#40B0A6"]
INK = "#2B3036"        # lines, junctions, outlines
TEXT = "#15181C"       # object names
TEXT_2 = "#3B4148"     # action labels
DASH = "#BFC2C7"       # thumbnail connectors
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

FS = 12.0              # font size of the names
FS_ACT = 11.5          # font size of the action labels
LH = 14.0              # line height
R = 8.0                # node radius
R_FINAL = 9.5          # radius of a final node (plus its outline)
SQ = 13.0              # junction square side
LW = 2.2               # line width
LANE = 32.0            # minimum distance between two lanes
CORNER = 7.0           # radius of the elbow corners
S_CURVE = 28.0         # horizontal length of the S curves between adjacent lanes
ARROW = 7.0            # arrow head length
LABEL_W = 118.0        # max width of a wrapped name
PAD = 10.0
STEP_MIN = 150.0       # min distance between two columns
THUMB_GAP = 10.0       # horizontal gap between thumbnails
THUMB_ALPHA = 0.6      # opacity of the masks painted on the thumbnails

PLACEHOLDERS = {"", "todo", "none"}

# Helvetica advance widths (1/1000 em) for ASCII 32..126, used to measure text without a font engine.
_HELV = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556,
         556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
         500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469,
         556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
         278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]


def text_width(s: str, size: float = FS, bold: bool = False) -> float:
    """Approximate rendered width of *s* (Inter is ~5% wider than Helvetica, bold ~6% more)."""
    w = sum(_HELV[ord(ch) - 32] if 32 <= ord(ch) <= 126 else 600 for ch in s)
    return w / 1000.0 * size * 1.05 * (1.06 if bold else 1.0)


def wrap(s: str, max_w: float = LABEL_W, size: float = FS, bold: bool = False, max_lines: int = 3) -> list[str]:
    """Greedy word wrap to *max_w* pixels, at most *max_lines* lines (the last one gets an ellipsis if cut)."""
    lines: list[str] = []
    cur = ""
    for word in s.split():
        cand = f"{cur} {word}" if cur else word
        if cur and text_width(cand, size, bold) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "…"
    return lines or [""]


def _cap(s: str) -> str:
    s = " ".join(s.replace("_", " ").split())
    return s[:1].upper() + s[1:]


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- graph model
@dataclass
class _Node:
    idx: int
    obj_id: int
    name: str
    col: int                     # column where the node is drawn
    kind: str                    # "root" (first frame), "appear" (prompted later) or "event" (result of a transformation)
    death: int | None = None     # column of the transformation that consumes it
    frames: tuple[int, int] = (0, 0)
    lane: int = 0
    color: str = INK
    lines: list[str] = field(default_factory=list)

    @property
    def final(self) -> bool:
        return self.death is None


@dataclass
class _Event:
    col: int
    frame: int
    inputs: list[int]            # node indices
    outputs: list[int]
    label: str
    main_in: int = -1
    main_out: int = -1


@dataclass
class _Column:
    frame: int
    event: _Event | None = None  # None for the first column (initial objects) and for columns where objects appear
    appear: list[int] = field(default_factory=list)


class _Model:
    """Object nodes, transformation events and columns extracted from a StateGraph."""

    def __init__(self, graph: StateGraph) -> None:
        frames = sorted(graph.states)
        self.f0 = frames[0] if frames else 0
        first_seen: dict[int, int] = {}
        last_seen: dict[int, int] = {}
        names: dict[int, str] = {}
        for f in frames:
            for oid, obj in graph.states[f].objects.items():
                first_seen.setdefault(oid, f)
                last_seen[oid] = f
                if obj.name and obj.name.strip().lower() not in PLACEHOLDERS:
                    names.setdefault(oid, obj.name)

        raw = []
        for f in sorted(graph.transformations):
            t_list = graph.transformations[f]
            if not isinstance(t_list, list):
                t_list = [t_list]
            for t in t_list:
                if t.description.strip().lower() == "none" or set(t.parts) == set(t.next_ids):
                    continue
                raw.append((f, sorted(t.parts), sorted(t.next_ids), t.description))

        produced = {o for _, _, outs, _ in raw for o in outs}
        appear_at: dict[int, list[int]] = defaultdict(list)
        for oid, f in first_seen.items():
            if oid not in produced and f != self.f0:
                appear_at[f].append(oid)

        # columns: one per transformation, plus one per frame where new objects are prompted
        items = [(f, 0, k) for k, (f, _, _, _) in enumerate(raw)] + [(f, 1, -1) for f in appear_at]
        items.sort()
        self.columns: list[_Column] = [_Column(frame=self.f0)]
        self.nodes: list[_Node] = []
        self.events: list[_Event] = []
        current: dict[int, int] = {}     # obj_id -> node alive now
        self.alive_at: list[dict[int, int]] = []

        def new_node(oid: int, col: int, kind: str, frame: int) -> int:
            raw_name = names.get(oid)
            name = _cap(raw_name) if raw_name else f"Object {oid}"
            n = _Node(idx=len(self.nodes), obj_id=oid, name=name, col=col, kind=kind)
            n.frames = (max(first_seen.get(oid, frame), frame), last_seen.get(oid, frame))
            self.nodes.append(n)
            current[oid] = n.idx
            return n.idx

        roots0 = sorted(oid for oid, f in first_seen.items() if f == self.f0 and oid not in produced)
        # inputs never seen in a state and never produced: treat them as initial objects
        roots0 += sorted({o for _, ins, _, _ in raw for o in ins} - set(first_seen) - produced - set(roots0))
        for oid in roots0:
            new_node(oid, 0, "root", self.f0)
        self.alive_at.append(dict(current))

        for f, kind, k in items:
            col = len(self.columns)
            if kind == 1:
                c = _Column(frame=f)
                self.columns.append(c)
                c.appear = [new_node(oid, col, "appear", f) for oid in sorted(appear_at[f])]
            else:
                _, ins, outs, desc = raw[k]
                in_nodes = [current.pop(o) for o in ins if o in current]
                for ni in in_nodes:
                    self.nodes[ni].death = col
                out_nodes = [new_node(o, col, "event", f) for o in outs]
                if desc.strip().lower() in PLACEHOLDERS:
                    desc = ("Assemble" if len(ins) > 1 and len(outs) == 1
                            else "Disassemble" if len(ins) == 1 and len(outs) > 1 else "Transform")
                ev = _Event(col=col, frame=f, inputs=in_nodes, outputs=out_nodes, label=_cap(desc))
                self.events.append(ev)
                self.columns.append(_Column(frame=f, event=ev))
            self.alive_at.append(dict(current))

        # trunk heuristics: a merge keeps the lane of the input with the longest history, a split keeps on its lane
        # the output with the longest future
        anc = [0] * len(self.nodes)
        for ev in self.events:
            a = 1 + max((anc[i] for i in ev.inputs), default=0)
            for o in ev.outputs:
                anc[o] = a
        desc_n = [0] * len(self.nodes)
        for ev in reversed(self.events):
            d = 1 + max((desc_n[o] for o in ev.outputs), default=0)
            for i in ev.inputs:
                desc_n[i] = max(desc_n[i], d)
        self.anc, self.desc = anc, desc_n
        n_cols = len(self.columns)
        self.trunk = [n.kind != "event" for n in self.nodes]   # True if the node continues its parent's lane
        for ev in self.events:
            if ev.outputs:
                life = lambda o: (self.nodes[o].death or n_cols) - self.nodes[o].col
                ev.main_out = max(ev.outputs, key=lambda o: (desc_n[o], life(o), -o))
                self.trunk[ev.main_out] = True

    # half-column units: node of column c at 2c, junction of column c at 2c - 1
    def span(self, n: _Node) -> tuple[float, float]:
        start = 2 * n.col - 1 if n.kind == "event" else 2 * n.col
        end = 2 * n.death - 1 if n.death is not None else 2 * n.col + 0.6
        return start, end


# ---------------------------------------------------------------- lane assignment
class _Lanes:
    def __init__(self, m: _Model) -> None:
        self.m = m
        self.roots = [n.idx for n in m.nodes if n.kind == "root"]
        # binary choices: side of the split results / of the objects that appear later
        self.choices = [c for c, col in enumerate(m.columns)
                        if (col.event is not None and len(col.event.outputs) > 1) or col.appear]

    def simulate(self, order: list[int], flip: dict[int, int]) -> tuple[float, dict[int, int]]:
        m = self.m
        lane: dict[int, int] = {}
        occ: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
        span = {n.idx: m.span(n) for n in m.nodes}

        def free(l: int, s: float, e: float) -> bool:
            return all(e2 < s or s2 > e for s2, e2, _ in occ[l])

        def put(ni: int, l: int) -> None:
            lane[ni] = l
            occ[l].append((*span[ni], ni))

        def nearest(target: int, ni: int, d0: int, below_first: bool) -> int:
            s, e = span[ni]
            sgn = 1 if below_first else -1
            for d in itertools.count(d0):
                for l in ((target + sgn * d, target - sgn * d) if d else (target,)):
                    if free(l, s, e):
                        return l
            raise AssertionError

        for k, ni in enumerate(order):
            put(ni, k)
        main_in: dict[int, int] = {}
        for c, col in enumerate(m.columns[1:], start=1):
            ev = col.event
            if ev is None:
                for ni in col.appear:
                    # next to the objects it will be combined with, if any
                    partners = [lane[p] for e2 in m.events if ni in e2.inputs for p in e2.inputs if p in lane]
                    target = round(float(np.median(partners))) if partners else (max(lane.values(), default=-1) + 1)
                    put(ni, nearest(target, ni, 0, flip.get(c, 0) == 0))
                continue
            ins = [i for i in ev.inputs if i in lane]
            if ins:
                med = float(np.median([lane[i] for i in ins]))
                mi = max(ins, key=lambda i: (m.anc[i], m.trunk[i], -abs(lane[i] - med), -lane[i]))
                main_in[c] = mi
                jl = lane[mi]
            else:
                jl = nearest(max(lane.values(), default=-1) + 1, ev.outputs[0], 0, True) if ev.outputs else 0
            if ev.main_out >= 0:
                put(ev.main_out, jl)
            for o in ev.outputs:
                if o != ev.main_out:
                    put(o, nearest(jl, o, 1, flip.get(c, 0) == 0))

        # cost: lanes crossed by the vertical connectors, then vertical travel, then height
        cross = vert = 0
        for c, col in enumerate(m.columns):
            ev = col.event
            if ev is None or c not in main_in and ev.main_out < 0:
                continue
            jl = lane[main_in[c]] if c in main_in else lane[ev.main_out]
            x = 2 * c - 1
            involved = set(ev.inputs) | set(ev.outputs)
            for o in list(ev.inputs) + list(ev.outputs):
                if o not in lane or lane[o] == jl:
                    continue
                lo, hi = sorted((lane[o], jl))
                vert += hi - lo
                for l in range(lo + 1, hi):
                    cross += any(s <= x <= e and ni not in involved for s, e, ni in occ[l])
        n_lanes = max(lane.values()) - min(lane.values()) + 1 if lane else 1
        self.main_in = main_in
        return cross * 1000 + vert * 2 + n_lanes, lane

    def solve(self) -> dict[int, int]:
        roots, choices = self.roots, self.choices
        space = math.factorial(min(len(roots), 10)) * 2 ** min(len(choices), 20)
        best: tuple[float, list[int], dict[int, int]] = (math.inf, roots, {})
        if space <= 5000:
            for perm in itertools.permutations(roots):
                for bits in itertools.product((0, 1), repeat=len(choices)):
                    flip = dict(zip(choices, bits))
                    cost, _ = self.simulate(list(perm), flip)
                    if cost < best[0]:
                        best = (cost, list(perm), flip)
        else:
            rng = random.Random(0)
            budget = int(np.clip(4e5 / (1 + len(self.m.nodes)) ** 1.3, 150, 4000))
            for restart in range(3):
                perm, flip = list(roots), {c: 0 for c in choices}
                if restart:
                    rng.shuffle(perm)
                    flip = {c: rng.randint(0, 1) for c in choices}
                cur, _ = self.simulate(perm, flip)
                for _ in range(budget // 3):
                    p2, f2 = list(perm), dict(flip)
                    if choices and (len(perm) < 2 or rng.random() < 0.5):
                        c = rng.choice(choices)
                        f2[c] = 1 - f2[c]
                    elif len(perm) >= 2:
                        i, j = rng.sample(range(len(perm)), 2)
                        p2[i], p2[j] = p2[j], p2[i]
                    cost, _ = self.simulate(p2, f2)
                    if cost <= cur:
                        cur, perm, flip = cost, p2, f2
                if cur < best[0]:
                    best = (cur, perm, flip)
        _, lane = self.simulate(best[1], best[2])
        used = sorted(set(lane.values()))
        remap = {l: k for k, l in enumerate(used)}
        return {ni: remap[l] for ni, l in lane.items()}


# ---------------------------------------------------------------- object colours
def _bgr(hexc: str) -> tuple[int, int, int]:
    return int(hexc[5:7], 16), int(hexc[3:5], 16), int(hexc[1:3], 16)


def object_colors(graph: StateGraph) -> dict[int, tuple[int, int, int]]:
    """BGR colour of each object, as drawn in the graph SVG (one node per object)."""
    return {n.obj_id: _bgr(PALETTE[n.idx % len(PALETTE)]) for n in _Model(graph).nodes}


# ---------------------------------------------------------------- thumbnails
def _thumbnail(graph: StateGraph, frame_names: list[str], frame: int, colors: dict[int, str],
               width: int) -> tuple[str, int] | None:
    if frame >= len(frame_names):
        return None
    img = cv2.imread(os.path.join(graph.video_path, frame_names[frame]))
    if img is None:
        return None
    img = graph.overlay_masks(img, frame, {oid: _bgr(c) for oid, c in colors.items()}, THUMB_ALPHA)
    h, w = img.shape[:2]
    # render at 2x for sharpness on high-dpi screens / print
    height = round(width * h / w)
    small = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), height


# ---------------------------------------------------------------- rendering
def render_svg(graph: StateGraph, out_path: str, thumbnails: bool = True) -> str:
    m = _Model(graph)
    nodes, cols = m.nodes, m.columns
    lanes = _Lanes(m)
    lane_of = lanes.solve()
    main_in = lanes.main_in
    for n in nodes:
        n.lane = lane_of.get(n.idx, 0)
        n.color = PALETTE[n.idx % len(PALETTE)]
        n.lines = wrap(n.name, bold=n.final)
    n_lanes = max((n.lane for n in nodes), default=0) + 1

    # ---- x geometry
    def lbl_w(n: _Node) -> float:
        return max(text_width(s, FS, n.final) for s in n.lines)

    w_node = max((lbl_w(n) for n in nodes if n.kind != "root"), default=0.0)
    w_root = max((lbl_w(n) for n in nodes if n.kind == "root"), default=0.0)
    w_act = max((text_width(e.label, FS_ACT) for e in m.events), default=0.0)
    gap_jn = max(64.0, w_node / 2 + w_act / 2 + 6)                    # junction -> node
    step = max(STEP_MIN, gap_jn + w_node / 2 + w_act / 2 + PAD, w_node + 2 * PAD)
    thumb_w = int(step - THUMB_GAP)

    frame_names: list[str] = []
    if thumbnails and graph.video_path and os.path.isdir(graph.video_path):
        frame_names = sorted(os.listdir(graph.video_path))
    show_thumbs = bool(frame_names)

    x0 = w_root + R + 8 + PAD
    if show_thumbs:
        x0 = max(x0, thumb_w / 2 + 2)
    xn = lambda c: x0 + c * step                 # node x of column c
    xj = lambda c: xn(c) - gap_jn                # junction x of column c

    # ---- thumbnails (computed first: they set the top margin)
    thumbs: list[tuple[str, int] | None] = [None] * len(cols)
    if show_thumbs:
        for c, col in enumerate(cols):
            alive = m.alive_at[c]
            colors = {oid: PALETTE[ni % len(PALETTE)] for oid, ni in alive.items()}
            thumbs[c] = _thumbnail(graph, frame_names, col.frame, colors, thumb_w)
    thumb_h = max((t[1] for t in thumbs if t), default=0)
    top = PAD + (thumb_h + 14 if show_thumbs else 0) + 24   # room for the action labels above the first lane

    # ---- y geometry: lanes get extra room where a name below a node would sit on a busy lane
    def px_span(n: _Node) -> tuple[float, float]:
        s = xj(n.col) if n.kind == "event" else xn(n.col) - R
        e = xj(n.death) + SQ / 2 if n.death is not None else xn(n.col) + R_FINAL
        return s, e

    on_lane: dict[int, list[_Node]] = defaultdict(list)
    for n in nodes:
        on_lane[n.lane].append(n)
    gaps = [LANE] * max(n_lanes - 1, 0)
    for l in range(n_lanes - 1):
        for n in on_lane[l]:
            if n.kind == "root":
                for n2 in on_lane[l + 1]:
                    if n2.kind == "root":
                        gaps[l] = max(gaps[l], (len(n.lines) + len(n2.lines)) * LH / 2 + 4)
                continue
            half = lbl_w(n) / 2 + 4
            a, b = xn(n.col) - half, xn(n.col) + half
            if any(s < b and e > a for s, e in (px_span(n2) for n2 in on_lane[l + 1])):
                gaps[l] = max(gaps[l], R_FINAL + 6 + len(n.lines) * LH + 8)
    lane_y = [top]
    for g in gaps:
        lane_y.append(lane_y[-1] + g)
    y = lambda n: lane_y[n.lane]

    last_lane_lines = max((len(n.lines) for n in on_lane[n_lanes - 1] if n.kind != "root"), default=0)
    height = lane_y[-1] + max(R_FINAL + 6 + last_lane_lines * LH, LH) + PAD
    width = xn(len(cols) - 1) + max(w_node / 2, R_FINAL + 4, thumb_w / 2 if show_thumbs else 0) + PAD

    # ---- drawing
    under: list[str] = []      # dashed connectors
    lines: list[str] = []      # lanes and connectors
    marks: list[str] = []      # arrows, junctions, nodes
    texts: list[str] = []
    top_layer: list[str] = []  # thumbnails

    def path(d: str) -> None:
        lines.append(f'<path d="{d}"/>')

    def arrow(x_tip: float, yy: float) -> None:
        marks.append(f'<path d="M{_f(x_tip)},{_f(yy)} l{_f(-ARROW)},{_f(-ARROW * 0.6)} '
                     f'v{_f(ARROW * 1.2)} z" fill="{INK}"/>')

    def text_block(x: float, y_first: float, ls: list[str], anchor: str, bold: bool, title: str) -> None:
        weight = ' font-weight="650"' if bold else ""
        spans = "".join(f'<tspan x="{_f(x)}" dy="{0 if k == 0 else _f(LH)}">{escape(s)}</tspan>'
                        for k, s in enumerate(ls))
        texts.append(f'<text x="{_f(x)}" y="{_f(y_first)}" text-anchor="{anchor}"{weight}>'
                     f'<title>{escape(title)}</title>{spans}</text>')

    def elbow_in(x_from: float, y_from: float, x_v: float, y_to: float) -> str:
        """horizontal on y_from up to x_v, then vertical to y_to (rounded corner)"""
        sg = 1 if y_to > y_from else -1
        r = min(CORNER, abs(y_to - y_from) / 2)
        return (f"M{_f(x_from)},{_f(y_from)} H{_f(x_v - r)} Q{_f(x_v)},{_f(y_from)} {_f(x_v)},{_f(y_from + sg * r)} "
                f"V{_f(y_to)}")

    def elbow_out(x_v: float, y_from: float, y_to: float, x_to: float) -> str:
        """vertical on x_v from y_from to y_to, then horizontal to x_to (rounded corner)"""
        sg = 1 if y_to > y_from else -1
        r = min(CORNER, abs(y_to - y_from) / 2)
        return (f"M{_f(x_v)},{_f(y_from)} V{_f(y_to - sg * r)} Q{_f(x_v)},{_f(y_to)} {_f(x_v + r)},{_f(y_to)} "
                f"H{_f(x_to)}")

    def s_curve(xa: float, ya: float, xb: float, yb: float) -> str:
        k = (xb - xa) * 0.55
        return f"C{_f(xa + k)},{_f(ya)} {_f(xb - k)},{_f(yb)} {_f(xb)},{_f(yb)}"

    def node_title(n: _Node) -> str:
        f0, f1 = n.frames
        return f"{n.name} (id {n.obj_id}, frames {f0}-{f1})"

    def junction_lane(c: int) -> int:
        ev = cols[c].event
        assert ev is not None
        mi = main_in.get(c)
        return nodes[mi].lane if mi is not None else nodes[ev.main_out].lane

    def smooth(c: int, ni: int, ends: list[int]) -> bool:
        """S curve (instead of an elbow) for a connector to the adjacent lane that is alone on its side"""
        jl, l = junction_lane(c), nodes[ni].lane
        return abs(l - jl) == 1 and sum((nodes[o].lane > jl) == (l > jl) for o in ends if nodes[o].lane != jl) == 1

    # lanes leaving each node, towards the junction that consumes it
    for n in nodes:
        if n.death is None:
            continue
        x_a, ya = xn(n.col), y(n)
        ev = cols[n.death].event
        assert ev is not None
        jx = xj(n.death)
        mi = main_in.get(n.death)
        jy = lane_y[nodes[mi].lane] if mi is not None else ya
        if n.idx == mi or abs(jy - ya) < 1e-6:
            path(f"M{_f(x_a)},{_f(ya)} H{_f(jx - SQ / 2)}")
        elif smooth(n.death, n.idx, ev.inputs):
            xs = jx - SQ / 2 - S_CURVE
            path(f"M{_f(x_a)},{_f(ya)} H{_f(xs)} " + s_curve(xs, ya, jx - SQ / 2, jy))
        else:
            path(elbow_in(x_a, ya, jx, jy + (SQ / 2 if ya > jy else -SQ / 2)))

    # junctions, arrows to the results, action labels
    for c, col in enumerate(cols):
        ev = col.event
        if ev is None:
            continue
        jx = xj(c)
        mi = main_in.get(c)
        jl = nodes[mi].lane if mi is not None else nodes[ev.main_out].lane
        jy = lane_y[jl]
        up = down = False
        for o in ev.outputs:
            n = nodes[o]
            r = R_FINAL if n.final else R
            x_tip = xn(c) - r - (1.5 if n.final else 0.5)
            if n.lane == jl:
                path(f"M{_f(jx + SQ / 2)},{_f(jy)} H{_f(x_tip - ARROW + 1)}")
            elif smooth(c, o, ev.outputs):
                xs = jx + SQ / 2
                path(f"M{_f(xs)},{_f(jy)} " + s_curve(xs, jy, xs + S_CURVE, y(n)) + f" H{_f(x_tip - ARROW + 1)}")
            else:
                path(elbow_out(jx, jy + (SQ / 2 if y(n) > jy else -SQ / 2), y(n), x_tip - ARROW + 1))
            arrow(x_tip, y(n))
            up |= n.lane < jl
            down |= n.lane > jl
        for i in ev.inputs:
            up |= nodes[i].lane < jl
            down |= nodes[i].lane > jl
        marks.append(f'<rect x="{_f(jx - SQ / 2)}" y="{_f(jy - SQ / 2)}" width="{_f(SQ)}" height="{_f(SQ)}" '
                     f'rx="2.5" fill="{INK}"><title>{escape(ev.label)} (frame {ev.frame})</title></rect>')
        if not up:
            lx, ly, anchor = jx, jy - SQ / 2 - 6, "middle"
        elif not down:
            lx, ly, anchor = jx, jy + SQ / 2 + FS_ACT + 2, "middle"
        else:
            lx, ly, anchor = jx + SQ / 2 + 4, jy - 6, "start"
        texts.append(f'<text x="{_f(lx)}" y="{_f(ly)}" text-anchor="{anchor}" font-style="italic" '
                     f'font-size="{FS_ACT}" fill="{TEXT_2}">{escape(ev.label)}</text>')

    # nodes and their names
    for n in nodes:
        x, yy = xn(n.col), y(n)
        if n.final:
            marks.append(f'<circle cx="{_f(x)}" cy="{_f(yy)}" r="{_f(R_FINAL)}" fill="{n.color}" stroke="{INK}" '
                         f'stroke-width="2"><title>{escape(node_title(n))}</title></circle>')
        else:
            marks.append(f'<circle cx="{_f(x)}" cy="{_f(yy)}" r="{_f(R)}" fill="{n.color}">'
                         f'<title>{escape(node_title(n))}</title></circle>')
        if n.kind == "root":
            y_first = yy - (len(n.lines) - 1) * LH / 2 + FS * 0.36
            text_block(x - R - 8, y_first, n.lines, "end", n.final, node_title(n))
        else:
            text_block(x, yy + (R_FINAL if n.final else R) + 4 + FS * 0.8, n.lines, "middle", n.final, node_title(n))

    # thumbnails, linked to the new nodes of their column
    for c, col in enumerate(cols):
        t = thumbs[c]
        if t is None:
            continue
        href, th = t
        tx, ty = xn(c) - thumb_w / 2, PAD + (thumb_h - th)
        new = col.event.outputs if col.event is not None else col.appear
        if c > 0 and new:
            y_end = min(y(nodes[o]) for o in new) - R_FINAL - 3
            under.append(f'<line x1="{_f(xn(c))}" y1="{_f(ty + th + 1)}" x2="{_f(xn(c))}" y2="{_f(y_end)}"/>')
        cid = f"thumb-clip-{c}"
        top_layer.append(
            f'<clipPath id="{cid}"><rect x="{_f(tx)}" y="{_f(ty)}" width="{thumb_w}" height="{th}" rx="4"/></clipPath>'
            f'<image xlink:href="{href}" x="{_f(tx)}" y="{_f(ty)}" width="{thumb_w}" height="{th}" '
            f'preserveAspectRatio="none" clip-path="url(#{cid})"><title>frame {col.frame}</title></image>')
        # t_i badge
        bw, bh = 21.0, 17.0
        bx, by = tx + thumb_w - bw - 4, ty + th - bh - 4
        top_layer.append(
            f'<rect x="{_f(bx)}" y="{_f(by)}" width="{_f(bw)}" height="{_f(bh)}" rx="4" fill="{INK}" '
            f'fill-opacity="0.9"/><text x="{_f(bx + bw / 2 - 1)}" y="{_f(by + bh - 5)}" text-anchor="middle" '
            f'font-size="11" font-style="italic" font-weight="700" fill="#ffffff">t<tspan font-size="8" '
            f'dy="2.5">{c}</tspan></text>')

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {_f(width)} {_f(height)}" '
            f'width="{_f(width)}" height="{_f(height)}" font-family="{FONT}" font-size="{FS}" fill="{TEXT}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<g stroke="{DASH}" stroke-width="1.2" stroke-dasharray="3 3">', *under, "</g>",
        f'<g fill="none" stroke="{INK}" stroke-width="{LW}" stroke-linecap="round" stroke-linejoin="round">',
        *lines, "</g>",
        "<g>", *marks, "</g>",
        "<g>", *texts, "</g>",
        "<g>", *top_layer, "</g>",
        "</svg>",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    return out_path
