"""
rhythm_tree.py
==============
Beat-cell tree data structures and the home-grid warp engine.

Architecture
------------
Home grid
    The meter-natural flat division (e.g. rhythm_division=16 in 4/4) defines N
    evenly-spaced rational anchor positions 0/N … N/N across one bar.  These are
    the only positions where warp parameters (swing, pocket, rubato) are *defined*.

WarpCurve
    A piecewise mapping  Fraction(bar position) → float(warped bar fraction).
    Built once per render from swing / pocket / rubato params; all other code just
    calls warp(frac).  Between anchors the curve interpolates using a pluggable
    interpolator (linear by default; cosine or cubic available).

    Float / irrational meters
        meter_num may be 4.5, 3.5, pi, …  The integer beats fill the grid
        normally.  The fractional remainder is *not* a new concept — it is absorbed
        as extra duration distributed proportionally to the local warp derivative at
        each home anchor.  Anchors that are already stretching time absorb more of
        the leftover; anchors that are compressing absorb nothing.  The warp curve
        is then renormalized so it still maps [0, 1) → [0, 1).  There is no pause
        object, no rest token — the time wells just get a little deeper.

BeatNode
    One cell in the rhythm grid.  Leaf nodes (children=[]) are the actual
    schedulable events.  Interior nodes are structural grouping only (subdivided
    beat cells).

    position : Fraction   — bar-fraction onset, 0 ≤ position < 1
    duration : Fraction   — bar-fraction span
    on       : bool       — whether this leaf fires a note
    vel      : float
    art      : int        — 0=normal 1=staccato 2=legato 3=drone
    children : list       — non-empty ↔ subdivided; children partition [position, position+duration)

Flat / tree compatibility
    RhythmPattern continues to work unchanged via flat_leaves() which walks the
    tree and returns all leaf nodes in position order.  Legacy code that reads
    pat.steps[i] is served by steps_from_tree() which projects leaves back to
    the legacy integer-indexed flat array.

    The tree is *optional* on a per-pattern basis.  When beat_nodes is None the
    pattern is fully described by its steps/vel/art flat lists as before.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Interpolators
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _cosine_interp(a: float, b: float, t: float) -> float:
    mu = (1.0 - math.cos(t * math.pi)) * 0.5
    return a + (b - a) * mu


def _cubic_interp(y0: float, y1: float, y2: float, y3: float, t: float) -> float:
    """Catmull-Rom through y1..y2."""
    a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    a1 =       y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    a2 = -0.5 * y0             + 0.5 * y2
    a3 = y1
    return a0 * t**3 + a1 * t**2 + a2 * t + a3


INTERPOLATORS: dict[str, Callable] = {
    "linear":  _lerp,
    "cosine":  _cosine_interp,
}


# ---------------------------------------------------------------------------
# WarpCurve
# ---------------------------------------------------------------------------

@dataclass
class WarpCurve:
    """
    Piecewise warp mapping defined at N+1 home-grid anchor points.

    Anchor i sits at bar-fraction i/home_div.
    warped[i] is the *warped* bar-fraction at that anchor.

    Parameters
    ----------
    home_div        : int    — number of home-grid steps (e.g. 16)
    swing           : float  — fraction of step_s to push odd-indexed anchors forward
    pocket          : float  — uniform shift in beats (applied as fraction of bar)
    rubato_shape    : str    — "off"|"sine"|"troughs"|"slow_go"|"go_slow"
    rubato_amount   : float  — 0..0.95
    meter_num       : float  — possibly fractional/irrational; integer part = full beats
    interpolator    : str    — "linear"|"cosine"
    """
    home_div:      int   = 16
    swing:         float = 0.0
    pocket:        float = 0.0
    rubato_shape:  str   = "off"
    rubato_amount: float = 0.0
    meter_num:     float = 4.0
    beats_per_bar: float = 4.0
    interpolator:  str   = "linear"
    frac_beat_mode: str  = "warp"   # "warp" = absorb leftover, "grid" = skip

    # Computed on build — do not set manually
    anchors:  list[float] = field(default_factory=list)   # straight bar-fracs [0..1]
    warped:   list[float] = field(default_factory=list)   # warped bar-fracs

    def __post_init__(self) -> None:
        self._build()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _rubato_phase(self, u: float) -> float:
        u   = max(0.0, min(1.0, u))
        amt = max(0.0, min(0.95, self.rubato_amount))
        if amt <= 1e-9 or self.rubato_shape == "off":
            return u
        shape = self.rubato_shape
        if shape == "sine":
            return u + amt * math.sin(2.0 * math.pi * u) / (2.0 * math.pi)
        if shape == "troughs":
            return u + amt * math.sin(4.0 * math.pi * u) / (4.0 * math.pi)
        if shape == "slow_go":
            return (1.0 - amt) * u + amt * (u * u)
        if shape == "go_slow":
            return (1.0 - amt) * u + amt * (1.0 - (1.0 - u) ** 2)
        return u

    def _build(self) -> None:
        """Compute anchors[] and warped[] with float-meter remainder absorption."""
        n       = max(1, self.home_div)
        step_s  = 1.0 / n   # bar-fraction per step (straight time)
        pocket_frac = self.pocket / max(1e-9, self.beats_per_bar)

        # 1. Build raw warped values at each anchor (swing + pocket + rubato)
        raw: list[float] = []
        for i in range(n + 1):
            u = i / n
            # Swing: push odd-indexed steps forward
            if i % 2 == 1 and i < n:
                u_swung = u + self.swing * step_s
            else:
                u_swung = u
            # Pocket: uniform shift
            u_pkt = u_swung + pocket_frac
            # Rubato
            w = self._rubato_phase(max(0.0, min(1.0, u_pkt)))
            raw.append(w)

        # 2. Float-meter remainder absorption
        #    full_beats = floor(meter_num), leftover = meter_num - full_beats
        #    The leftover is extra duration; it is distributed proportionally to
        #    how much each adjacent anchor pair is already being stretched by the
        #    warp (local derivative > 1 = stretching).  Compressed regions absorb
        #    nothing.  Result is renormalized to [0, 1].
        #
        #    In "grid" mode the grid already contains a fractional beat cell, so
        #    the warp must NOT absorb the leftover (it would double-count).
        full_beats = math.floor(max(0.125, self.meter_num) + 1e-9)
        leftover   = max(0.0, self.meter_num - float(full_beats))

        if leftover > 1e-6 and n > 0 and self.frac_beat_mode != "grid":
            # Local derivative at each inter-anchor gap (straight = step_s)
            derivs = []
            for i in range(n):
                d = (raw[i + 1] - raw[i]) / step_s   # > 1 = stretched, < 1 = compressed
                derivs.append(max(0.0, d - 1.0))     # only positive (stretch) portion

            total_stretch = sum(derivs)
            if total_stretch > 1e-9:
                weights = [d / total_stretch for d in derivs]
            else:
                # Warp is flat — distribute leftover uniformly
                weights = [1.0 / n] * n

            # Apply: shift each anchor by its accumulated weight of leftover
            # leftover is expressed as a fraction of the total bar duration
            leftover_frac = leftover / max(1e-9, self.meter_num)
            delta = [w * leftover_frac for w in weights]
            # delta[i] is extra duration absorbed into gap i; shift anchors i+1..n
            cum = 0.0
            adjusted = [raw[0]]
            for i in range(n):
                cum += delta[i]
                adjusted.append(raw[i + 1] + cum)
            raw = adjusted

        # 3. Renormalize so first=0, last=1 (preserve monotonicity)
        lo, hi = raw[0], raw[-1]
        span   = hi - lo
        if span < 1e-12:
            # Degenerate — just use linear
            raw = [i / n for i in range(n + 1)]
            lo, span = 0.0, 1.0
        self.anchors = [i / n for i in range(n + 1)]
        self.warped  = [(v - lo) / span for v in raw]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warp(self, frac: "Fraction | float") -> float:
        """
        Map a bar-fraction position through the warp curve.

        The fraction may be any rational, including sub-divisions deeper than
        home_div (triplets, quintuplets, etc.).  The curve interpolates between
        the two flanking home anchors.

        Returns a float warped bar-fraction in [0, 1].
        """
        f = float(frac)
        f = max(0.0, min(1.0, f))
        n = len(self.anchors) - 1
        if n <= 0:
            return f

        # Binary-search for bracketing anchors
        idx = int(f * n)
        idx = max(0, min(n - 1, idx))
        a0  = self.anchors[idx]
        a1  = self.anchors[idx + 1]
        if a1 <= a0:
            return self.warped[idx]

        t = (f - a0) / (a1 - a0)
        t = max(0.0, min(1.0, t))

        w0 = self.warped[idx]
        w1 = self.warped[idx + 1]

        if self.interpolator == "cosine":
            return _cosine_interp(w0, w1, t)
        return _lerp(w0, w1, t)

    def warp_to_seconds(self, frac: "Fraction | float", bar_s: float) -> float:
        """Convert a bar-fraction position to real onset seconds."""
        return self.warp(frac) * bar_s

    def derivative(self, frac: "Fraction | float") -> float:
        """
        Local time-stretch factor at *frac* (warped_duration / straight_duration).
        Values > 1 mean the warp is stretching time here (a 'pocket').
        """
        f  = float(frac)
        n  = len(self.anchors) - 1
        if n <= 0:
            return 1.0
        idx = max(0, min(n - 1, int(f * n)))
        da  = self.anchors[idx + 1] - self.anchors[idx]
        dw  = self.warped[idx + 1]  - self.warped[idx]
        if da < 1e-12:
            return 1.0
        return dw / da


# ---------------------------------------------------------------------------
# BeatNode
# ---------------------------------------------------------------------------

@dataclass
class BeatNode:
    """
    One cell in the rhythm-grid tree.

    Leaf nodes (children=[]) are schedulable.
    Interior nodes group their children — they carry no on/vel/art of their own
    (those fields are ignored for interior nodes during scheduling).

    position : Fraction  — onset as bar-fraction [0, 1)
    duration : Fraction  — span as bar-fraction > 0
    on       : bool
    vel      : float     — 0.0..1.0
    art      : int       — 0=normal 1=staccato 2=legato 3=drone
    group    : int       — trigger group color index; 0=ungrouped, 1..8=colored.
                           Adjacent ON leaves with the same non-zero group merge
                           into a single sustained event during scheduling.
    children : list[BeatNode]
    node_id  : str       — stable uuid hex for grouping / merge tracking
    """
    position:  Fraction           = field(default_factory=lambda: Fraction(0))
    duration:  Fraction           = field(default_factory=lambda: Fraction(1, 16))
    on:        bool               = False
    vel:       float              = 1.0
    art:       int                = 0
    group:     int                = 0
    children:  List["BeatNode"]   = field(default_factory=list)
    node_id:   str                = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaves(self) -> Iterator["BeatNode"]:
        """Yield all leaf descendants in position order."""
        if self.is_leaf():
            yield self
        else:
            for child in sorted(self.children, key=lambda c: c.position):
                yield from child.leaves()

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    # ------------------------------------------------------------------
    # Subdivision
    # ------------------------------------------------------------------

    def subdivide(self, n: int) -> None:
        """
        Split this leaf node into *n* equal children.
        The first child inherits on/vel/art; others default to off.
        The node becomes interior.
        """
        if n < 2:
            return
        if not self.is_leaf():
            raise ValueError("subdivide() called on an interior node")
        child_dur = self.duration / n
        for k in range(n):
            child = BeatNode(
                position = self.position + child_dur * k,
                duration = child_dur,
                on       = self.on if k == 0 else False,
                vel      = self.vel,
                art      = self.art,
                group    = self.group,
            )
            self.children.append(child)
        # Interior node: clear own on/vel/art/group (no longer a leaf)
        self.on    = False
        self.vel   = 1.0
        self.art   = 0
        self.group = 0

    # ------------------------------------------------------------------
    # Merge helpers (called from parent)
    # ------------------------------------------------------------------

    def collapse(self) -> None:
        """
        Flatten children back into this node (undo subdivision).
        Inherits on/vel/art from first leaf child.
        """
        first = next(self.leaves(), None)
        if first is not None:
            self.on    = first.on
            self.vel   = first.vel
            self.art   = first.art
            self.group = first.group
        self.children.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {
            "node_id":  self.node_id,
            "position": [self.position.numerator, self.position.denominator],
            "duration": [self.duration.numerator, self.duration.denominator],
            "on":       self.on,
            "vel":      self.vel,
            "art":      self.art,
            "group":    self.group,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BeatNode":
        pos_r = d.get("position", [0, 1])
        dur_r = d.get("duration", [1, 16])
        node  = cls(
            position = Fraction(int(pos_r[0]), max(1, int(pos_r[1]))),
            duration = Fraction(int(dur_r[0]), max(1, int(dur_r[1]))),
            on       = bool(d.get("on", False)),
            vel      = float(d.get("vel", 1.0)),
            art      = int(d.get("art", 0)),
            group    = int(d.get("group", 0)),
            node_id  = str(d.get("node_id", uuid.uuid4().hex[:12])),
        )
        for cd in d.get("children", []):
            node.children.append(BeatNode.from_dict(cd))
        return node


# ---------------------------------------------------------------------------
# BeatTree  — one bar of beat nodes (corresponds to one RhythmPattern bar)
# ---------------------------------------------------------------------------

@dataclass
class BeatTree:
    """
    A flat list of top-level BeatNodes that together span one bar.

    nodes covers [0, 1) without overlap or gap at the home-division level.
    Nodes may be subdivided (children) but the top-level list always reflects
    the home grid.
    """
    home_div: int              = 16
    nodes:    List[BeatNode]   = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new_uniform(
            cls,
            home_div: int = 16,
            default_on:  bool  = False,
            default_vel: float = 1.0,
    ) -> "BeatTree":
        """Create a flat uniform tree — no subdivisions, all leaves identical."""
        n    = max(1, home_div)
        tree = cls(home_div=n)
        for i in range(n):
            tree.nodes.append(BeatNode(
                position = Fraction(i, n),
                duration = Fraction(1, n),
                on       = default_on,
                vel      = default_vel,
            ))
        return tree

    @classmethod
    def from_flat(
            cls,
            steps: list,
            vel:   list,
            art:   list,
            home_div: int = 16,
    ) -> "BeatTree":
        """Build a flat (no subdivision) tree from legacy step arrays."""
        n    = max(1, home_div)
        tree = cls(home_div=n)
        for i in range(n):
            step_on  = bool(steps[i]) if i < len(steps) else False
            step_vel = float(vel[i])  if i < len(vel)   else 1.0
            step_art = int(art[i])    if i < len(art)    else 0
            tree.nodes.append(BeatNode(
                position = Fraction(i, n),
                duration = Fraction(1, n),
                on       = step_on,
                vel      = step_vel,
                art      = step_art,
            ))
        return tree

    # ------------------------------------------------------------------
    # Flat projection (for legacy code)
    # ------------------------------------------------------------------

    def flat_leaves(self) -> list[BeatNode]:
        """All leaf nodes sorted by position."""
        leaves = []
        for node in self.nodes:
            leaves.extend(node.leaves())
        return sorted(leaves, key=lambda n: n.position)

    def steps_array(self, div: int | None = None) -> list[bool]:
        """
        Project leaves onto an integer-indexed array of length *div*.
        Leaves that don't align exactly are snapped to the nearest index.
        Used only to bridge legacy code that reads pat.steps[i].
        """
        n    = div if div is not None else self.home_div
        arr  = [False] * n
        for leaf in self.flat_leaves():
            idx = int(round(float(leaf.position) * n)) % n
            arr[idx] = leaf.on
        return arr

    def vel_array(self, div: int | None = None) -> list[float]:
        n   = div if div is not None else self.home_div
        arr = [1.0] * n
        for leaf in self.flat_leaves():
            idx = int(round(float(leaf.position) * n)) % n
            arr[idx] = leaf.vel
        return arr

    def art_array(self, div: int | None = None) -> list[int]:
        n   = div if div is not None else self.home_div
        arr = [0] * n
        for leaf in self.flat_leaves():
            idx = int(round(float(leaf.position) * n)) % n
            arr[idx] = leaf.art
        return arr

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    def node_at(self, position: Fraction) -> BeatNode | None:
        """Return the top-level node whose span contains *position*."""
        for node in self.nodes:
            if node.position <= position < node.position + node.duration:
                return node
        return None

    def leaf_at(self, position: Fraction) -> BeatNode | None:
        """Return the deepest leaf that contains *position*."""
        top = self.node_at(position)
        if top is None:
            return None
        if top.is_leaf():
            return top
        for leaf in top.leaves():
            if leaf.position <= position < leaf.position + leaf.duration:
                return leaf
        return None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def subdivide_at(self, position: Fraction, n: int) -> BeatNode | None:
        """Subdivide the leaf containing *position* into *n* children."""
        leaf = self.leaf_at(position)
        if leaf is not None and leaf.is_leaf():
            leaf.subdivide(n)
        return leaf

    def collapse_at(self, position: Fraction) -> BeatNode | None:
        """Collapse the innermost interior node containing *position*."""
        # Walk down to find deepest interior node
        top = self.node_at(position)
        if top is None:
            return None

        def _deepest_interior(node: BeatNode) -> BeatNode | None:
            if node.is_leaf():
                return None
            for child in node.children:
                if child.position <= position < child.position + child.duration:
                    deeper = _deepest_interior(child)
                    return deeper if deeper is not None else node
            return node

        target = _deepest_interior(top)
        if target is not None:
            target.collapse()
        return target

    def merge_nodes(self, positions: list[Fraction]) -> BeatNode | None:
        """
        Merge a contiguous run of top-level nodes (identified by their positions)
        into a single node.  The first node absorbs the span; others are removed.
        """
        targets = sorted(
            [n for n in self.nodes if n.position in positions],
            key=lambda n: n.position,
        )
        if len(targets) < 2:
            return None
        # Verify contiguity
        for i in range(len(targets) - 1):
            if targets[i].position + targets[i].duration != targets[i + 1].position:
                return None   # not contiguous — refuse

        first = targets[0]
        total_dur = sum((t.duration for t in targets), Fraction(0))
        first.duration = total_dur
        first.children.clear()
        first.on  = any(t.on for t in targets)
        first.vel = max(t.vel for t in targets)
        first.art = targets[0].art

        for t in targets[1:]:
            self.nodes.remove(t)
        return first

    # ------------------------------------------------------------------
    # Structure synchronisation
    # ------------------------------------------------------------------

    def snap_structure_from(self, source: "BeatTree") -> None:
        """Restructure *self* to match *source*'s subdivision layout.

        Leaf data (on, vel, art, group) is preserved by positional coverage:
        when a source leaf has no exact match in the old tree, the old leaf
        whose span *covers* that position donates its values.  This means
        subdividing keeps old data, and collapsing inherits from the first
        covered leaf.
        """
        # Snapshot old leaves sorted by position for coverage lookup
        old_leaves = self.flat_leaves()

        def _covering(pos: Fraction) -> BeatNode | None:
            for lf in old_leaves:
                if lf.position <= pos < lf.position + lf.duration:
                    return lf
            return None

        def _clone(src: BeatNode) -> BeatNode:
            if src.is_leaf():
                donor = _covering(src.position)
                return BeatNode(
                    position = src.position,
                    duration = src.duration,
                    on       = donor.on       if donor else False,
                    vel      = donor.vel      if donor else 1.0,
                    art      = donor.art      if donor else 0,
                    group    = donor.group    if donor else 0,
                )
            node = BeatNode(position=src.position, duration=src.duration)
            for child in src.children:
                node.children.append(_clone(child))
            return node

        self.nodes.clear()
        self.home_div = source.home_div
        for src_node in source.nodes:
            self.nodes.append(_clone(src_node))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "home_div": self.home_div,
            "nodes":    [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeatTree":
        tree = cls(home_div=int(d.get("home_div", 16)))
        tree.nodes = [BeatNode.from_dict(nd) for nd in d.get("nodes", [])]
        return tree


# ---------------------------------------------------------------------------
# Scheduling helpers used by _build_rhythm_schedule
# ---------------------------------------------------------------------------

def iter_leaf_events(
        tree:     BeatTree,
        warp:     WarpCurve,
        bar_s:    float,
        abs_bar:  int,
) -> Iterator[tuple[BeatNode, float, float]]:
    """
    Yield (leaf_node, onset_seconds, straight_step_s) for every ON leaf in *tree*.

    onset_seconds  : absolute onset time (abs_bar warped through the warp curve)
    straight_step_s: the leaf's straight duration in seconds (un-warped), used
                     for gate calculations in the scheduler
    """
    bar_offset = abs_bar * bar_s
    for leaf in tree.flat_leaves():
        if not leaf.on:
            continue
        onset_frac    = leaf.position
        dur_frac      = leaf.duration
        onset_warped  = warp.warp(onset_frac)
        onset_s       = bar_offset + onset_warped * bar_s
        straight_step = float(dur_frac) * bar_s
        yield leaf, max(0.0, onset_s), straight_step


# Group color palette — 8 distinct colors, index 1..8.
# 0 = ungrouped (no color).  Used by both scheduler and UI.
GROUP_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),          # 0: unused / ungrouped
    (220, 80, 80),       # 1: red
    (80, 180, 220),      # 2: cyan
    (220, 180, 60),      # 3: gold
    (100, 220, 100),     # 4: green
    (200, 120, 220),     # 5: violet
    (220, 140, 70),      # 6: orange
    (80, 140, 220),      # 7: blue
    (220, 100, 180),     # 8: pink
]


def iter_grouped_events(
        tree:     BeatTree,
        warp:     WarpCurve,
        bar_s:    float,
        abs_bar:  int,
) -> Iterator[tuple[BeatNode, float, float]]:
    """
    Like iter_leaf_events but merges adjacent ON leaves that share a non-zero group.

    When consecutive ON leaves have the same group index (1..8), they fuse into
    one event: the first leaf's onset, with duration spanning the whole run.
    The yielded leaf is the *first* in the run (carries its vel/art).
    Ungrouped leaves (group=0) pass through individually as usual.

    Yields (first_leaf, onset_seconds, merged_duration_seconds).
    """
    bar_offset = abs_bar * bar_s
    leaves = tree.flat_leaves()  # sorted by position

    i = 0
    while i < len(leaves):
        leaf = leaves[i]
        if not leaf.on:
            i += 1
            continue

        onset_warped = warp.warp(leaf.position)
        onset_s      = bar_offset + onset_warped * bar_s

        if leaf.group == 0:
            # Ungrouped — single event
            straight_step = float(leaf.duration) * bar_s
            yield leaf, max(0.0, onset_s), straight_step
            i += 1
        else:
            # Grouped — merge consecutive ON leaves with same group
            merged_dur = leaf.duration
            j = i + 1
            while j < len(leaves):
                nxt = leaves[j]
                if nxt.on and nxt.group == leaf.group:
                    merged_dur += nxt.duration
                    j += 1
                else:
                    break
            # End-of-run warped position for duration calculation
            end_frac   = leaf.position + merged_dur
            end_warped = warp.warp(min(end_frac, Fraction(1)))
            if end_frac > Fraction(1):
                # Run extends beyond bar — use straight for the overflow
                merged_s = (end_warped - onset_warped) * bar_s + float(end_frac - 1) * bar_s
            else:
                merged_s = (end_warped - onset_warped) * bar_s
            yield leaf, max(0.0, onset_s), max(0.0, merged_s)
            i = j


def build_warp_curve(
        home_div:      int,
        swing:         float,
        pocket:        float,
        rubato_shape:  str,
        rubato_amount: float,
        meter_num:     float,
        beats_per_bar: float,
        interpolator:  str = "linear",
        frac_beat_mode: str = "warp",
) -> WarpCurve:
    """Convenience constructor — validates and returns a WarpCurve."""
    return WarpCurve(
        home_div      = max(1, int(home_div)),
        swing         = float(swing),
        pocket        = float(pocket),
        rubato_shape  = str(rubato_shape),
        rubato_amount = max(0.0, min(0.95, float(rubato_amount))),
        meter_num     = max(0.125, float(meter_num)),
        beats_per_bar = max(1e-6, float(beats_per_bar)),
        interpolator  = str(interpolator),
        frac_beat_mode = str(frac_beat_mode),
    )
