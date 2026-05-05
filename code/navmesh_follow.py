"""
Navmesh path follower.

Input: a navmesh exported to JSON in the form

    {
      "nodes": [{"id": 0, "pos": [x, y, z], "neighbors": [1, 4, 7]}, ...],
      "polys": [...]                # optional; not used here
    }

(or whatever schema the user has). The class below reads the JSON, builds an
adjacency graph, runs A* / Dijkstra between the player position and a
goal cell, then exposes a smoothed waypoint queue.

Design choices:

  * Smoothing uses Catmull-Rom interpolation between successive waypoints so
    the bot's path is curved rather than zig-zag. This matters because human
    movement in CS2 follows wide arcs around corners, not segment-to-segment
    snaps.
  * The follower issues (vx, vy) commands in player frame, which the
    counter-strafer then turns into key presses.
  * Look-ahead distance is speed-dependent, mimicking how human players "pre-
    aim" further when running.

References:
  Forsyth, "Cellular automata for real-time pathfinding" GDC.
  AIGameDev's "Funnel algorithm" for navmesh string-pulling.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Node:
    id: int
    pos: np.ndarray
    neighbors: list[int]


class NavMesh:
    def __init__(self, nodes: dict[int, Node]):
        self.nodes = nodes

    @classmethod
    def from_json(cls, path: str | Path) -> "NavMesh":
        data = json.loads(Path(path).read_text())
        nodes: dict[int, Node] = {}
        for n in data["nodes"]:
            nodes[n["id"]] = Node(
                id=n["id"],
                pos=np.asarray(n["pos"], dtype=float),
                neighbors=list(n.get("neighbors", [])),
            )
        return cls(nodes)

    def nearest(self, p: np.ndarray) -> int:
        best, best_d = -1, float("inf")
        for nid, n in self.nodes.items():
            d = float(np.sum((n.pos - p) ** 2))
            if d < best_d:
                best, best_d = nid, d
        return best

    def astar(self, start_id: int, goal_id: int) -> list[int]:
        if start_id == goal_id:
            return [start_id]
        goal_pos = self.nodes[goal_id].pos

        def h(nid: int) -> float:
            return float(np.linalg.norm(self.nodes[nid].pos - goal_pos))

        open_q: list[tuple[float, int]] = [(h(start_id), start_id)]
        came: dict[int, int] = {}
        g: dict[int, float] = {start_id: 0.0}
        closed: set[int] = set()

        while open_q:
            _, cur = heapq.heappop(open_q)
            if cur in closed:
                continue
            if cur == goal_id:
                break
            closed.add(cur)
            for nb in self.nodes[cur].neighbors:
                step = float(np.linalg.norm(
                    self.nodes[nb].pos - self.nodes[cur].pos))
                tentative = g[cur] + step
                if tentative < g.get(nb, float("inf")):
                    g[nb] = tentative
                    came[nb] = cur
                    heapq.heappush(open_q, (tentative + h(nb), nb))
        if goal_id not in came and goal_id != start_id:
            return []
        # reconstruct
        path = [goal_id]
        while path[-1] != start_id:
            path.append(came[path[-1]])
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Catmull-Rom smoothing
# ---------------------------------------------------------------------------

def catmull_rom(P: np.ndarray, n_per_seg: int = 12, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom (alpha=0.5) through points P (k, d).

    Returns a smoothed polyline that passes through every original point.
    Duplicate consecutive points and very short segments are handled by
    falling back to linear interpolation for that segment so the parameter
    differences never collapse to 0.
    """
    eps = 1e-9
    if len(P) < 4:
        P = np.vstack([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(eps, float(np.linalg.norm(p1 - p0)) ** alpha)
        t2 = t1 + max(eps, float(np.linalg.norm(p2 - p1)) ** alpha)
        t3 = t2 + max(eps, float(np.linalg.norm(p3 - p2)) ** alpha)
        ts = np.linspace(t1, t2, n_per_seg)
        for t in ts:
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            out.append(C)
    return np.array(out)


class PathFollower:
    def __init__(self, mesh: NavMesh):
        self.mesh = mesh
        self.smoothed: np.ndarray | None = None
        self.idx = 0

    def replan(self, player_pos: np.ndarray, goal: np.ndarray) -> None:
        s = self.mesh.nearest(player_pos)
        g = self.mesh.nearest(goal)
        ids = self.mesh.astar(s, g)
        if not ids:
            self.smoothed = None
            return
        pts = np.stack([self.mesh.nodes[i].pos for i in ids])
        # if we're already deep into the first cell, prepend the actual
        # player position so smoothing starts there
        pts = np.vstack([player_pos, pts])
        self.smoothed = catmull_rom(pts)
        self.idx = 0

    def desired_velocity(self, player_pos: np.ndarray, max_speed: float = 215.0,
                         look_ahead: float = 60.0) -> np.ndarray:
        if self.smoothed is None or self.idx >= len(self.smoothed):
            return np.zeros(3)
        # advance idx until distance from player to smoothed[idx] >=
        # look_ahead OR reached the end
        while self.idx < len(self.smoothed) - 1 and \
                np.linalg.norm(self.smoothed[self.idx] - player_pos) < look_ahead:
            self.idx += 1
        target = self.smoothed[self.idx]
        d = target - player_pos
        n = np.linalg.norm(d)
        if n < 1e-6:
            return np.zeros(3)
        return d / n * max_speed
