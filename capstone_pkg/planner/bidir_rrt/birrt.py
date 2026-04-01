from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import random

import numpy as np

from capstone_pkg.collision_check.collision import get_self_collision_checker
from capstone_pkg.collision_check.edge_collision import EdgeCollisionChecker
from .joint_limits import get_joint_limits


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int]  # index in nodes list


class Tree:
    def __init__(self, root_q: np.ndarray):
        self.nodes: List[Node] = [Node(root_q.copy(), None)]

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(Node(q.copy(), parent))
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        # brute-force (OK for few thousand nodes)
        qs = np.stack([n.q for n in self.nodes], axis=0)
        d = np.linalg.norm(qs - q[None, :], axis=1)
        return int(np.argmin(d))

    def path_to_root(self, idx: int) -> List[np.ndarray]:
        out=[]
        cur=idx
        while cur is not None:
            out.append(self.nodes[cur].q)
            cur=self.nodes[cur].parent
        out.reverse()
        return out


def _steer(q_from: np.ndarray, q_to: np.ndarray, step: float, active_idx: List[int]) -> np.ndarray:
    q_new = q_from.copy()
    diff = q_to[active_idx] - q_from[active_idx]
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return q_new
    scale = min(1.0, step / dist)
    q_new[active_idx] = q_from[active_idx] + diff * scale
    return q_new


def _sample(lowers: np.ndarray, uppers: np.ndarray, base: np.ndarray, active_idx: List[int], goal: np.ndarray, goal_bias: float) -> np.ndarray:
    q = base.copy()
    if random.random() < goal_bias:
        q[active_idx] = goal[active_idx]
        return q
    r = np.random.uniform(lowers, uppers)
    q[active_idx] = r
    return q


def plan_birrt_jointspace(
    *,
    robot_yml: str,
    q_start: List[float],
    q_goal: List[float],
    active_joint_names: List[str],
    cspace_joint_names: List[str],
    cpu: bool = False,
    step: float = 0.15,
    max_iters: int = 100000,
    goal_bias: float = 0.10,
    connect_threshold: float = 0.20,
    world_yml: Optional[str] = None,
) -> Tuple[bool, List[List[float]]]:
    """Bi-directional RRT in joint-space (full cspace vector), expanding only selected arm joints.
    Returns (success, path) where path is list of q vectors in cspace order.
    """
    q0 = np.asarray(q_start, dtype=np.float32)
    qg = np.asarray(q_goal, dtype=np.float32)
    if q0.shape != qg.shape:
        raise ValueError("q_start and q_goal must have same dimension")
    dim = int(q0.shape[0])

    # active indices in full cspace vector
    name_to_idx = {n:i for i,n in enumerate(cspace_joint_names)}
    active_idx = [name_to_idx[n] for n in active_joint_names if n in name_to_idx]
    if not active_idx:
        raise ValueError("active_joint_names not found in cspace_joint_names")

    # joint limits for active joints (best-effort)
    jl = get_joint_limits(robot_yml, active_joint_names, cpu=cpu)
    lowers = np.asarray(jl.lower, dtype=np.float32)
    uppers = np.asarray(jl.upper, dtype=np.float32)

    # collision checkers
    sc = get_self_collision_checker(robot_yml, cpu=cpu, world_yml=world_yml)
    ec = EdgeCollisionChecker(robot_yml, cpu=cpu, world_yml=world_yml)

    # validate endpoints
    in0, _, _ = sc.check_single(q0.tolist())
    ing, _, _ = sc.check_single(qg.tolist())
    if in0:
        return False, []
    if ing:
        return False, []

    Ta = Tree(q0)
    Tb = Tree(qg)

    base = q0.copy()
    for it in range(max_iters):
        # swap trees every iter to balance
        if it % 2 == 0:
            Tnear, Tother = Ta, Tb
            q_other_root = qg
        else:
            Tnear, Tother = Tb, Ta
            q_other_root = q0

        q_rand = _sample(lowers, uppers, base, active_idx, qg if Tnear is Ta else q0, goal_bias)
        idx_near = Tnear.nearest(q_rand)
        q_near = Tnear.nodes[idx_near].q

        q_new = _steer(q_near, q_rand, step, active_idx)

        # state collision
        in_new, _, _ = sc.check_single(q_new.tolist())
        if in_new:
            continue
        # edge collision
        e = ec.check_edge(q_near.tolist(), q_new.tolist(), return_first_hit=False)
        if e.edge_in_collision:
            continue

        idx_new = Tnear.add(q_new, idx_near)

        # try connect to other tree
        idx_other_near = Tother.nearest(q_new)
        q_other = Tother.nodes[idx_other_near].q
        if float(np.linalg.norm(q_other[active_idx] - q_new[active_idx])) < connect_threshold:
            # validate connecting edge both directions
            e2 = ec.check_edge(q_new.tolist(), q_other.tolist(), return_first_hit=False)
            if not e2.edge_in_collision:
                # build final path
                if Tnear is Ta:
                    path_a = Ta.path_to_root(idx_new)
                    path_b = Tb.path_to_root(idx_other_near)
                    path_b.reverse()  # from connect -> goal
                    full = path_a + path_b[1:]
                else:
                    # Tnear is Tb
                    path_b = Tb.path_to_root(idx_new)
                    path_a = Ta.path_to_root(idx_other_near)
                    path_b.reverse()  # from connect -> goal? careful
                    full = path_a + path_b[1:]
                return True, [q.astype(float).tolist() for q in full]

    return False, []
