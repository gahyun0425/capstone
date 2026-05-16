from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


TRACE_SCHEMA_VERSION = 2
_LEVEL_ORDER = {"nodes": 0, "summary": 1, "debug": 2}


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _q_to_list(q: torch.Tensor | List[float]) -> List[float]:
    if torch.is_tensor(q):
        return q.detach().cpu().view(-1).tolist()
    return list(q)


def _value_list(v: torch.Tensor | Iterable[Any] | Any, *, to_bool: bool = False) -> List[Any]:
    if torch.is_tensor(v):
        vals = v.detach().cpu().view(-1).tolist()
    elif isinstance(v, (list, tuple)):
        vals = list(v)
    else:
        vals = [v]
    if to_bool:
        return [bool(x) for x in vals]
    return vals


def _q_rows_to_list(q: torch.Tensor | Iterable[Iterable[float]]) -> List[List[float]]:
    if torch.is_tensor(q):
        return q.detach().cpu().tolist()
    return [list(row) for row in q]


def _optional_value_list(v: torch.Tensor | Iterable[Any] | Any, n: int) -> List[Any]:
    if v is None:
        return [None for _ in range(n)]
    vals = _value_list(v)
    if len(vals) == 1 and n != 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError("optional trace input length mismatch")
    return vals


@dataclass
class PlanningTraceRecorder:
    planner: str
    level: str = "summary"
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.level not in _LEVEL_ORDER:
            raise ValueError(f"unsupported trace level: {self.level!r}")

    def wants(self, level: str) -> bool:
        return _LEVEL_ORDER.get(self.level, _LEVEL_ORDER["summary"]) >= _LEVEL_ORDER.get(level, _LEVEL_ORDER["summary"])

    def record_event(self, kind: str, **payload: Any) -> Dict[str, Any]:
        event = {"seq": len(self.events), "kind": str(kind)}
        event.update({str(k): _json_safe(v) for k, v in payload.items() if v is not None})
        self.events.append(event)
        return event

    def record_node(
        self,
        *,
        tree: str,
        batch_idx: int,
        node_idx: int,
        parent_idx: int,
        ts_id: int,
        q: torch.Tensor | List[float],
        is_proj_root: bool,
        iter_idx: Optional[int] = None,
        phase: Optional[str] = None,
        slot_idx: Optional[int] = None,
        escape_step: Optional[int] = None,
    ) -> None:
        if not self.wants("nodes"):
            return
        self.record_event(
            "node_add",
            tree=str(tree),
            batch_idx=int(batch_idx),
            node_idx=int(node_idx),
            parent_idx=int(parent_idx),
            ts_id=int(ts_id),
            is_proj_root=bool(is_proj_root),
            q=_q_to_list(q),
            iter=(None if iter_idx is None else int(iter_idx)),
            phase=phase,
            slot_idx=(None if slot_idx is None else int(slot_idx)),
            escape_step=(None if escape_step is None else int(escape_step)),
        )

    def record_nodes_batch(
        self,
        *,
        tree: str,
        batch_idx: torch.Tensor | Iterable[int] | int,
        node_idx: torch.Tensor | Iterable[int] | int,
        parent_idx: torch.Tensor | Iterable[int] | int,
        ts_id: torch.Tensor | Iterable[int] | int,
        q: torch.Tensor | Iterable[Iterable[float]],
        is_proj_root: torch.Tensor | Iterable[bool] | bool,
        iter_idx: torch.Tensor | Iterable[int] | int | None = None,
        phase: str | None = None,
        slot_idx: torch.Tensor | Iterable[int] | int | None = None,
        escape_step: int | None = None,
    ) -> None:
        if not self.wants("nodes"):
            return
        batch_list = _value_list(batch_idx)
        node_list = _value_list(node_idx)
        parent_list = _value_list(parent_idx)
        ts_list = _value_list(ts_id)
        proj_list = _value_list(is_proj_root, to_bool=True)
        q_list = _q_rows_to_list(q)

        n = len(node_list)
        if not (
            len(batch_list) == len(parent_list) == len(ts_list) == len(proj_list) == len(q_list) == n
        ):
            raise ValueError("record_nodes_batch inputs must have matching lengths")
        iter_list = _optional_value_list(iter_idx, n)
        slot_list = _optional_value_list(slot_idx, n)

        tree_name = str(tree)
        for i in range(n):
            self.record_event(
                "node_add",
                tree=tree_name,
                batch_idx=int(batch_list[i]),
                node_idx=int(node_list[i]),
                parent_idx=int(parent_list[i]),
                ts_id=int(ts_list[i]),
                is_proj_root=bool(proj_list[i]),
                q=list(q_list[i]),
                iter=(None if iter_list[i] is None else int(iter_list[i])),
                phase=phase,
                slot_idx=(None if slot_list[i] is None else int(slot_list[i])),
                escape_step=(None if escape_step is None else int(escape_step)),
            )

    def record_connection(
        self,
        *,
        iter_idx: int,
        batch_idx: int,
        idx_a: int,
        idx_b: int,
        tree_a: str = "start",
        tree_b: str = "goal",
    ) -> None:
        self.record_event(
            "connection",
            iter=int(iter_idx),
            batch_idx=int(batch_idx),
            idxA=int(idx_a),
            idxB=int(idx_b),
            treeA=str(tree_a),
            treeB=str(tree_b),
        )

    def record_connect_step(self, *, iter_idx: int, tree: str, **payload: Any) -> None:
        if self.wants("summary"):
            self.record_event("connect_step", iter=int(iter_idx), tree=str(tree), **payload)

    def record_escape_spawn_batch(self, *, iter_idx: int, tree: str, **payload: Any) -> None:
        if self.wants("summary"):
            self.record_event("escape_spawn", iter=int(iter_idx), tree=str(tree), **payload)

    def record_escape_result(self, *, iter_idx: int, tree: str, **payload: Any) -> None:
        if self.wants("summary"):
            self.record_event("escape_result", iter=int(iter_idx), tree=str(tree), **payload)

    def record_iter_summary(self, *, iter_idx: int, **payload: Any) -> None:
        if self.wants("summary"):
            self.record_event("iter_summary", iter=int(iter_idx), **payload)

    def record_connection_attempt(self, *, iter_idx: int, **payload: Any) -> None:
        if self.wants("summary"):
            self.record_event("connection_attempt", iter=int(iter_idx), **payload)

    def record_path(self, *, stage: str, iter_idx: int, batch_idx: int, q: Any, **payload: Any) -> None:
        self.record_event(
            "solution_path",
            stage=str(stage),
            iter=int(iter_idx),
            batch_idx=int(batch_idx),
            q=_json_safe(q),
            path_len=len(q) if hasattr(q, "__len__") else None,
            **payload,
        )

    def has_events_for_batch(self, batch_idx: int) -> bool:
        return any(int(event.get("batch_idx", -1)) == int(batch_idx) for event in self.events)

    def choose_batch_idx(self, preferred: Optional[int] = None) -> int:
        if preferred is not None and int(preferred) >= 0 and self.has_events_for_batch(int(preferred)):
            return int(preferred)

        counts: Dict[int, int] = {}
        for event in self.events:
            batch_idx = int(event.get("batch_idx", -1))
            if batch_idx < 0:
                continue
            counts[batch_idx] = counts.get(batch_idx, 0) + 1
        if not counts:
            return 0
        return max(sorted(counts), key=lambda batch_idx: counts[batch_idx])

    def save(self, path: str, *, meta: Dict[str, Any]) -> Path:
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta = dict(meta)
        out_meta["schema_version"] = TRACE_SCHEMA_VERSION
        out_meta["trace_level"] = str(self.level)
        out_meta["planner"] = str(self.planner)
        payload = {"meta": _json_safe(out_meta), "events": self.events}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path
