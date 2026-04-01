from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, DefaultDict
from collections import defaultdict


@dataclass
class HeuristicStats:
    """Lightweight counters/histograms for heuristic debugging."""

    counters: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    hists: Dict[str, DefaultDict[Any, int]] = field(default_factory=dict)

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] += int(n)

    def add_hist(self, name: str, item: Any, n: int = 1) -> None:
        if name not in self.hists:
            self.hists[name] = defaultdict(int)
        self.hists[name][item] += int(n)

    def summary(self) -> Dict[str, Any]:
        # Convert defaultdicts into normal dicts for pretty printing / serialization.
        out: Dict[str, Any] = {
            "counters": dict(self.counters),
            "hists": {k: dict(v) for k, v in self.hists.items()},
        }
        return out


_GLOBAL: HeuristicStats | None = None


def get_stats(*, reset: bool = False) -> HeuristicStats:
    global _GLOBAL
    if _GLOBAL is None or reset:
        _GLOBAL = HeuristicStats()
    return _GLOBAL
