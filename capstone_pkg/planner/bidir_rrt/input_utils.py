from __future__ import annotations

import json
from typing import List

def read_vec(name: str, n: int, example: str) -> List[float]:
    """Read a vector of length n from terminal input."""
    while True:
        s = input(f"{name} ({n} floats), ex) {example} > ").strip()
        try:
            if s.startswith("["):
                arr = json.loads(s)
                if not isinstance(arr, list):
                    raise ValueError("not a list")
                out = [float(x) for x in arr]
            else:
                parts = [p.strip() for p in (s.split(",") if "," in s else s.split()) if p.strip()]
                out = [float(x) for x in parts]
            if len(out) != n:
                raise ValueError(f"length must be {n}, got {len(out)}")
            return out
        except Exception as e:
            print(f"[INPUT ERROR] {e}. Try again.")

def xyzw_to_wxyz(q_xyzw: List[float]) -> List[float]:
    if len(q_xyzw) != 4:
        raise ValueError("quat must have 4 elements [x,y,z,w]")
    x,y,z,w = [float(v) for v in q_xyzw]
    return [w,x,y,z]
