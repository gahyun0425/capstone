from __future__ import annotations

from typing import Sequence

from capstone_pkg.planner.ARM_PICKING.runner import main_arm_picking


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_picking(argv)


if __name__ == "__main__":
    raise SystemExit(main())
