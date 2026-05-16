from __future__ import annotations

from typing import Sequence

from capstone_pkg.planner.arm_rrt_common.single_arm_runner import main_single_arm
from capstone_pkg.utils.config import CART_YAML


_COLLISION_MODELS = {
    "cart": CART_YAML,
}


def main_arm_placing(argv: Sequence[str] | None = None) -> int:
    return main_single_arm(
        argv,
        planner_name="ARM_PLACING",
        collision_models=_COLLISION_MODELS,
        default_collision_model="cart",
    )


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_placing(argv)


if __name__ == "__main__":
    raise SystemExit(main())
