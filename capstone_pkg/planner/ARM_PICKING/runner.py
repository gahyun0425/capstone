from __future__ import annotations

from typing import Sequence

from capstone_pkg.planner.ARM_PICKING.action_server import main_arm_picking_action_server
from capstone_pkg.planner.arm_rrt_common.single_arm_runner import main_single_arm
from capstone_pkg.utils.config import LONG_SHELF_YAML, SHELF_YAML


_COLLISION_MODELS = {
    "long_shelf": LONG_SHELF_YAML,
    "shelf": SHELF_YAML,
    "shelf_1": SHELF_YAML,
    "shelf_2": LONG_SHELF_YAML,
}


def main_arm_picking(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    if argv_list is not None and "--legacy_cli" in argv_list:
        filtered_argv = [arg for arg in argv_list if arg != "--legacy_cli"]
        return main_single_arm(
            filtered_argv,
            planner_name="ARM_PICKING",
            collision_models=_COLLISION_MODELS,
            default_collision_model="long_shelf",
        )
    return main_arm_picking_action_server(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_picking(argv)


if __name__ == "__main__":
    raise SystemExit(main())
