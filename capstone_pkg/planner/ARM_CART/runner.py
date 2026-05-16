from __future__ import annotations

from typing import Sequence

from capstone_pkg.planner.arm_rrt_common.dual_arm_runner import main_dual_arm

_DEFAULT_STORED_TRAJECTORY_JSON = (
    "/home/gaga/capstone_ws/src/capstone_pkg/data/arm_cart_picking_trajectory.json"
)


def _strip_option_with_nargs(
    src: Sequence[str],
    option: str,
    nargs: int,
) -> list[str]:
    out: list[str] = []
    skip = 0

    for token in src:
        if skip > 0:
            skip -= 1
            continue
        if token == option:
            skip = int(nargs)
            continue
        if token.startswith(f"{option}="):
            continue
        out.append(token)

    return out


def _build_fixed_arm_cart_picking_args(argv: Sequence[str] | None) -> list[str]:
    src = list(argv) if argv is not None else []
    args = _strip_option_with_nargs(src, "--world_yml", 1)
    args = _strip_option_with_nargs(args, "--planner_mode", 1)
    args = _strip_option_with_nargs(args, "--plot_path", 0)
    args = _strip_option_with_nargs(args, "--no-plot_path", 0)
    args = _strip_option_with_nargs(args, "--left_xyz", 3)
    args = _strip_option_with_nargs(args, "--left_quat_xyzw", 4)
    args = _strip_option_with_nargs(args, "--right_xyz", 3)
    args = _strip_option_with_nargs(args, "--right_quat_xyzw", 4)
    has_stored_json = any(
        token == "--stored_trajectory_json" or token.startswith("--stored_trajectory_json=")
        for token in args
    )

    stored_json_args = []
    if not has_stored_json:
        stored_json_args = ["--stored_trajectory_json", _DEFAULT_STORED_TRAJECTORY_JSON]

    return [
        "--world_yml",
        "none",
        "--planner_mode",
        "spline_only",
        "--no-plot_path",
        "--left_xyz",
        "0.4",
        "0.2",
        "1.0",
        "--left_quat_xyzw",
        "0.5",
        "0.5",
        "0.5",
        "0.5",
        "--right_xyz",
        "0.4",
        "-0.2",
        "1.0",
        "--right_quat_xyzw",
        "0.5",
        "-0.5",
        "0.5",
        "-0.5",
        *stored_json_args,
        *args,
    ]


def main_arm_cart_picking(argv: Sequence[str] | None = None) -> int:
    return main_dual_arm(_build_fixed_arm_cart_picking_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    return main_arm_cart_picking(argv)


if __name__ == "__main__":
    raise SystemExit(main())
