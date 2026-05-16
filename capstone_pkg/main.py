from __future__ import annotations

import sys
from typing import List, Sequence


_PLANNER_ALIASES = {
    "arm_picking": "arm_picking",
    "arm-picking": "arm_picking",
    "arm_placing": "arm_placing",
    "arm-placing": "arm_placing",
    "arm_cart_picking": "arm_cart_picking",
    "arm-cart-picking": "arm_cart_picking",
    "tbrrt": "tbrrt",
    "arm_cart": "arm_cart",
    "arm-cart": "arm_cart",
    "cart": "arm_cart",
    "cart_tbrrt": "arm_cart",
    "cart-tbrrt": "arm_cart",
    "arm_cart_profile": "arm_cart_profile",
    "arm-cart-profile": "arm_cart_profile",
    "cart_profile": "arm_cart_profile",
    "cart-profile": "arm_cart_profile",
}


def _planner_usage() -> str:
    return (
        "Usage:\n"
        "  ros2 run capstone_pkg main -- planner arm_picking [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner arm_placing [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner arm_cart_picking [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner tbrrt [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner arm_cart [planner args...]\n"
        "  ros2 run capstone_pkg main -- planner arm_cart_profile [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner arm_picking [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner arm_placing [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner arm_cart_picking [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner arm_cart [planner args...]\n"
        "  ros2 run capstone_pkg main -- --planner arm_cart_profile [planner args...]\n"
        "\n"
        "Available planners:\n"
        "  arm_picking\n"
        "  arm_placing\n"
        "  arm_cart_picking\n"
        "  tbrrt\n"
        "  arm_cart\n"
        "  arm_cart_profile\n"
        "\n"
        "Aliases:\n"
        "  cart -> arm_cart\n"
        "  cart_profile -> arm_cart_profile\n"
    )


def _normalize_argv(argv: Sequence[str] | None) -> List[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _resolve_planner(argv: Sequence[str] | None) -> tuple[str | None, List[str]]:
    args = _normalize_argv(argv)
    if not args:
        return None, []

    if args[0] in ("-h", "--help", "help"):
        return "help", []

    if args[0] == "planner":
        if len(args) < 2:
            raise ValueError("planner name is required after 'planner'")
        if args[1] in ("-h", "--help", "help"):
            return "help", []
        planner = _PLANNER_ALIASES.get(args[1].lower())
        return planner, args[2:]

    if args[0] in ("-p", "--planner"):
        if len(args) < 2:
            raise ValueError("planner name is required after '--planner'")
        if args[1] in ("-h", "--help", "help"):
            return "help", []
        planner = _PLANNER_ALIASES.get(args[1].lower())
        return planner, args[2:]

    planner = _PLANNER_ALIASES.get(args[0].lower())
    return planner, args[1:]


def main_arm_picking(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.ARM_PICKING.runner import main_arm_picking as _main_arm_picking

    return _main_arm_picking(_normalize_argv(argv))


def main_arm_placing(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.ARM_PLACING.runner import main_arm_placing as _main_arm_placing

    return _main_arm_placing(_normalize_argv(argv))


def main_arm_cart_picking(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.ARM_CART.runner import (
        main_arm_cart_picking as _main_arm_cart_picking,
    )

    return _main_arm_cart_picking(_normalize_argv(argv))


def main_tbrrt(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.tbrrt_runner import main_tbrrt as _main_tbrrt

    return _main_tbrrt(_normalize_argv(argv))


def main_arm_cart(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.ARM_CART.runner import main_arm_cart as _main_arm_cart

    return _main_arm_cart(_normalize_argv(argv))


def main_arm_cart_profile(argv: Sequence[str] | None = None) -> int:
    from capstone_pkg.planner.ARM_CART.precompute import (
        main_arm_cart_profile as _main_arm_cart_profile,
    )

    return _main_arm_cart_profile(_normalize_argv(argv))


def main_cart(argv: Sequence[str] | None = None) -> int:
    return main_arm_cart(argv)


def main_cart_profile(argv: Sequence[str] | None = None) -> int:
    return main_arm_cart_profile(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        planner, planner_args = _resolve_planner(argv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(_planner_usage(), file=sys.stderr)
        return 2

    if planner == "help":
        print(_planner_usage())
        return 0

    if planner == "arm_picking":
        return main_arm_picking(planner_args)
    if planner == "arm_placing":
        return main_arm_placing(planner_args)
    if planner == "arm_cart_picking":
        return main_arm_cart_picking(planner_args)
    if planner == "tbrrt":
        return main_tbrrt(planner_args)
    if planner == "arm_cart":
        return main_arm_cart(planner_args)
    if planner == "arm_cart_profile":
        return main_arm_cart_profile(planner_args)

    print(
        "[ERROR] planner must be one of: arm_picking, arm_placing, arm_cart_picking, tbrrt, arm_cart, arm_cart_profile",
        file=sys.stderr,
    )
    print(_planner_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
