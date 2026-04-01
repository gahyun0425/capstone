from __future__ import annotations

from typing import Sequence

from capstone_pkg.main import main_birrt


def main(argv: Sequence[str] | None = None) -> int:
    return main_birrt(argv)


if __name__ == "__main__":
    raise SystemExit(main())
