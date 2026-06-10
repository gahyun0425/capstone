from __future__ import annotations

import colorsys
import math
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from capstone_pkg.utils.config import BASE_FRAME, ROBOT_YAML


def _joint_color(index: int, total: int) -> Tuple[float, float, float]:
    """Generate stable, vivid colors for each joint line."""
    if total <= 0:
        return 1.0, 1.0, 1.0
    hue = float(index % total) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return float(r), float(g), float(b)


def _duration_from_seconds(seconds: float) -> Duration:
    lifetime = Duration(sec=0, nanosec=0)
    if seconds > 0.0:
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        lifetime = Duration(sec=sec, nanosec=nanosec)
    return lifetime


def _compute_fk_points(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    robot_yml: str,
    base_frame: str,
    ee_frame: str,
    cpu: bool,
) -> list[Tuple[float, float, float]]:
    from capstone_pkg.kinematics.curobo_test_fk import compute_relative_link_path_from_cspace

    try:
        return compute_relative_link_path_from_cspace(
            path,
            joint_names,
            robot_yml=robot_yml,
            base_link=base_frame,
            ee_link=ee_frame,
            cpu=cpu,
        )
    except Exception:
        return _compute_fk_points_urdf(
            path,
            joint_names,
            robot_yml=robot_yml,
            base_frame=base_frame,
            ee_frame=ee_frame,
        )


def _resolve_urdf_path_from_robot_yml(robot_yml: str) -> str:
    import yaml

    with open(str(robot_yml), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    kin = ((cfg.get("robot_cfg") or {}).get("kinematics") or {})
    urdf_path = kin.get("urdf_path", "")
    if not isinstance(urdf_path, str) or not urdf_path:
        raise RuntimeError(f"robot_yml has no robot_cfg.kinematics.urdf_path: {robot_yml}")

    candidates = []
    raw = Path(urdf_path).expanduser()
    candidates.append(raw)
    asset_root = kin.get("asset_root_path", "")
    if isinstance(asset_root, str) and asset_root:
        candidates.append(Path(asset_root).expanduser() / urdf_path)
    candidates.append(Path(robot_yml).expanduser().parent / urdf_path)

    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"URDF not found from robot_yml={robot_yml}: {urdf_path}")


def _compute_fk_points_urdf(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    robot_yml: str,
    base_frame: str,
    ee_frame: str,
) -> list[Tuple[float, float, float]]:
    from capstone_pkg.constraint_projection.bimanual_jacobian_compare_urdf import URDFModel

    urdf_path = _resolve_urdf_path_from_robot_yml(robot_yml)
    model = URDFModel(urdf_path)
    out: list[Tuple[float, float, float]] = []

    for waypoint_idx, q in enumerate(path):
        if len(q) != len(joint_names):
            raise ValueError(
                f"path[{waypoint_idx}] length {len(q)} != len(joint_names) {len(joint_names)}"
            )
        q_vec = [float(v) for v in q]
        T_ee, _ = model.fk_and_geometric_jacobian_world(
            str(ee_frame),
            q_vec,
            [str(n) for n in joint_names],
        )
        T_base, _ = model.fk_and_geometric_jacobian_world(
            str(base_frame),
            q_vec,
            [str(n) for n in joint_names],
        )
        rel = T_base[:3, :3].T @ (T_ee[:3, 3] - T_base[:3, 3])
        out.append((float(rel[0]), float(rel[1]), float(rel[2])))
    return out


def _safe_plot_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return safe.strip("_") or "path"


def resolve_plot_output_path(out_png: str | None, *, prefix: str, file_suffix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{_safe_plot_name(prefix)}_{timestamp}.png"
    raw = str(out_png or "").strip()
    if not raw:
        path = Path.cwd() / filename
    else:
        candidate = Path(raw).expanduser()
        if candidate.suffix.lower() == ".png":
            suffix = _safe_plot_name(file_suffix).strip("_")
            path = candidate.with_name(f"{candidate.stem}_{suffix}{candidate.suffix}") if suffix else candidate
        else:
            path = candidate / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_world_model(world_yml: str | None) -> dict[str, Any] | None:
    if world_yml in (None, "", "none", "None"):
        return None
    try:
        import yaml

        with open(str(world_yml), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _quat_wxyz_to_rotmat(q: Sequence[float]) -> list[list[float]]:
    w, x, y, z = [float(v) for v in q[:4]]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-12:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _cuboid_corners(center_xyz: Sequence[float], dims_xyz: Sequence[float], quat_wxyz: Sequence[float]):
    hx, hy, hz = [0.5 * float(v) for v in dims_xyz[:3]]
    local = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    rot = _quat_wxyz_to_rotmat(quat_wxyz)
    cx, cy, cz = [float(v) for v in center_xyz[:3]]
    out = []
    for x, y, z in local:
        out.append(
            (
                rot[0][0] * x + rot[0][1] * y + rot[0][2] * z + cx,
                rot[1][0] * x + rot[1][1] * y + rot[1][2] * z + cy,
                rot[2][0] * x + rot[2][1] * y + rot[2][2] * z + cz,
            )
        )
    return out


def _convex_hull_2d(points):
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _draw_world_cuboids_2d(ax, world_model: dict[str, Any] | None, plane_axes: tuple[int, int]) -> None:
    if not world_model:
        return
    try:
        from matplotlib.patches import Polygon
    except Exception:
        return
    cuboids = world_model.get("cuboid", {}) or {}
    if not isinstance(cuboids, dict):
        return
    i0, i1 = plane_axes
    for item in cuboids.values():
        if not isinstance(item, dict) or "dims" not in item or "pose" not in item:
            continue
        pose = item.get("pose") or []
        dims = item.get("dims") or []
        if len(pose) < 7 or len(dims) < 3:
            continue
        corners = _cuboid_corners(pose[:3], dims[:3], pose[3:7])
        hull = _convex_hull_2d([(p[i0], p[i1]) for p in corners])
        if len(hull) >= 3:
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    alpha=0.12,
                    linewidth=0.7,
                    edgecolor=(0.70, 0.05, 0.02, 0.75),
                    facecolor=(0.95, 0.20, 0.05, 0.18),
                )
            )


def _world_cuboid_corners(world_model: dict[str, Any] | None) -> list[list[tuple[float, float, float]]]:
    if not world_model:
        return []
    cuboids = world_model.get("cuboid", {}) or {}
    if not isinstance(cuboids, dict):
        return []

    out: list[list[tuple[float, float, float]]] = []
    for item in cuboids.values():
        if not isinstance(item, dict) or "dims" not in item or "pose" not in item:
            continue
        pose = item.get("pose") or []
        dims = item.get("dims") or []
        if len(pose) < 7 or len(dims) < 3:
            continue
        out.append(_cuboid_corners(pose[:3], dims[:3], pose[3:7]))
    return out


def _draw_world_cuboids_3d(ax, world_model: dict[str, Any] | None) -> None:
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    for corners in _world_cuboid_corners(world_model):
        poly3d = [[corners[idx] for idx in face] for face in faces]
        pc = Poly3DCollection(
            poly3d,
            alpha=0.18,
            linewidths=0.7,
            edgecolors=(0.70, 0.05, 0.02, 0.75),
            facecolors=(0.95, 0.20, 0.05, 0.18),
        )
        ax.add_collection3d(pc)


def _subsample_path(path: Sequence[Sequence[float]], max_points: int) -> list[list[float]]:
    rows = [[float(v) for v in q] for q in path]
    n = len(rows)
    if max_points <= 0 or n <= max_points:
        return rows
    if max_points == 1:
        return [rows[0]]
    idxs = [
        round(float(i) * float(n - 1) / float(max_points - 1))
        for i in range(max_points)
    ]
    return [rows[int(i)] for i in idxs]


def _set_equal_2d_limits(ax, xs: Sequence[float], ys: Sequence[float]) -> None:
    if not xs or not ys:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    radius = max(0.05, 0.55 * max(xmax - xmin, ymax - ymin))
    ax.set_xlim(xmid - radius, xmid + radius)
    ax.set_ylim(ymid - radius, ymid + radius)


def _set_equal_3d_limits(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    zs: Sequence[float],
) -> None:
    if not xs or not ys or not zs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)
    radius = max(0.05, 0.55 * max(xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_xlim(xmid - radius, xmid + radius)
    ax.set_ylim(ymid - radius, ymid + radius)
    ax.set_zlim(zmid - radius, zmid + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _transform_points_by_base_poses(
    points: Sequence[tuple[float, float, float]],
    base_poses: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float, float]]:
    if base_poses is None or len(base_poses) != len(points):
        return [(float(x), float(y), float(z)) for x, y, z in points]

    out: list[tuple[float, float, float]] = []
    for point, pose in zip(points, base_poses):
        if len(pose) < 3:
            out.append((float(point[0]), float(point[1]), float(point[2])))
            continue
        x, y, z = [float(v) for v in point]
        bx, by, byaw = [float(v) for v in pose[:3]]
        c = math.cos(byaw)
        s = math.sin(byaw)
        out.append((bx + c * x - s * y, by + s * x + c * y, z))
    return out


def save_joint_path_plot_matplotlib(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    out_png: str | None = None,
    prefix: str = "joint_path",
    x_step: float = 0.05,
    y_scale: float = 1.0,
    z_separation: float = 0.25,
    line_width: float = 1.75,
    title: str = "Joint Path Plot",
) -> str:
    out_path = resolve_plot_output_path(out_png, prefix=prefix)
    show_joint_path_plot_matplotlib(
        path,
        joint_names,
        x_step=x_step,
        y_scale=y_scale,
        z_separation=z_separation,
        line_width=line_width,
        title=title,
        block=False,
        out_png=out_path,
        show=False,
    )
    return out_path


def save_ee_path_plot_matplotlib(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    ee_frames: Sequence[tuple[str, str]],
    robot_yml: str = ROBOT_YAML,
    base_frame: str = BASE_FRAME,
    world_yml: str | None = None,
    out_png: str | None = None,
    prefix: str = "path_ee",
    title: str = "End-Effector Path",
    cpu: bool = False,
    max_path_points: int = 2000,
    base_poses: Sequence[Sequence[float]] | None = None,
) -> str:
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if not ee_frames:
        raise ValueError("ee_frames is empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = resolve_plot_output_path(out_png, prefix=prefix)
    plot_path = _subsample_path(path, int(max_path_points))
    plot_base_poses = _subsample_path(base_poses, int(max_path_points)) if base_poses is not None else None
    world_model = _load_world_model(world_yml)
    ee_paths = []
    for label, ee_frame in ee_frames:
        fk_points = _compute_fk_points(
            plot_path,
            joint_names,
            robot_yml=robot_yml,
            base_frame=base_frame,
            ee_frame=ee_frame,
            cpu=cpu,
        )
        fk_points = _transform_points_by_base_poses(fk_points, plot_base_poses)
        ee_paths.append((str(label), str(ee_frame), fk_points))

    plane_specs = [("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))]
    axis_labels = ["x", "y", "z"]
    n_cols = len(ee_paths)
    fig = plt.figure(figsize=(7 * n_cols, 18))
    axes_2d = [[None for _ in range(n_cols)] for _ in range(3)]
    cuboid_corners = _world_cuboid_corners(world_model)

    for col, (label, ee_frame, points) in enumerate(ee_paths):
        ax3d = fig.add_subplot(4, n_cols, col + 1, projection="3d")
        _draw_world_cuboids_3d(ax3d, world_model)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        ax3d.plot(xs, ys, zs, linewidth=2.4, color="#7B2CBF", label="ARM PATH")
        ax3d.scatter([xs[0]], [ys[0]], [zs[0]], s=85, marker="*", color="C2", label="ARM START")
        ax3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=85, marker="X", color="C3", label="ARM GOAL")
        if plot_base_poses is not None:
            bx = [float(p[0]) for p in plot_base_poses if len(p) >= 2]
            by = [float(p[1]) for p in plot_base_poses if len(p) >= 2]
            bz = [0.0 for _ in bx]
            byaw = [float(p[2]) for p in plot_base_poses if len(p) >= 3]
            if bx and by:
                ax3d.plot(bx, by, bz, linewidth=2.2, color="C0", linestyle="--", label="BASE PATH")
                ax3d.scatter([bx[0]], [by[0]], [0.0], s=70, marker="o", color="C0", label="BASE START")
                ax3d.scatter([bx[-1]], [by[-1]], [0.0], s=70, marker="s", color="C0", label="BASE GOAL")
                if byaw:
                    stride = max(1, len(bx) // 12)
                    arrow_len = max(0.03, 0.08 * max(max(bx) - min(bx), max(by) - min(by), 0.5))
                    ax3d.quiver(
                        bx[::stride],
                        by[::stride],
                        [0.0 for _ in bx[::stride]],
                        [math.cos(v) * arrow_len for v in byaw[::stride]],
                        [math.sin(v) * arrow_len for v in byaw[::stride]],
                        [0.0 for _ in byaw[::stride]],
                        color="C0",
                        linewidth=0.8,
                        arrow_length_ratio=0.35,
                    )

        limit_xs = list(xs)
        limit_ys = list(ys)
        limit_zs = list(zs)
        for corners in cuboid_corners:
            limit_xs.extend(p[0] for p in corners)
            limit_ys.extend(p[1] for p in corners)
            limit_zs.extend(p[2] for p in corners)
        if plot_base_poses is not None:
            limit_xs.extend(float(p[0]) for p in plot_base_poses if len(p) >= 2)
            limit_ys.extend(float(p[1]) for p in plot_base_poses if len(p) >= 2)
            limit_zs.extend(0.0 for p in plot_base_poses if len(p) >= 2)
        _set_equal_3d_limits(ax3d, limit_xs, limit_ys, limit_zs)
        ax3d.set_title(f"{label}: {ee_frame} (3D)")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.view_init(elev=24, azim=-58)
        ax3d.legend(loc="upper right")

    for row in range(3):
        for col in range(n_cols):
            axes_2d[row][col] = fig.add_subplot(4, n_cols, (row + 1) * n_cols + col + 1)

    for col, (label, ee_frame, points) in enumerate(ee_paths):
        xs_all = [p[0] for p in points]
        ys_all = [p[1] for p in points]
        zs_all = [p[2] for p in points]
        coords = [xs_all, ys_all, zs_all]
        for row, (plane_name, (i0, i1)) in enumerate(plane_specs):
            ax = axes_2d[row][col]
            _draw_world_cuboids_2d(ax, world_model, (i0, i1))
            ax.plot(coords[i0], coords[i1], linewidth=2.2, color="#7B2CBF", label="PATH")
            ax.scatter([coords[i0][0]], [coords[i1][0]], s=85, marker="*", color="C2", label="START")
            ax.scatter([coords[i0][-1]], [coords[i1][-1]], s=85, marker="X", color="C3", label="GOAL")
            if plot_base_poses is not None and plane_name == "XY":
                bx = [float(p[0]) for p in plot_base_poses if len(p) >= 2]
                by = [float(p[1]) for p in plot_base_poses if len(p) >= 2]
                byaw = [float(p[2]) for p in plot_base_poses if len(p) >= 3]
                if bx and by:
                    ax.plot(bx, by, linewidth=2.0, color="C0", linestyle="--", label="BASE")
                    ax.scatter([bx[0]], [by[0]], s=70, marker="o", color="C0", label="BASE START")
                    ax.scatter([bx[-1]], [by[-1]], s=70, marker="s", color="C0", label="BASE GOAL")
                    if byaw:
                        stride = max(1, len(bx) // 12)
                        arrow_len = max(0.03, 0.08 * max(max(bx) - min(bx), max(by) - min(by), 0.5))
                        ax.quiver(
                            bx[::stride],
                            by[::stride],
                            [math.cos(v) * arrow_len for v in byaw[::stride]],
                            [math.sin(v) * arrow_len for v in byaw[::stride]],
                            angles="xy",
                            scale_units="xy",
                            scale=1.0,
                            color="C0",
                            width=0.004,
                        )
            ax.set_title(f"{label}: {ee_frame} ({plane_name})")
            ax.set_xlabel(axis_labels[i0])
            ax.set_ylabel(axis_labels[i1])
            ax.grid(True, alpha=0.3)
            limit_xs = list(coords[i0])
            limit_ys = list(coords[i1])
            for corners in cuboid_corners:
                limit_xs.extend(p[i0] for p in corners)
                limit_ys.extend(p[i1] for p in corners)
            if plot_base_poses is not None and plane_name == "XY":
                limit_xs.extend([float(p[0]) for p in plot_base_poses if len(p) >= 2])
                limit_ys.extend([float(p[1]) for p in plot_base_poses if len(p) >= 2])
            _set_equal_2d_limits(ax, limit_xs, limit_ys)
            if row == 0 and col == 0:
                ax.legend(loc="best")

    fig.suptitle(str(title), y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_ee_path_plot_3d_matplotlib(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    ee_frames: Sequence[tuple[str, str]],
    robot_yml: str = ROBOT_YAML,
    base_frame: str = BASE_FRAME,
    world_yml: str | None = None,
    out_png: str | None = None,
    prefix: str = "path_ee_3d",
    title: str = "End-Effector Path 3D",
    cpu: bool = False,
    max_path_points: int = 2000,
    base_poses: Sequence[Sequence[float]] | None = None,
) -> str:
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if not ee_frames:
        raise ValueError("ee_frames is empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = resolve_plot_output_path(out_png, prefix=prefix, file_suffix="3d")
    plot_path = _subsample_path(path, int(max_path_points))
    plot_base_poses = _subsample_path(base_poses, int(max_path_points)) if base_poses is not None else None
    world_model = _load_world_model(world_yml)
    cuboid_corners = _world_cuboid_corners(world_model)
    ee_paths = []
    for label, ee_frame in ee_frames:
        fk_points = _compute_fk_points(
            plot_path,
            joint_names,
            robot_yml=robot_yml,
            base_frame=base_frame,
            ee_frame=ee_frame,
            cpu=cpu,
        )
        fk_points = _transform_points_by_base_poses(fk_points, plot_base_poses)
        ee_paths.append((str(label), str(ee_frame), fk_points))

    view_specs = [
        ("ISO", 24, -58),
        ("OPPOSITE", 24, 122),
        ("SIDE", 18, -8),
        ("TOP", 76, -90),
    ]
    n_cols = len(ee_paths)
    grid_cols = max(1, 2 * n_cols)
    fig = plt.figure(figsize=(8.5 * grid_cols, 15))

    for col, (label, ee_frame, points) in enumerate(ee_paths):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        limit_xs = list(xs)
        limit_ys = list(ys)
        limit_zs = list(zs)
        for corners in cuboid_corners:
            limit_xs.extend(p[0] for p in corners)
            limit_ys.extend(p[1] for p in corners)
            limit_zs.extend(p[2] for p in corners)
        if plot_base_poses is not None:
            limit_xs.extend(float(p[0]) for p in plot_base_poses if len(p) >= 2)
            limit_ys.extend(float(p[1]) for p in plot_base_poses if len(p) >= 2)
            limit_zs.extend(0.0 for p in plot_base_poses if len(p) >= 2)

        for view_idx, (view_name, elev, azim) in enumerate(view_specs):
            view_row = view_idx // 2
            view_col = view_idx % 2
            subplot_idx = view_row * grid_cols + col * 2 + view_col + 1
            ax = fig.add_subplot(2, grid_cols, subplot_idx, projection="3d")
            _draw_world_cuboids_3d(ax, world_model)
            ax.plot(xs, ys, zs, linewidth=3.2, color="#7B2CBF", label="ARM PATH")
            ax.scatter([xs[0]], [ys[0]], [zs[0]], s=140, marker="*", color="C2", label="ARM START")
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=140, marker="X", color="C3", label="ARM GOAL")

            if plot_base_poses is not None:
                bx = [float(p[0]) for p in plot_base_poses if len(p) >= 2]
                by = [float(p[1]) for p in plot_base_poses if len(p) >= 2]
                byaw = [float(p[2]) for p in plot_base_poses if len(p) >= 3]
                if bx and by:
                    ax.plot(bx, by, [0.0 for _ in bx], linewidth=2.8, color="C0", linestyle="--", label="BASE PATH")
                    ax.scatter([bx[0]], [by[0]], [0.0], s=110, marker="o", color="C0", label="BASE START")
                    ax.scatter([bx[-1]], [by[-1]], [0.0], s=110, marker="s", color="C0", label="BASE GOAL")
                    if byaw:
                        stride = max(1, len(bx) // 16)
                        arrow_len = max(0.04, 0.08 * max(max(bx) - min(bx), max(by) - min(by), 0.5))
                        ax.quiver(
                            bx[::stride],
                            by[::stride],
                            [0.0 for _ in bx[::stride]],
                            [math.cos(v) * arrow_len for v in byaw[::stride]],
                            [math.sin(v) * arrow_len for v in byaw[::stride]],
                            [0.0 for _ in byaw[::stride]],
                            color="C0",
                            linewidth=1.1,
                            arrow_length_ratio=0.35,
                        )

            _set_equal_3d_limits(ax, limit_xs, limit_ys, limit_zs)
            ax.set_title(f"{label}: {ee_frame} ({view_name})", pad=14)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            if view_idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle(str(title), y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _append_fk_plot_markers(
    markers: MarkerArray,
    fk_points: Sequence[Tuple[float, float, float]],
    *,
    frame_id: str,
    ee_frame_name: str,
    marker_lifetime_s: float,
    line_width: float,
) -> None:
    if not fk_points:
        return

    lifetime = _duration_from_seconds(marker_lifetime_s)
    marker_base_id = 100000

    fk_line = Marker()
    fk_line.header.frame_id = frame_id
    fk_line.ns = "arm_rrt_fk_path"
    fk_line.id = marker_base_id
    fk_line.type = Marker.LINE_STRIP
    fk_line.action = Marker.ADD
    fk_line.pose.orientation.w = 1.0
    fk_line.frame_locked = True
    fk_line.scale.x = float(line_width)
    fk_line.color.r = 1.0
    fk_line.color.g = 0.95
    fk_line.color.b = 0.05
    fk_line.color.a = 1.0
    fk_line.lifetime = lifetime
    for x, y, z in fk_points:
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        fk_line.points.append(p)
    markers.markers.append(fk_line)

    start = Marker()
    start.header.frame_id = frame_id
    start.ns = "arm_rrt_fk_path_endpoint"
    start.id = marker_base_id + 1
    start.type = Marker.SPHERE
    start.action = Marker.ADD
    start.pose.orientation.w = 1.0
    start.frame_locked = True
    start.scale.x = float(line_width * 3.0)
    start.scale.y = float(line_width * 3.0)
    start.scale.z = float(line_width * 3.0)
    start.color.r = 0.1
    start.color.g = 1.0
    start.color.b = 0.1
    start.color.a = 1.0
    start.lifetime = lifetime
    start.pose.position.x = float(fk_points[0][0])
    start.pose.position.y = float(fk_points[0][1])
    start.pose.position.z = float(fk_points[0][2])
    markers.markers.append(start)

    goal = Marker()
    goal.header.frame_id = frame_id
    goal.ns = "arm_rrt_fk_path_endpoint"
    goal.id = marker_base_id + 2
    goal.type = Marker.SPHERE
    goal.action = Marker.ADD
    goal.pose.orientation.w = 1.0
    goal.frame_locked = True
    goal.scale.x = float(line_width * 3.0)
    goal.scale.y = float(line_width * 3.0)
    goal.scale.z = float(line_width * 3.0)
    goal.color.r = 1.0
    goal.color.g = 0.2
    goal.color.b = 0.2
    goal.color.a = 1.0
    goal.lifetime = lifetime
    goal.pose.position.x = float(fk_points[-1][0])
    goal.pose.position.y = float(fk_points[-1][1])
    goal.pose.position.z = float(fk_points[-1][2])
    markers.markers.append(goal)

    label = Marker()
    label.header.frame_id = frame_id
    label.ns = "arm_rrt_fk_path_label"
    label.id = marker_base_id + 3
    label.type = Marker.TEXT_VIEW_FACING
    label.action = Marker.ADD
    label.pose.orientation.w = 1.0
    label.frame_locked = True
    label.scale.z = float(max(0.05, line_width * 8.0))
    label.color.r = 1.0
    label.color.g = 0.95
    label.color.b = 0.05
    label.color.a = 1.0
    label.text = f"FK EE path: {ee_frame_name}"
    label.lifetime = lifetime
    label.pose.position.x = float(fk_points[-1][0])
    label.pose.position.y = float(fk_points[-1][1])
    label.pose.position.z = float(fk_points[-1][2] + max(0.03, line_width * 2.0))
    markers.markers.append(label)


def _build_joint_plot_markers(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    frame_id: str,
    x_step: float,
    y_scale: float,
    z_separation: float,
    line_width: float,
    text_height: float,
    marker_lifetime_s: float,
) -> MarkerArray:
    markers = MarkerArray()
    lifetime = _duration_from_seconds(marker_lifetime_s)

    # Clear previous marker IDs on this topic so repeated runs stay clean.
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    total = len(joint_names)
    for joint_idx, joint_name in enumerate(joint_names):
        r, g, b = _joint_color(joint_idx, total)
        z = float(joint_idx) * float(z_separation)

        line = Marker()
        line.header.frame_id = frame_id
        line.ns = "arm_rrt_joint_path"
        line.id = int(joint_idx)
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.frame_locked = True
        line.scale.x = float(line_width)
        line.color.r = r
        line.color.g = g
        line.color.b = b
        line.color.a = 1.0
        line.lifetime = lifetime

        for waypoint_idx, q in enumerate(path):
            p = Point()
            p.x = float(waypoint_idx) * float(x_step)
            p.y = float(q[joint_idx]) * float(y_scale)
            p.z = z
            line.points.append(p)

        markers.markers.append(line)

        label = Marker()
        label.header.frame_id = frame_id
        label.ns = "arm_rrt_joint_path_label"
        label.id = int(total + joint_idx)
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.orientation.w = 1.0
        label.frame_locked = True
        label.scale.z = float(text_height)
        label.color.r = r
        label.color.g = g
        label.color.b = b
        label.color.a = 1.0
        label.text = joint_name
        label.lifetime = lifetime

        last_idx = max(0, len(path) - 1)
        label.pose.position.x = float(last_idx) * float(x_step) + 0.08
        label.pose.position.y = float(path[-1][joint_idx]) * float(y_scale)
        label.pose.position.z = z

        markers.markers.append(label)

    return markers


def publish_joint_path_plot(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    topic: str = "/arm_rrt/joint_path_plot",
    frame_id: str = "map",
    x_step: float = 0.05,
    y_scale: float = 1.0,
    z_separation: float = 0.25,
    line_width: float = 0.02,
    text_height: float = 0.12,
    marker_lifetime_s: float = 0.0,
    wait_subscriber_s: float = 1.0,
    publish_repeat: int = 1,
    publish_period_s: float = 0.1,
    keep_alive_s: float = 5.0,
    fk_path: Sequence[Sequence[float]] | None = None,
    fk_joint_names: Sequence[str] | None = None,
    fk_robot_yml: str = ROBOT_YAML,
    fk_base_frame: str = BASE_FRAME,
    fk_ee_frame: str | None = None,
    fk_line_width: float = 0.025,
    fk_cpu: bool = False,
    fk_fail_silently: bool = True,
) -> None:
    """Publish joint-space waypoints as an RViz2 MarkerArray plot."""
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if x_step <= 0.0:
        raise ValueError("x_step must be > 0")
    if line_width <= 0.0:
        raise ValueError("line_width must be > 0")
    if text_height <= 0.0:
        raise ValueError("text_height must be > 0")
    if fk_line_width <= 0.0:
        raise ValueError("fk_line_width must be > 0")

    n = len(joint_names)
    for i, q in enumerate(path):
        if len(q) != n:
            raise ValueError(f"path[{i}] length {len(q)} != len(joint_names) {n}")

    wants_fk = (fk_path is not None) or (fk_joint_names is not None) or (fk_ee_frame is not None)
    if wants_fk:
        if not fk_path:
            raise ValueError("fk_path is empty or not provided")
        if not fk_joint_names:
            raise ValueError("fk_joint_names is empty or not provided")
        if not fk_ee_frame:
            raise ValueError("fk_ee_frame is empty or not provided")

    owns_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        owns_rclpy = True

    qos = QoSProfile(depth=1)
    qos.reliability = QoSReliabilityPolicy.RELIABLE
    qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

    node = Node("arm_rrt_plot_publisher")
    try:
        pub = node.create_publisher(MarkerArray, topic, qos)

        t_end = time.monotonic() + max(0.0, float(wait_subscriber_s))
        while rclpy.ok() and time.monotonic() < t_end and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)

        markers = _build_joint_plot_markers(
            path,
            joint_names,
            frame_id=frame_id,
            x_step=x_step,
            y_scale=y_scale,
            z_separation=z_separation,
            line_width=line_width,
            text_height=text_height,
            marker_lifetime_s=marker_lifetime_s,
        )

        if wants_fk and fk_path is not None and fk_joint_names is not None and fk_ee_frame is not None:
            try:
                fk_points = _compute_fk_points(
                    fk_path,
                    fk_joint_names,
                    robot_yml=fk_robot_yml,
                    base_frame=fk_base_frame,
                    ee_frame=fk_ee_frame,
                    cpu=fk_cpu,
                )
                _append_fk_plot_markers(
                    markers,
                    fk_points,
                    frame_id=fk_base_frame,
                    ee_frame_name=fk_ee_frame,
                    marker_lifetime_s=marker_lifetime_s,
                    line_width=fk_line_width,
                )
            except Exception as exc:
                if fk_fail_silently:
                    node.get_logger().warning(f"FK plot disabled: {exc}")
                else:
                    raise

        repeat = max(1, int(publish_repeat))
        for idx in range(repeat):
            now = node.get_clock().now().to_msg()
            for marker in markers.markers:
                marker.header.stamp = now
            pub.publish(markers)
            rclpy.spin_once(node, timeout_sec=0.0)
            if idx + 1 < repeat:
                time.sleep(max(0.0, float(publish_period_s)))

        # Keep publisher alive for discovery / late RViz subscription.
        if keep_alive_s < 0.0:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            t_keep_end = time.monotonic() + max(0.0, float(keep_alive_s))
            while rclpy.ok() and time.monotonic() < t_keep_end:
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()


def show_joint_path_plot_matplotlib(
    path: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    *,
    x_step: float = 0.05,
    y_scale: float = 1.0,
    z_separation: float = 0.25,
    line_width: float = 1.75,
    title: str = "Joint Path Plot",
    block: bool = True,
    out_png: str | None = None,
    show: bool = True,
) -> None:
    """Show the joint-space path in a matplotlib 3D window."""
    if not path:
        raise ValueError("path is empty")
    if not joint_names:
        raise ValueError("joint_names is empty")
    if x_step <= 0.0:
        raise ValueError("x_step must be > 0")
    if line_width <= 0.0:
        raise ValueError("line_width must be > 0")

    n = len(joint_names)
    for i, q in enumerate(path):
        if len(q) != n:
            raise ValueError(f"path[{i}] length {len(q)} != len(joint_names) {n}")

    if out_png is not None and not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [float(idx) * float(x_step) for idx in range(len(path))]
    total = len(joint_names)
    for joint_idx, joint_name in enumerate(joint_names):
        r, g, b = _joint_color(joint_idx, total)
        ys = [float(q[joint_idx]) * float(y_scale) for q in path]
        zs = [float(joint_idx) * float(z_separation) for _ in path]
        ax.plot(xs, ys, zs, color=(r, g, b), linewidth=float(line_width))
        ax.text(
            float(xs[-1]) + 0.08,
            float(ys[-1]),
            float(zs[-1]),
            joint_name,
            color=(r, g, b),
            fontsize=9,
        )

    ax.set_title(str(title))
    ax.set_xlabel("Waypoint * x_step")
    ax.set_ylabel("Joint value * y_scale")
    ax.set_zlabel("Joint index * z_sep")
    ax.view_init(elev=24.0, azim=-58.0)
    fig.tight_layout()
    if out_png is not None:
        out_path = Path(out_png).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)
        print(f"[PLOT] saved joint path png: {out_path}")
    if show:
        plt.show(block=bool(block))
    else:
        plt.close(fig)
