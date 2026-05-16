from __future__ import annotations

import argparse
import os
import struct
from typing import Dict, Iterable, List, Tuple

import yaml


DEFAULT_SPHERES_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/spheres/ffw_sg2_spheres.yaml"
DEFAULT_URDF = "/home/gaga/capstone_ws/src/capstone_pkg/models/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
DEFAULT_OUT = "/tmp/gripper_tip_spheres.png"


TIP_LINKS = [
    "gripper_l_rh_p12_rn_r2",
    "gripper_l_rh_p12_rn_l2",
    "gripper_r_rh_p12_rn_r2",
    "gripper_r_rh_p12_rn_l2",
]


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_spheres(path: str) -> Dict[str, List[Dict[str, List[float]]]]:
    data = _load_yaml(path) or {}
    spheres = data.get("collision_spheres", {}) or {}
    if not isinstance(spheres, dict):
        raise ValueError("collision_spheres yaml format is invalid")
    return spheres


def _parse_link_meshes_from_urdf(urdf_path: str) -> Dict[str, str]:
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    out: Dict[str, str] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        coll = link.find("collision")
        if coll is None:
            continue
        geom = coll.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename")
        if filename:
            out[name] = filename
    return out


def _stl_bounds_mm(path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    with open(path, "rb") as f:
        _ = f.read(80)
        tri_count = struct.unpack("<I", f.read(4))[0]
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        for _ in range(tri_count):
            rec = f.read(50)
            if len(rec) < 50:
                break
            vals = struct.unpack("<12fH", rec)
            verts = [vals[3:6], vals[6:9], vals[9:12]]
            for vx, vy, vz in verts:
                mins[0] = min(mins[0], vx)
                mins[1] = min(mins[1], vy)
                mins[2] = min(mins[2], vz)
                maxs[0] = max(maxs[0], vx)
                maxs[1] = max(maxs[1], vy)
                maxs[2] = max(maxs[2], vz)
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def _mm_to_m(bounds_mm: Tuple[Tuple[float, float, float], Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mins, maxs = bounds_mm
    return tuple(v * 0.001 for v in mins), tuple(v * 0.001 for v in maxs)


def _iter_tip_links(links: Iterable[str]) -> List[str]:
    return [name for name in links if name in TIP_LINKS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize gripper tip collision spheres in local link frames.")
    ap.add_argument("--spheres_yaml", default=DEFAULT_SPHERES_YAML)
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    spheres = _load_spheres(args.spheres_yaml)
    link_meshes = _parse_link_meshes_from_urdf(args.urdf)
    selected = _iter_tip_links(spheres.keys())
    if not selected:
        raise RuntimeError("No tip links found in sphere yaml")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, link_name in zip(axes, selected):
        mesh_path = link_meshes.get(link_name)
        if not mesh_path or not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh not found for {link_name}: {mesh_path}")

        mins, maxs = _mm_to_m(_stl_bounds_mm(mesh_path))
        min_y, min_z = mins[1], mins[2]
        size_y = maxs[1] - mins[1]
        size_z = maxs[2] - mins[2]

        rect = Rectangle((min_y, min_z), size_y, size_z, fill=False, linewidth=1.5)
        ax.add_patch(rect)

        for idx, sph in enumerate(spheres[link_name]):
            cy = float(sph["center"][1])
            cz = float(sph["center"][2])
            radius = float(sph["radius"])
            color = "tab:red" if idx == len(spheres[link_name]) - 1 and len(spheres[link_name]) > 1 else "tab:blue"
            ax.add_patch(Circle((cy, cz), radius, fill=False, linewidth=2.0, color=color))
            ax.scatter([cy], [cz], s=18, color=color)
            ax.text(cy, cz, f"  s{idx}", fontsize=8)

        ax.set_title(link_name)
        ax.set_xlabel("local y (m)")
        ax.set_ylabel("local z (m)")
        ax.set_aspect("equal", adjustable="box")

        pad = 0.01
        ax.set_xlim(min_y - pad, maxs[1] + pad)
        ax.set_ylim(min_z - pad, maxs[2] + pad)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Gripper tip link spheres (YZ view, local link frame)\nblue=existing, red=new tip sphere")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    print(f"[viz] saved: {args.out}")
    print("[viz] link summary:")
    for link_name in selected:
        print(f"  {link_name}")
        for idx, sph in enumerate(spheres[link_name]):
            center = [float(v) for v in sph["center"]]
            radius = float(sph["radius"])
            tag = "new_tip" if idx == len(spheres[link_name]) - 1 and len(spheres[link_name]) > 1 else "existing"
            print(f"    - {tag}: center={center}, radius={radius}")


if __name__ == "__main__":
    main()
