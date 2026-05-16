# bimanual_fk_test.py
from __future__ import annotations
from typing import List, Sequence, Tuple

import argparse
import torch

from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig


def _parse_floats(values: List[str]) -> List[float]:
    # "1 2 3" 또는 "1,2,3" 혼용 입력 지원
    out: List[float] = []
    for v in values:
        for s in v.replace(",", " ").split():
            s = s.strip()
            if s:
                out.append(float(s))
    return out


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _quat_wxyz_to_rotmat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    q = quat_wxyz.to(dtype=torch.float64)
    q = q / torch.linalg.norm(q).clamp(min=1e-12)
    w, x, y, z = q.unbind()
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return torch.stack(
        [
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ]
    )


def _extract_link_pose_from_state(state, robot_model: CudaRobotModel, q_full: torch.Tensor, link_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    link_poses_dict = state.link_poses or {}
    if link_name in link_poses_dict:
        pose = link_poses_dict[link_name]
        pos = pose.position[0]
        quat = pose.quaternion[0]
        return pos.detach(), quat.detach()

    with torch.no_grad():
        one = robot_model.get_state(q_full, link_name=link_name)
    pos = one.ee_position[0]
    quat = one.ee_quaternion[0]
    return pos.detach(), quat.detach()


def compute_relative_link_path_from_cspace(
    path: Sequence[Sequence[float]],
    cspace_joint_names: Sequence[str],
    *,
    robot_yml: str,
    base_link: str,
    ee_link: str,
    cpu: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Compute Cartesian path of ee_link represented in base_link coordinates using cuRobo FK.
    """
    if not path:
        raise ValueError("path is empty")
    if not cspace_joint_names:
        raise ValueError("cspace_joint_names is empty")

    tensor_args = TensorDeviceType()
    if cpu:
        tensor_args = TensorDeviceType(device=torch.device("cpu"))

    cfg = load_yaml(robot_yml)
    kin = cfg["robot_cfg"]["kinematics"]
    # FK 검증에서 특정 링크를 보고 싶을 때는 YAML 기본 ee_link보다
    # 호출자가 요청한 ee_link를 우선 사용해야 한다.
    primary_ee = str(ee_link).strip() or (kin.get("ee_link", "") or "")
    if not primary_ee:
        raise ValueError("ee_link is empty")

    try:
        model_cfg = CudaRobotModelConfig.from_robot_yaml_file(
            robot_yml,
            ee_link=primary_ee,
            tensor_args=tensor_args,
        )
        robot_model = CudaRobotModel(model_cfg)
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "joint_vec.is_cuda()" in msg
            or "cudaGetDeviceCount" in msg
            or "torch._C._cuda_init" in msg
        ):
            raise RuntimeError(
                "cuRobo FK failed to initialize CUDA backend. "
                "Current CudaRobotModel build requires a working CUDA runtime/device."
            ) from exc
        raise

    model_names = list(robot_model.joint_names)
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    missing_names = [n for n in cspace_joint_names if n not in name_to_idx]
    if missing_names:
        raise ValueError(
            f"cspace joints not found in cuRobo robot_model.joint_names: {missing_names}"
        )

    q_full = torch.zeros((1, robot_model.dof), dtype=torch.float32, device=tensor_args.device)
    out: List[Tuple[float, float, float]] = []

    with torch.no_grad():
        for waypoint_idx, q in enumerate(path):
            if len(q) != len(cspace_joint_names):
                raise ValueError(
                    f"path[{waypoint_idx}] length {len(q)} != len(cspace_joint_names) {len(cspace_joint_names)}"
                )

            q_full.zero_()
            for jname, jval in zip(cspace_joint_names, q):
                q_full[0, name_to_idx[jname]] = float(jval)

            state = robot_model.get_state(q_full)
            ee_pos_w, _ = _extract_link_pose_from_state(state, robot_model, q_full, ee_link)
            base_pos_w, base_quat_wxyz = _extract_link_pose_from_state(state, robot_model, q_full, base_link)

            R_wb = _quat_wxyz_to_rotmat(base_quat_wxyz)
            rel = R_wb.transpose(0, 1) @ (ee_pos_w.to(dtype=torch.float64) - base_pos_w.to(dtype=torch.float64))
            out.append((float(rel[0].item()), float(rel[1].item()), float(rel[2].item())))

    return out


def main():
    ap = argparse.ArgumentParser(description="cuRobo FK CLI (bimanual): joint angles -> link poses")
    ap.add_argument(
        "--robot_yml",
        default="/home/gaga/capstone_ws/src/capstone_pkg/models/test_curobo.yaml",
        help="robot config yaml path",
    )
    ap.add_argument(
        "--q",
        nargs="+",
        required=True,
        help="joint angles (radians). Example: --q 0 0.1 -0.2 ...  or --q 0,0.1,-0.2,...",
    )
    ap.add_argument(
        "--links",
        nargs="*",
        default=None,
        help=(
            "FK를 뽑을 링크 이름들. 미지정이면 YAML의 kinematics/ee_link + kinematics/link_poses의 키들을 사용."
        ),
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument(
        "--quat_xyzw",
        action="store_true",
        help="출력 쿼터니언 순서를 (x,y,z,w)로 변환해서 출력 (기본은 wxyz)",
    )
    args = ap.parse_args()

    tensor_args = TensorDeviceType()
    if args.cpu:
        tensor_args = TensorDeviceType(device=torch.device("cpu"))

    # ---------------------------
    # 1) YAML에서 기본 링크들 추출
    # ---------------------------
    cfg = load_yaml(args.robot_yml)
    kin = cfg["robot_cfg"]["kinematics"]
    primary_ee: str = kin["ee_link"]

    # 1) link_names가 있으면 그걸 사용 (양팔 EE가 여기에 있음)
    yaml_link_names = kin.get("link_names", []) or []

    # 2) 그래도 없으면 link_poses key 사용
    link_poses = kin.get("link_poses", {}) or {}
    secondary_links = list(link_poses.keys())

    if args.links is None:
        # primary EE + (link_names + secondary links)에서 중복 제거
        links_to_report = _unique_keep_order([primary_ee] + list(yaml_link_names) + secondary_links)
    else:
        links_to_report = _unique_keep_order(list(args.links))

    # ---------------------------
    # 2) CudaRobotModel 로드 (FK 전용)
    # ---------------------------
    # from_robot_yaml_file: cuRobo 포맷 YAML에서 로드 :contentReference[oaicite:2]{index=2}
    model_cfg = CudaRobotModelConfig.from_robot_yaml_file(
        args.robot_yml,
        ee_link=primary_ee,  # "기본 EE" 정의용 (FK 자체는 링크별로 뽑을 수 있음)
        tensor_args=tensor_args,
    )
    robot_model = CudaRobotModel(model_cfg)

    # ---------------------------
    # 3) q 입력 체크
    # ---------------------------
    q_list = _parse_floats(args.q)
    dof = robot_model.dof

    # YAML에서 팔 14개 관절 이름(입력 순서)
    cspace_names = cfg["robot_cfg"]["cspace"]["joint_names"]

    # 로봇 모델이 실제로 갖고 있는 active joint 이름(16개일 수 있음)
    model_names = list(robot_model.joint_names)

    # q는 cspace 기준으로 받는다 (14개)
    if len(q_list) != len(cspace_names):
        raise ValueError(f"q 길이가 cspace joint_names와 달라요. got {len(q_list)}, expected {len(cspace_names)}")

    # 모델 dof(16)에 맞는 q_full 만들기
    q_full = torch.zeros((1, robot_model.dof), dtype=torch.float32, device=tensor_args.device)

    name_to_idx = {n: i for i, n in enumerate(model_names)}

    for jname, jval in zip(cspace_names, q_list):
        if jname not in name_to_idx:
            raise ValueError(f"cspace joint '{jname}'이(가) robot_model.joint_names에 없어요. 모델 조인트: {model_names}")
        q_full[0, name_to_idx[jname]] = float(jval)

    # 이제 FK는 q_full로
    state = robot_model.get_state(q_full)

    print("=== FK Result (bimanual) ===")
    print(f"robot_yml   : {args.robot_yml}")
    print(f"device      : {q_full.device}")
    print(f"dof         : {dof}")
    print(f"joint_names : {robot_model.joint_names}")  # actuated joints :contentReference[oaicite:4]{index=4}
    print(f"q(rad)      : {q_list}")
    print(f"links       : {links_to_report}")
    print("note        : quaternion default is wxyz (qw,qx,qy,qz) :contentReference[oaicite:5]{index=5}")
    print()

    # state.link_poses: {link_name: Pose} :contentReference[oaicite:6]{index=6}
    link_poses_dict = state.link_poses or {}

    for link in links_to_report:
        if link in link_poses_dict:
            pose = link_poses_dict[link]
            pos = pose.position[0].detach().cpu().tolist()
            quat_wxyz = pose.quaternion[0].detach().cpu().tolist()
        else:
            # 혹시 link_poses에 없으면, 해당 링크를 "query"해서라도 뽑아보기
            with torch.no_grad():
                one = robot_model.get_state(q_full, link_name=link)  # link_name pose :contentReference[oaicite:7]{index=7}
            pos = one.ee_position[0].detach().cpu().tolist()
            quat_wxyz = one.ee_quaternion[0].detach().cpu().tolist()

        if args.quat_xyzw:
            # wxyz -> xyzw
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            quat_out = quat_xyzw
        else:
            quat_out = quat_wxyz

        print(f"[{link}]")
        print(f"  pos(m) : {pos}")
        print(f"  quat   : {quat_out}")
        print()

    # 참고: state.ee_position/ee_quaternion은 model_cfg.ee_link(=primary_ee) 기준 :contentReference[oaicite:8]{index=8}


if __name__ == "__main__":
    main()
