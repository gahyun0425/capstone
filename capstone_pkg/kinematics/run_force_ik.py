from capstone_pkg.kinematics.force_curobo_ik import ForceCuroboIK
from capstone_pkg.utils.config import ROBOT_YAML, ROBOT_URDF, WORLD_YAML

solver = ForceCuroboIK(
    robot_yml=ROBOT_YAML,
    urdf_path=ROBOT_URDF,
    cpu=False,
    world_yml=WORLD_YAML,
)

out = solver.solve_max_forward_force(
    left_xyz=[0.4, 0.2, 1.2],
    left_quat_wxyz=[0.5, 0.5, 0.5, 0.5],
    right_xyz=[0.4, -0.2, 1.2],
    right_quat_wxyz=[0.5, -0.5, 0.5, -0.5],
    q_start_cspace=[0.0] * 14,
    forward_direction_base=(1.0, 0.0, 0.0),
    num_trials=24,
    seed_noise_std=0.25,
    random_seed=0,
    balance_weight=1.0,
    refine_best=True,
    refine_top_k=3,
    refinement_steps=20,
    refinement_step_size=0.12,
    gradient_eps=1.0e-3,
    position_tolerance=2.0e-3,
    rotation_tolerance_rad=2.0e-2,
)

print(out)
