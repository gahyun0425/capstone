from __future__ import annotations
from pathlib import Path
import os

ROBOT_XML = "/home/gaga/capstone_ws/src/capstone_pkg/models/ffw_sg2.xml"
ROBOT_URDF = "/home/gaga/capstone_ws/src/capstone_pkg/models/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
ROBOT_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/test_curobo.yaml"
WORLD_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/world_collision.yaml"
SHELF_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/shelf_collision.yaml"
LONG_SHELF_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/long_shelf_collision.yaml"
CART_YAML = "/home/gaga/capstone_ws/src/capstone_pkg/models/cart_collision.yaml"



BASE_FRAME = "base_link"
# LEFT_EE_FRAME  = "gripper_l_rh_p12_rn_base"
# RIGHT_EE_FRAME = "gripper_r_rh_p12_rn_base"

LEFT_EE_FRAME = "gripper_l_tcp"
RIGHT_EE_FRAME = "gripper_r_tcp"


LEFT_PREFIX  = "arm_l_joint"
RIGHT_PREFIX = "arm_r_joint"

LEFT_JOINTS = [
    "arm_l_joint1","arm_l_joint2","arm_l_joint3","arm_l_joint4","arm_l_joint5","arm_l_joint6","arm_l_joint7",
]
RIGHT_JOINTS = [
    "arm_r_joint1","arm_r_joint2","arm_r_joint3","arm_r_joint4","arm_r_joint5","arm_r_joint6","arm_r_joint7",
]

# # 환경변수로 덮어쓰기 가능하게(선택)
# ROBOT_XML = Path(os.environ.get("PARALLEL_TB_WORLD_YAML", str(DEFAULT_ROBOT_XML)))
# ROBOT_URDF = Path(os.environ.get("PARALLEL_TB_WORLD_YAML", str(DEFAULT_ROBOT_URDF)))
# ROBOT_YAML = Path(os.environ.get("PARALLEL_TB_ROBOT_YAML", str(DEFAULT_ROBOT_YAML)))
# WORLD_YAML = Path(os.environ.get("PARALLEL_TB_ROBOT_YAML", str(DEFAULT_WORLD_YAML)))



# # 문자열이 필요한 코드용
# ROBOT_XML_STR = str(ROBOT_XML)
# ROBOT_URDF_STR = str(ROBOT_URDF)
# ROBOT_YAML_STR = str(ROBOT_YAML)
# WORLD_YAML_STR = str(WORLD_YAML)


JOINT_LIMIT = "/home/gaga/capstone_ws/src/capstone_pkg/models/joint_limits.yaml"
DEFAULT_ROBOT_YAML = ROBOT_YAML
DEFAULT_WORLD_YAML = WORLD_YAML
DEFAULT_SHELF_YAML = SHELF_YAML
DEFAULT_CART_YAML = CART_YAML
DEFAULT_JOINT_LIMIT = JOINT_LIMIT
DEFAULT_MODEL = ROBOT_XML
LEFT_GRIPPER = LEFT_EE_FRAME
RIGHT_GRIPPER = RIGHT_EE_FRAME
CSPACE_JOINT_NAMES_14 = LEFT_JOINTS + RIGHT_JOINTS
ROBOT_YAML_STR = str(ROBOT_YAML)
WORLD_YAML_STR = str(WORLD_YAML)
SHELF_YAML_STR = str(SHELF_YAML)
LONG_SHELF_YAML_STR = str(LONG_SHELF_YAML)
CART_YAML_STR = str(CART_YAML)
JOINT_LIMIT_STR = str(JOINT_LIMIT)
MODEL_STR = str(ROBOT_XML)
