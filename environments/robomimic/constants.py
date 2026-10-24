"""
Shared constants/methods across environments
"""

from pathlib import Path

LOW_DIM_OBS_KEYS = [
    "object",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]

IMAGE_OBS_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
]


STATE_SHAPE_META = {
    # "robot0_joint_pos_cos": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    # "robot0_joint_pos_sin": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    "robot0_eef_pos": {
        "shape": [3],
        "type": "low_dim",
    },
    "robot0_eef_quat": {
        "shape": [4],
        "type": "low_dim",
    },
    "robot0_gripper_qpos": {
        "shape": [2],
        "type": "low_dim",
    },
}
