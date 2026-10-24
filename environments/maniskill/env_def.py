from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.envs.tasks.tabletop.pull_cube import PullCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat

IMAGE_SIZE = 64

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_pose_180_z(p, q):
    # Rotation of 180 degrees around X-axis
    q_180_x = R.from_euler("z", 180, degrees=True)

    # Original rotation
    r = R.from_quat(q)

    # Compose rotations
    new_r = q_180_x * r
    new_q = new_r.as_quat()
    return np.array(p)[0], new_q[0]


@register_env("PullCubeTwoCam", max_episode_steps=50)
class PullCubeTwoCamEnv(PullCubeEnv):

    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose2 = look_at([0.0, 0.0, 0.0], [-0.2, 0.0, 0.4])
        pose3 = look_at([0.0, 0.0, 0.0], [1.0, 0.0, 0.3])
        q = pose2.q.squeeze().tolist() if isinstance(pose2.q, torch.Tensor) else pose2.q
        pose_base = look_at(eye=[0.3, 0, 0.3], target=[-0.1, 0, 0.1])
        pose_ext = look_at(eye=[0.2, 0.4, 0.1], target=[-0.1, 0.0, 0.1])
        return [
            CameraConfig(
                uid="robot0_eye_in_hand_image",
                pose=sapien.Pose(p=[0.1, 0, -0.1], q=q),
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=1.2 * np.pi / 2,
                near=0.01,
                far=10,
                shader_pack="default",
                mount=self.agent.robot.links_map["panda_hand_tcp"],
            ),
            CameraConfig(
                "agentview_image",
                pose=pose_ext,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=1,
                near=0.01,
                far=100,
                shader_pack="default",
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return self._default_sensor_configs


@register_env("LiftPegUprightTwoCam", max_episode_steps=100)
class LiftPegUprightTwoCamEnv(LiftPegUprightEnv):
    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose2 = look_at([0.0, 0.0, 0.0], [-0.2, 0.0, 0.4])
        q = pose2.q.squeeze().tolist() if isinstance(pose2.q, torch.Tensor) else pose2.q
        pose_top = look_at(eye=[0.3, 0.4, 0.3], target=[0, 0, 0.2])
        return [
            CameraConfig(
                "agentview_image",
                pose=pose_top,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack="default",
            ),
            CameraConfig(
                uid="robot0_eye_in_hand_image",
                pose=sapien.Pose(p=[0.1, 0, -0.1], q=q),
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=1.2 * np.pi / 2,
                near=0.01,
                far=10,
                mount=self.agent.robot.links_map["panda_hand_tcp"],
                shader_pack="default",
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return self._default_sensor_configs


@register_env("PokeCubeTwoCam", max_episode_steps=50)
class PokeCubeTwoCamEnv(PokeCubeEnv):

    @property
    def _default_sensor_configs(self):
        pose_top = look_at(eye=[0.3, 0.4, 0.3], target=[0.05, 0, 0.2])

        pose_topdown = look_at([0.1, 0, -0.11], [0.11, 0, 0.2])
        inv_p, inv_q = rotate_pose_180_z(pose_topdown.p, pose_topdown.q)
        pose_topdown = sapien.Pose(p=inv_p, q=inv_q)
        return [
            CameraConfig(
                "agentview_image",
                pose=pose_top,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack="default",
            ),
            CameraConfig(
                uid="robot0_eye_in_hand_image",
                pose=pose_topdown,
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                fov=1.2 * np.pi / 2,
                near=0.01,
                far=10,
                mount=self.agent.robot.links_map["panda_hand_tcp"],
                shader_pack="default",
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return self._default_sensor_configs
