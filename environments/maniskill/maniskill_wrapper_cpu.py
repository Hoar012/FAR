import os
import sys

import gymnasium as gym
import mani_skill
import numpy as np
import torch
from gymnasium import spaces

sys.path.append(os.path.join(os.getcwd(), "environments"))
import numpy as np


class ManiskilEnvCPU:
    def __init__(
        self,
        config,
        env_name,
        max_steps,
        state_dim,
        action_dim,
        action_repeat,
    ):
        reward_mode = "dense"
        if config.high_res_render:
            env_image_size = config.highres_img_size
        else:
            env_image_size = config.image_size

        self.env = gym.make(
            env_name,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            reward_mode=reward_mode,
            render_mode="rgb_array",
            sensor_configs=dict(width=env_image_size, height=env_image_size),
            human_render_camera_configs=dict(width=384, height=384),
            reconfiguration_freq=1,
            sim_backend="physx_cpu",
        )

        self.max_steps = max_steps

        # Make observation space
        self.observation_space = spaces.Dict()
        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.observation_space["agentview_image"] = spaces.Box(
            low=0,
            high=255,
            shape=(config.image_size, config.image_size, 3),
            dtype=np.uint8,
        )
        self.observation_space["robot0_eye_in_hand_image"] = spaces.Box(
            low=0,
            high=255,
            shape=(config.image_size, config.image_size, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self._step = 0
        self.action_repeat = action_repeat
        self.config = config

    def process_obs(self, obs):
        new_obs = {}

        state_list = []
        state_list.append(obs["agent"]["qpos"])  # len x ...
        extra_keys = list(obs["extra"].keys())
        for key in extra_keys:
            datapoint = obs["extra"][key]
            if len(datapoint.shape) == 1:
                datapoint = datapoint.reshape(-1, 1)
            state_list.append(datapoint)

        new_obs["state"] = torch.cat(state_list, axis=-1).cpu().numpy()
        new_obs["agentview_image"] = (
            obs["sensor_data"]["agentview_image"]["rgb"].cpu().numpy()
        )
        new_obs["robot0_eye_in_hand_image"] = (
            obs["sensor_data"]["robot0_eye_in_hand_image"]["rgb"].cpu().numpy()
        )
        # Remove batch dimension
        for k, v in new_obs.items():
            new_obs[k] = v.squeeze(0)
        return new_obs

    def reset(self, **kwargs):
        self._step = 0
        raw_obs, _ = self.env.reset()
        obs = self.process_obs(raw_obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

    def step(self, action):
        # raw_obs, reward, terminated, truncated, info = self.env.step(action)
        action = action["action"]
        for _ in range(self.action_repeat):
            raw_obs, reward, terminated, truncated, info = self.env.step(action)

        self._step += 1

        obs = self.process_obs(raw_obs)
        reward = reward.cpu().numpy()[0]
        terminated = terminated.cpu().numpy()[0]
        truncated = truncated.cpu().numpy()[0]
        success = info["success"].cpu().numpy()[0]

        if self._step < self.max_steps:
            # Force cancel termination
            terminated = success
            truncated = success

        done = terminated or truncated
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = terminated

        info = {"orig_reward": reward, "success": success}
        return obs, reward, done, info

    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
