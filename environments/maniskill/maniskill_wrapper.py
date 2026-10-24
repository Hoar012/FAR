import os
import sys

import gymnasium as gym
import mani_skill
import numpy as np
import torch
from gymnasium import spaces

sys.path.append(os.path.join(os.getcwd(), "environments"))
import numpy as np


class ManiskilEnv:
    def __init__(
        self,
        config,
        env_name,
        num_envs,
        max_steps,
        state_dim,
        action_dim,
        action_repeat,
    ):
        reward_mode = "dense"

        self.env = gym.make(
            env_name,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            num_envs=num_envs,
            reward_mode=reward_mode,
            render_mode="rgb_array",
            sensor_configs=dict(width=config.image_size, height=config.image_size),
            human_render_camera_configs=dict(width=384, height=384),
            reconfiguration_freq=1 if num_envs > 1 else None,
            sim_backend="gpu",
        )
        self.num_envs = num_envs
        self.max_steps = max_steps

        # Make observation space
        self.observation_space = spaces.Dict()
        env_space = self.env.observation_space
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
        return new_obs

    def reset(self, **kwargs):
        self._step = 0
        raw_obs, _ = self.env.reset()
        obs = self.process_obs(raw_obs)
        obs["is_first"] = np.ones(self.num_envs, dtype=np.bool_)
        obs["is_last"] = np.zeros(self.num_envs, dtype=np.bool_)
        obs["is_terminal"] = np.zeros(self.num_envs, dtype=np.bool_)
        return obs

    def step(self, action):
        # raw_obs, reward, terminated, truncated, info = self.env.step(action)
        for _ in range(self.action_repeat):
            raw_obs, reward, terminated, truncated, info = self.env.step(action)

        self._step += 1

        obs = self.process_obs(raw_obs)
        reward = reward.cpu().numpy()
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()
        success = info["success"].cpu().numpy()

        if self._step < self.max_steps:
            # Force cancel termination
            # terminated = truncated = np.zeros(self.num_envs, dtype=np.bool_)
            # If success, set as terminated and truncated, else set as all zeros
            terminated = np.logical_or(np.zeros(self.num_envs, dtype=np.bool_), success)
            truncated = np.logical_or(np.zeros(self.num_envs, dtype=np.bool_), success)

        done = np.logical_or(terminated, truncated)
        obs["is_first"] = np.zeros(self.num_envs, dtype=np.bool_)
        obs["is_last"] = done
        obs["is_terminal"] = terminated

        info = {"orig_reward": reward, "success": success}
        return obs, reward, done, info

    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
