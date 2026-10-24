import contextlib
import copy
import gc
import os
from collections import defaultdict

import numpy as np

from environments.global_utils import resize_to_given_size


# This class stacks things in batch dim and returns multiple envs
class ConcurrentEnvs:
    def __init__(self, config, env_make, num_envs):
        self.num_envs = num_envs
        self.envs = [env_make(config) for _ in range(num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.config = config

        self.reset()

    def process_obs(self, obs):
        if self.config.high_res_render:
            obs["agentview_image_highres"] = copy.deepcopy(obs["agentview_image"])
            obs["robot0_eye_in_hand_image_highres"] = copy.deepcopy(
                obs["robot0_eye_in_hand_image"]
            )
            obs["agentview_image"] = resize_to_given_size(
                obs["agentview_image"], self.config.image_size
            )
            obs["robot0_eye_in_hand_image"] = resize_to_given_size(
                obs["robot0_eye_in_hand_image"], self.config.image_size
            )
        return obs

    def reset(self):
        self.is_env_done = [False] * len(self.envs)

        # Reset the environment
        reset_states = defaultdict(list)
        for env in self.envs:
            for k, v in env.reset().items():
                reset_states[k].append(v)

        # Get stacked obs as a dict, record zero obs for padding in step
        self.zero_obs = {}  # zero obs for padding
        obs = {}
        for k, v in reset_states.items():
            self.zero_obs[k] = np.zeros_like(v[0])
            obs[k] = np.stack(v, axis=0)

        self.obs_keys = list(obs.keys())

        return self.process_obs(obs)

    def step(self, actions):
        all_obs = defaultdict(
            list
        )  # each key is a list of observations of length num_envs
        all_rewards, all_dones, all_successes, all_orig_rewards = [], [], [], []
        for i, env in enumerate(self.envs):
            if self.is_env_done[i]:
                # Append dummy values in all obs keys
                for k in self.obs_keys:
                    if self.prev_obs is not None:
                        all_obs[k].append(self.prev_obs[k][i])
                    else:
                        all_obs[k].append(self.zero_obs[k])

                all_rewards.append(0.0)
                all_dones.append(True)
                all_successes.append(False)
                all_orig_rewards.append(0.0)
                continue
            else:
                obs, reward, done, info = env.step({"action": actions[i]})
                for k in self.obs_keys:
                    all_obs[k].append(obs[k])

                all_rewards.append(reward)
                all_dones.append(done)
                all_successes.append(info["success"])
                all_orig_rewards.append(
                    info["orig_reward"] if "orig_reward" in info else reward
                )
                if done:
                    self.is_env_done[i] = True

        self.prev_obs = copy.deepcopy(all_obs)

        # Stack all obs, rewards, and dones
        for k in self.obs_keys:
            all_obs[k] = np.stack(all_obs[k], axis=0)

        all_rewards = np.stack(all_rewards, axis=0)
        all_dones = np.stack(all_dones, axis=0)
        all_successes = np.stack(all_successes, axis=0)
        all_orig_rewards = np.stack(all_orig_rewards, axis=0)
        return (
            self.process_obs(all_obs),
            all_rewards,
            all_dones,
            {"success": all_successes, "orig_reward": all_orig_rewards},
        )

    def close(self):
        for env in self.envs:
            env.close()

        self._suppress_egl_error()

    def _suppress_egl_error(self):
        """
        Suppresses EGL error messages from being printed to stderr.
        """
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            gc.collect()  # Force garbage collection to run
