from collections import OrderedDict

import numpy as np
import robosuite as suite
from gym import spaces
from robosuite.wrappers import GymWrapper
from termcolor import cprint


class RoboCasaWrapper:
    def __init__(
        self, env, shape_meta, config, action_repeat, keys=None, n_succ_before_term=5
    ):
        self.env = env
        if keys is None:
            keys = ["agentview_image", "robot0_eye_in_hand_image"]

        # create action space
        self.n_succ_before_term = n_succ_before_term
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(config.action_dim,), dtype=np.float32
        )
        self.config = config

        # create observation space
        observation_space = spaces.Dict()
        # store state-based keys here instead of observation_space
        state_keys = []
        has_joint_cos, has_joint_sin = False, False
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("image"):
                min_value, max_value = 0, 255
                dtype = np.uint8
            else:
                has_joint_cos = has_joint_cos or "joint_pos_cos" in key
                has_joint_sin = has_joint_sin or "joint_pos_sin" in key
                state_keys.append(key)
                dtype = np.float32
                continue
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=dtype
            )
            observation_space[key] = this_space

        observation_space["state"] = spaces.Box(
            low=-100, high=100, shape=(config.state_dim,), dtype=np.float64
        )
        self.observation_space = observation_space

        # render_cache is used to store the last rendered image
        self.render_cache = None
        self.render_obs_key = keys[0]
        self.state_keys = state_keys
        self.convert_joint_pos = has_joint_cos and has_joint_sin
        self.action_repeat = action_repeat

    def get_observation(self, raw_obs):
        self.render_cache = raw_obs[self.render_obs_key]
        obs = OrderedDict()
        for observation_space_key in self.observation_space.keys():
            if observation_space_key == "state":
                # "state" only exists as key if self.add_state == True
                state_obs = []
                if self.convert_joint_pos:
                    joint_qpos = np.arctan2(
                        raw_obs["robot0_joint_pos_sin"], raw_obs["robot0_joint_pos_cos"]
                    )
                    state_obs.append(joint_qpos)

                for state_key in self.state_keys:
                    if not (
                        self.convert_joint_pos
                        and (
                            "joint_pos_cos" in state_key or "joint_pos_sin" in state_key
                        )
                    ):
                        state_obs.append(raw_obs[state_key])

                obs[observation_space_key] = np.concatenate(state_obs, axis=0)
            else:
                obs[observation_space_key] = raw_obs[observation_space_key]

        return obs

    def reset(self, **kwargs):
        raw_obs = self.env.reset()
        obs = self.get_observation(raw_obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self.step_count = 0
        self.success_count = 0
        self.prev_success = 0
        return obs

    def step(self, action):
        for _ in range(self.action_repeat):
            raw_obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        # Give success 1 if reward==1 for n_succ_before_term consecutive steps
        if reward >= 0.99:
            if self.prev_success == 1:
                self.success_count += 1
            else:
                self.success_count = 1
        else:
            self.success_count = 0

        self.prev_success = reward >= 0.99

        if self.success_count >= self.n_succ_before_term:
            done = True
            info["success"] = True
        else:
            info["success"] = False

        obs = self.get_observation(raw_obs)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = done
        info["orig_reward"] = reward  # Includes just reward from the environment

        # Reward Shifting
        reward = reward - 1

        return obs, reward, done, info

    def close(self):
        del self.env
