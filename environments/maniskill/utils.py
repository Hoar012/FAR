import collections
import os
from collections import defaultdict

import h5py
import numpy as np
from termcolor import cprint

from environments.maniskill.env_def import *
from environments.maniskill.maniskill_wrapper import ManiskilEnv
from environments.maniskill.maniskill_wrapper_cpu import ManiskilEnvCPU
from far.classes.rollout_utils import get_act_stacked, get_obs_stacked
from far.dreamer.tools import add_to_cache

H5DIR = {
    "pullcube": "pullcube/demo.rgb.pd_ee_delta_pose.physx_cpu.h5",
    "liftpeg": "liftpeg/demo.rgb.pd_ee_delta_pose.physx_cpu.h5",
    "pokecube": "pokecube/demo.rgb.pd_ee_delta_pose.physx_cpu.h5",
}

ENV_MAP = {
    "pullcube": "PullCubeTwoCam",
    "liftpeg": "LiftPegUprightTwoCam",
    "pokecube": "PokeCubeTwoCam",
}


def make_maniskill_env(config, suite, task):
    assert suite == "maniskill"
    env_name = ENV_MAP[task]
    max_steps = config.time_limit

    if config.use_cpu_env:
        env = ManiskilEnvCPU(
            config=config,
            env_name=env_name,
            max_steps=max_steps,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            action_repeat=config.action_repeat,
        )
    else:
        cprint("Maniskill GPU Env", "green")
        env = ManiskilEnv(
            config=config,
            env_name=env_name,
            num_envs=config.num_envs,
            max_steps=max_steps,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            action_repeat=config.action_repeat,
        )
    cprint(
        f"Initialized maniskill envs with action repeat: {config.action_repeat}, time limit: {config.time_limit}",
        "yellow",
    )
    return env


def add_traj_to_cache(traj_id, npz_data, cache, config, norm_dict=None):
    traj_data = npz_data[traj_id]

    # Concat state keys to create "state" key
    concat_state = []
    for t in range(len(traj_data["state"])):
        curr_obs_state_vec = np.array(traj_data["state"][t], dtype=np.float32)
        concat_state.append(curr_obs_state_vec)

        # Update norm_dict for the environment
        if norm_dict is not None:
            norm_dict["ob_max"] = np.maximum(norm_dict["ob_max"], curr_obs_state_vec)
            norm_dict["ob_min"] = np.minimum(norm_dict["ob_min"], curr_obs_state_vec)

    # Stack Observations for State and Pixel Keys
    stacked_obs = {}
    stacked_obs["state"] = get_obs_stacked(concat_state, config.obs_horizon)

    # Stack Actions
    stacked_acts = get_act_stacked(traj_data["action"], config.pred_horizon)

    # Stack Images
    stacked_images = {}
    stacked_images["agentview_image"] = get_obs_stacked(
        traj_data["agentview_image"], config.obs_horizon
    )
    if "robot0_eye_in_hand_image" in traj_data.keys():
        stacked_images["robot0_eye_in_hand_image"] = get_obs_stacked(
            traj_data["robot0_eye_in_hand_image"], config.obs_horizon
        )

    # Update norm_dict for the environment
    if norm_dict is not None:
        acts_np_array = np.array(traj_data["action"])
        norm_dict["ac_max"] = np.maximum(
            norm_dict["ac_max"], np.max(acts_np_array, axis=0)
        )
        norm_dict["ac_min"] = np.minimum(
            norm_dict["ac_min"], np.min(acts_np_array, axis=0)
        )

    # Fill all the transitions in the cache
    for t in range(len(traj_data["state"])):
        transition = defaultdict(np.array)
        transition["state"] = stacked_obs["state"][t]
        for key, value in stacked_images.items():
            transition[key] = value[t]
        rewards_t = traj_data["reward"][t]
        transition["success"] = traj_data["success"][t]

        # Set done to 1 at last transition
        done_t = 0 if t < len(stacked_acts) - 1 else 1

        transition["reward"] = np.array(rewards_t, dtype=np.float32)
        transition["is_first"] = np.array(t == 0, dtype=np.bool_)
        transition["action"] = stacked_acts[t]
        transition["is_last"] = np.array(done_t, dtype=np.bool_)
        transition["is_terminal"] = np.array(done_t, dtype=np.bool_)
        add_to_cache(cache, f"exp_traj_{traj_id}", transition)


def clip_data_to_first_5_success(npz_traj_data):
    success = npz_traj_data["success"]

    # Find all indices where success is True
    true_indices = np.where(success == True)[0]

    # Iterate forwards to find the first point where there are 5 continuous True values
    for i in range(len(true_indices) - 4):
        if np.all(success[true_indices[i] : true_indices[i] + 5]):
            end_timestep = true_indices[i] + 5
            break
    else:
        # If no such sequence is found, keep the full trajectory
        end_timestep = len(success)

    # Clip the trajectory
    for key in npz_traj_data.keys():
        npz_traj_data[key] = npz_traj_data[key][:end_timestep]

    return npz_traj_data


def get_train_val_datasets_maniskill(config):
    num_exp_trajs = config.num_exp_trajs
    num_val_trajs = config.num_exp_val_trajs

    suite, task = config.task.split("__", 1)
    task = task.lower()
    assert suite == "maniskill"
    cprint(
        f"Loading {num_exp_trajs} train and {num_val_trajs} val trajs for {suite}:{task}",
        "green",
    )

    h5_file_name = os.path.join(config.datadir, H5DIR[task])
    h5_file = h5py.File(h5_file_name, "r")
    cprint(f"Total Number of demos: {len(h5_file.keys())}", "green")

    # Print keys in first h5 demo sensor_data
    first_demo = h5_file[list(h5_file.keys())[0]]
    cprint(
        f"Keys in first demo sensor_data: {first_demo['obs']['sensor_data'].keys()}",
        "green",
    )
    cprint(f"Action repeat set to {config.action_repeat}", "green")

    # Convert to how npz would look like. Keep keys
    npz_data = []
    for count, demo_key in enumerate(h5_file.keys()):
        if len(npz_data) >= num_exp_trajs + num_val_trajs:
            break
        h5_traj_data = h5_file[demo_key]

        npz_traj_data = {}

        # Build the state
        state_list = []
        state_list.append(np.array(h5_traj_data["obs"]["agent"]["qpos"]))  # len x ...
        extra_keys = list(h5_traj_data["obs"]["extra"].keys())
        for key in extra_keys:
            datapoint = h5_traj_data["obs"]["extra"][key]
            if len(datapoint.shape) > 1:
                state_list.append(np.array(h5_traj_data["obs"]["extra"][key]))
            else:
                state_list.append(np.array(h5_traj_data["obs"]["extra"][key])[:, None])
        npz_traj_data["state"] = np.concatenate(state_list, axis=-1)
        traj_len = len(npz_traj_data["state"]) // config.action_repeat

        # Skip trajectories that are too long (some issue in collecting data)
        if traj_len > 1.5 * config.time_limit:
            print(
                f"Trajectory {demo_key} is too long, skipping. Max allowed: {config.time_limit*1.5}, length: {traj_len}"
            )
            continue

        npz_traj_data["action"] = h5_traj_data["actions"]
        npz_traj_data["reward"] = h5_traj_data["rewards"]
        npz_traj_data["agentview_image"] = h5_traj_data["obs"]["sensor_data"][
            "agentview_image"
        ]["rgb"]
        npz_traj_data["robot0_eye_in_hand_image"] = h5_traj_data["obs"]["sensor_data"][
            "robot0_eye_in_hand_image"
        ]["rgb"]

        npz_traj_data["success"] = np.array(h5_traj_data["success"])

        # Find the last timestep that success is True for 5 continuous steps, clip to that
        npz_traj_data = clip_data_to_first_5_success(npz_traj_data)

        # Skip every ACTION_REPEAT_MANISKILL frames if that is not 1
        if config.action_repeat > 1:
            for key, value in npz_traj_data.items():
                value_np = np.array(value)
                value_np_skipped = value_np[:: config.action_repeat, ...]
                npz_traj_data[key] = value_np_skipped

        npz_data.append(npz_traj_data)

    # Print average length of the trajectories
    total_length = 0
    for traj in npz_data:
        total_length += len(traj["state"])
    avg_length = total_length / len(npz_data)
    print(f"Average length of the trajectories: {avg_length}")
    print(
        f"Minimum length of the trajectories: {min([len(traj['state']) for traj in npz_data])}"
    )
    print(
        f"Maximum length of the trajectories: {max([len(traj['state']) for traj in npz_data])}"
    )

    # Assert if we have enough data
    assert len(npz_data) >= num_exp_trajs + num_val_trajs, (
        f"Not enough data! Found {len(npz_data)} expert trajectories, "
        f"but config specifies {num_exp_trajs} training and "
        f"{num_val_trajs} validation episodes."
    )

    cprint(f"Adding {len(npz_data)} trajectories to cache", "green")

    state_dim = npz_data[0]["state"].shape[-1]
    action_dim = npz_data[0]["action"].shape[-1]

    norm_dict = {
        "ob_max": -np.inf * np.ones(state_dim, dtype=np.float32),
        "ob_min": np.inf * np.ones(state_dim, dtype=np.float32),
        "ac_max": -np.inf * np.ones(action_dim, dtype=np.float32),
        "ac_min": np.inf * np.ones(action_dim, dtype=np.float32),
    }

    train_expert_eps = collections.OrderedDict()
    val_expert_eps = collections.OrderedDict()

    for i in range(num_exp_trajs):
        add_traj_to_cache(
            traj_id=i,
            npz_data=npz_data,
            cache=train_expert_eps,
            config=config,
            norm_dict=norm_dict,
        )
    print("Loaded", len(train_expert_eps.keys()), "training episodes")

    for i in range(num_val_trajs):
        add_traj_to_cache(
            traj_id=i + num_exp_trajs,
            npz_data=npz_data,
            cache=val_expert_eps,
            config=config,
        )

    print("Loaded", len(val_expert_eps.keys()), "validation episodes")

    return train_expert_eps, val_expert_eps, norm_dict, state_dim, action_dim
