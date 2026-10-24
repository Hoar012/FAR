import collections
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from environments.robomimic.constants import STATE_SHAPE_META
from far.classes.rollout_utils import get_act_stacked, get_obs_stacked
from far.dreamer.tools import add_to_cache


class HiddenPrints:
    """
    Suppress print output.
    """

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


def to_np(x):
    return x.detach().cpu().numpy()


def get_robomimic_dataset_path_and_env_meta(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
    datadir="/home/dreamerv3/robomimic_datasets",  # Default for docker/singularity
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    assert int(done_mode) in [0, 1, 2]

    dataset_name = obs_type
    if image_size != 0:
        dataset_name += f"_{image_size}"
    if shaped:
        dataset_name += "_shaped"
    dataset_name += f"_done{done_mode}"
    dataset_path = f"{env_id.lower()}/{collection_type}/{dataset_name}_v141.hdf5"

    root_dir = datadir
    dataset_path = Path(root_dir, dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    return dataset_path, env_meta


def evaluate(agent, eval_env, video_env, num_episodes, num_episodes_to_record=1):
    """
    Evaluate the policy in environment, record video of first episode.
    """
    success = 0
    for i in trange(num_episodes, desc="Eval rollouts", ncols=0, leave=False):
        if num_episodes_to_record > 0:
            curr_env = video_env
            num_episodes_to_record -= 1
        else:
            curr_env = eval_env

        observation, done = curr_env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, _ = curr_env.step(action)
        if float(reward) == 0:
            success += 1
    return {
        "return": np.mean(eval_env.first_env.return_queue),
        "length": np.mean(eval_env.first_env.length_queue),
        "success": success / num_episodes,
    }


def create_shape_meta(img_size, include_state):
    shape_meta = {
        "obs": {
            "agentview_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
        },
        "action": {"shape": [7]},
    }
    if include_state:
        shape_meta["obs"].update(STATE_SHAPE_META)
    return shape_meta


def get_dataset_path_and_meta_info(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
    datadir="/home/dreamerv3/robomimic_datasets",
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id,
        collection_type=collection_type,
        obs_type=obs_type,
        shaped=shaped,
        image_size=image_size,
        done_mode=done_mode,
        datadir=datadir,
    )
    shape_meta = create_shape_meta(image_size, include_state=True)
    return dataset_path, env_meta, shape_meta


def add_traj_to_cache(
    traj_id, demo, cache, f, config, pixel_keys, state_keys, norm_dict=None
):
    traj = f["data"][demo]

    # Concat state keys to create "state" key
    concat_state = []
    for t in range(len(traj["obs"][pixel_keys[0]])):
        curr_obs_state_vec = [traj["obs"][obs_key][t] for obs_key in state_keys]
        curr_obs_state_vec = np.concatenate(curr_obs_state_vec, dtype=np.float32)
        concat_state.append(curr_obs_state_vec)

        # Update norm_dict for the environment
        if norm_dict is not None:
            norm_dict["ob_max"] = np.maximum(norm_dict["ob_max"], curr_obs_state_vec)
            norm_dict["ob_min"] = np.minimum(norm_dict["ob_min"], curr_obs_state_vec)

    # Stack Observations for State and Pixel Keys
    stacked_obs = {}
    stacked_obs["state"] = get_obs_stacked(concat_state, config.obs_horizon)
    for key in pixel_keys:
        stacked_obs[key] = get_obs_stacked(traj["obs"][key], config.obs_horizon)

    # Stack Actions
    stacked_acts = get_act_stacked(traj["actions"], config.pred_horizon)

    # Update norm_dict for the environment
    if norm_dict is not None:
        acts_np_array = np.array(traj["actions"])
        norm_dict["ac_max"] = np.maximum(
            norm_dict["ac_max"], np.max(acts_np_array, axis=0)
        )
        norm_dict["ac_min"] = np.minimum(
            norm_dict["ac_min"], np.min(acts_np_array, axis=0)
        )

    # Fill all the transitions in the cache
    for t in range(len(traj["obs"][pixel_keys[0]])):
        transition = defaultdict(np.array)
        for obs_key in pixel_keys:
            transition[obs_key] = stacked_obs[obs_key][t]

        transition["state"] = stacked_obs["state"][t]
        rewards_t = traj["rewards"][t]
        done_t = 1 if t == len(traj["obs"][pixel_keys[0]]) - 1 else 0
        transition["success"] = rewards_t == 1  # success condition for robomimic

        # Reward shifting by max value
        rewards_t = np.array(traj["rewards"][t], dtype=np.float32) - 1

        # Insert into the transition dict
        transition["reward"] = np.array(rewards_t, dtype=np.float32)
        transition["is_first"] = np.array(t == 0, dtype=np.bool_)
        transition["action"] = stacked_acts[t]
        transition["is_last"] = np.array(done_t, dtype=np.bool_)
        transition["is_terminal"] = np.array(done_t, dtype=np.bool_)
        add_to_cache(cache, f"exp_traj_{traj_id}", transition)


def get_train_val_datasets(config):
    num_train_trajs = config.num_exp_trajs
    num_val_trajs = config.num_exp_val_trajs

    suite, task = config.task.split("__", 1)
    task = task.lower()
    assert suite == "robomimic"

    train_eps = collections.OrderedDict()
    val_eps = collections.OrderedDict()

    # Load the h5py files
    dataset_path, _, shape_meta = get_dataset_path_and_meta_info(
        env_id=task,
        shaped=config.shape_rewards,
        image_size=config.image_size,
        done_mode=config.done_mode,
        datadir=config.datadir,
    )

    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())

    # Assert that we have enough data
    # print(len(demos), "expert trajectories found in dataset")
    assert num_train_trajs + num_val_trajs <= len(demos), "Not enough expert data"

    obs_keys = shape_meta["obs"].keys()
    pixel_keys = sorted([key for key in obs_keys if "image" in key])
    state_keys = sorted([key for key in obs_keys if "image" not in key])

    # Initialize norm_dict
    # Read ob_dim and ac_dim from the first datapoint in the first demo
    first_demo = f["data"][demos[0]]
    ob_dim = 0
    for key in state_keys:
        ob_dim += np.prod(first_demo["obs"][key].shape[1:])
    ac_dim = first_demo["actions"].shape[1]

    print(f"Initizalizing norm_dict with ob_dim={ob_dim} and ac_dim={ac_dim}")
    norm_dict = {
        "ob_max": -np.inf * np.ones(ob_dim, dtype=np.float32),
        "ob_min": np.inf * np.ones(ob_dim, dtype=np.float32),
        "ac_max": -np.inf * np.ones(ac_dim, dtype=np.float32),
        "ac_min": np.inf * np.ones(ac_dim, dtype=np.float32),
    }

    # Set state_dim and action_dim
    state_dim = 0
    for key in state_keys:
        state_dim += np.prod(first_demo["obs"][key].shape[1:])

    action_dim = first_demo["actions"].shape[1]

    # Fill the Train Dataset
    for i in range(num_train_trajs):
        demo = demos[i]
        add_traj_to_cache(
            i, demo, train_eps, f, config, pixel_keys, state_keys, norm_dict
        )

    print("Loaded", len(train_eps.keys()), "training episodes")

    # Fill the Val Dataset
    for i in range(num_train_trajs, num_train_trajs + num_val_trajs):
        demo = demos[i]
        add_traj_to_cache(i, demo, val_eps, f, config, pixel_keys, state_keys)

    print("Loaded", len(val_eps.keys()), "validation episodes")

    # Close the h5py file
    f.close()

    # Print min, max and avg expert demo length
    traj_lens = [len(train_eps[key]["state"]) for key in train_eps.keys()]
    traj_lens = np.array(traj_lens)
    print("Min expert demo length:", np.min(traj_lens))
    print("Max expert demo length:", np.max(traj_lens))
    print("Avg expert demo length:", np.mean(traj_lens))

    return train_eps, val_eps, norm_dict, state_dim, action_dim
