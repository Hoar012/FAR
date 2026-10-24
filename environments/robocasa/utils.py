import collections
import os

import h5py
import numpy as np
from termcolor import cprint

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
from environments.robocasa.additional_envs import *
from environments.robocasa.robocasa_wrapper import RoboCasaWrapper
from environments.robomimic.utils import add_traj_to_cache, create_shape_meta
from far.dreamer.tools import set_seed_everywhere


def sanitize_for_robomimic(config):
    if "layout_ids" in config:
        del config["layout_ids"]
    if "style_ids" in config:
        del config["style_ids"]
    if "obj_groups" in config:
        del config["obj_groups"]
    if "translucent_robot" in config:
        del config["translucent_robot"]
    if "obj_instance_split" in config:
        del config["obj_instance_split"]
    return config


def get_env_details(config, suite, task):
    # image_64_shaped_done1_v141.hdf5
    hdf5_name = f"image_{config.image_size}_shaped_done1_v141.hdf5"
    dataset_path = os.path.join(config.datadir, task.lower(), hdf5_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    if task.lower() in ["stack", "door"]:
        env_meta["env_kwargs"] = sanitize_for_robomimic(env_meta["env_kwargs"])

    shape_meta = create_shape_meta(
        img_size=config.image_size,
        include_state=True,
    )
    return dataset_path, env_meta, shape_meta


def make_env_robocasa(config, suite, task):
    assert suite == "robocasa", f"Only robocasa is supported, but got {suite}"

    _, env_meta, shape_meta = get_env_details(config, suite, task)
    if config.high_res_render:
        camera_shape = config.highres_img_size
    else:
        camera_shape = config.image_size

    # Set deterministic forward pass
    env_meta["env_kwargs"]["lite_physics"] = False

    set_seed_everywhere(config.seed)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_height=camera_shape,
        camera_width=camera_shape,
        reward_shaping=True,
    )
    cprint(
        f"Initialized robocasa env with action repeat: {config.action_repeat}, time limit: {config.time_limit}",
        "yellow",
    )
    return RoboCasaWrapper(
        env=env,
        shape_meta=shape_meta,
        config=config,
        action_repeat=config.action_repeat,
    )


def get_index_of_first_non_zero_velocity(avg_joint_vel):
    # Smoothen the average joint velocity
    avg_joint_vel = np.convolve(avg_joint_vel, np.ones(10) / 10, mode="valid")

    # Get the index of the first non-zero velocity
    first_ind = np.where(np.abs(avg_joint_vel) > 0.01)[0][0]
    return first_ind


def get_train_val_datasets(config):
    num_train_trajs = config.num_exp_trajs
    num_val_trajs = config.num_exp_val_trajs
    action_repeat = config.action_repeat

    suite, task = config.task.split("__", 1)
    assert suite == "robocasa"

    train_eps = collections.OrderedDict()
    val_eps = collections.OrderedDict()

    # Load the h5py files
    dataset_path, env_meta, shape_meta = get_env_details(config, suite, task)

    h5py_file = h5py.File(dataset_path, "r")
    demos = list(h5py_file["data"].keys())

    # Assert that we have enough data
    assert num_train_trajs + num_val_trajs <= len(
        demos
    ), f"Not enough expert data, requested {num_train_trajs} + {num_val_trajs} = {num_train_trajs + num_val_trajs} but only {len(demos)} available"

    # Apply velocity filter and action repeat
    new_data_dict = {"data": {}}
    ii = 0
    clean_demos = []
    while len(clean_demos) < num_train_trajs + num_val_trajs:
        # Find index of timestep where velocity is non-zero
        avg_joint_vel = np.array(
            h5py_file["data"][demos[ii]]["obs"]["robot0_joint_vel"]
        ).mean(axis=1)
        first_ind = get_index_of_first_non_zero_velocity(avg_joint_vel)

        # If number of timesteps with zero velocity is greater than 5, trim it to 5
        # 5 is chosen here as most correct demos have about 5 timesteps of zero velocity
        if first_ind > 5:
            print(
                f"Zero actions detected at demo: {demos[ii]}, trimming first {first_ind-5} timesteps"
            )
            first_ind = max(first_ind - 5, 0)
        else:
            first_ind = 0

        demo_orig = h5py_file["data"][demos[ii]]
        demo_new = {demos[ii]: {}}

        # Apply action repeat to demo_orig["obs"]
        demo_new[demos[ii]]["obs"] = {}
        for key in demo_orig["obs"].keys():
            npz_data = np.array(demo_orig["obs"][key])
            demo_new[demos[ii]]["obs"][key] = npz_data[first_ind:][::action_repeat]

        # Apply action repeat to demo_orig["actions"]
        demo_new[demos[ii]]["actions"] = np.array(demo_orig["actions"])[first_ind:][
            ::action_repeat
        ]

        # Apply action repeat to demo_orig["rewards]
        demo_new[demos[ii]]["rewards"] = np.array(demo_orig["rewards"])[first_ind:][
            ::action_repeat
        ]

        # Apply action repeat to demo_orig["dones"]
        demo_new[demos[ii]]["dones"] = np.array(demo_orig["dones"])[first_ind:][
            ::action_repeat
        ]

        new_data_dict["data"].update(demo_new)
        clean_demos.append(demos[ii])
        ii += 1

    # Close the h5py file
    h5py_file.close()

    obs_keys = shape_meta["obs"].keys()
    pixel_keys = sorted([key for key in obs_keys if "image" in key])
    state_keys = sorted([key for key in obs_keys if "image" not in key])

    # Initialize norm_dict
    # Read ob_dim and ac_dim from the first datapoint in the first demo
    first_demo = new_data_dict["data"][clean_demos[0]]
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
    for ii in range(num_train_trajs):
        demo = clean_demos[ii]
        add_traj_to_cache(
            ii,
            demo,
            train_eps,
            new_data_dict,
            config,
            pixel_keys,
            state_keys,
            norm_dict,
        )

    # Compute average length in data in train_eps
    lengths = [len(ep["state"]) for ep in train_eps.values()]
    print(
        "Min length:",
        min(lengths),
        "Max length:",
        max(lengths),
        "Mean length:",
        np.mean(lengths),
    )
    print(
        "Loaded",
        len(train_eps.keys()),
        "training episodes, action_repeat=",
        action_repeat,
    )

    # Fill the Val Dataset
    for ii in range(num_train_trajs, num_train_trajs + num_val_trajs):
        demo = clean_demos[ii]
        add_traj_to_cache(
            ii, demo, val_eps, new_data_dict, config, pixel_keys, state_keys
        )
    print(
        "Loaded",
        len(val_eps.keys()),
        "validation episodes, action_repeat=",
        action_repeat,
    )

    return train_eps, val_eps, norm_dict, state_dim, action_dim
