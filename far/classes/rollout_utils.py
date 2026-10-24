import copy
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import einops
import numpy as np
import torch
from termcolor import cprint

from far.dreamer.tools import add_to_cache


def get_obs_stacked(obs_list, obs_horizon):
    # maintain the last obs_horizon observations, repeat the first observation if there are not enough
    obs_stacked = []  # rolling buffer
    all_obs_stacked = []  # list of all stacked observations

    # fill the first observation obs_horizon times
    for i in range(obs_horizon - 1):
        obs_stacked.append(obs_list[0])

    for i in range(len(obs_list)):
        # insert current observation
        obs_stacked.append(obs_list[i])

        # push to all_obs_stacked
        if obs_horizon == 1:
            all_obs_stacked.append(np.expand_dims(obs_stacked[-1], axis=-1))
        else:
            all_obs_stacked.append(np.stack(obs_stacked, axis=-1))

        # remove the oldest observation
        obs_stacked.pop(0)

    return all_obs_stacked


def get_act_stacked(act_list, pred_horizon):
    # append the current and next pred_horizon-1 actions
    # for the last pred_horizon-1 actions, repeat the last action
    act_stacked = []  # rolling buffer
    all_act_stacked = []  # list of all stacked actions

    # fill in the first pred_horizon-1 actions
    for i in range(pred_horizon):
        if i < len(act_list):
            act_stacked.append(act_list[i])
        else:
            act_stacked.append(act_list[-1])

    for i in range(pred_horizon, len(act_list) + pred_horizon):
        # push to all_act_stacked
        if pred_horizon == 1:
            all_act_stacked.append(np.expand_dims(act_stacked[-1], axis=-1))
        else:
            all_act_stacked.append(np.stack(act_stacked, axis=-1))

        # pop the oldest action
        act_stacked.pop(0)

        # append the current action (or last action)
        if i < len(act_list):
            act_stacked.append(act_list[i])
        else:
            act_stacked.append(act_list[-1])

    return all_act_stacked


def add_env_obs_to_dict(
    obs,
    obs_traj: dict,
    base_action,
    residual_action,
    action,
    rewards,
    dones,
    pixel_keys: list,
    step_idx,
    max_traj_len,
):
    """
    obs shape: num_envs x ...
    obs_traj: List of len num_envs of dictionaries
    """
    num_envs = len(obs_traj)
    assert (
        num_envs == obs["state"].shape[0]
    ), "Number of envs should be same as first dim of obs input"

    for env_idx in range(num_envs):
        obs_traj_env = obs_traj[env_idx]

        # If the previous entry for this environment had is_terminal as True, then skip this environment
        if len(obs_traj_env["is_terminal"]) > 0 and obs_traj_env["is_terminal"][-1]:
            continue

        obs_traj_env["state"].append(obs["state"][env_idx])
        obs_traj_env["base_action"].append(base_action[env_idx])
        obs_traj_env["residual_action"].append(residual_action[env_idx])
        obs_traj_env["action"].append(action[env_idx])
        obs_traj_env["reward"].append(rewards[env_idx])
        obs_traj_env["is_first"].append(step_idx == 0)
        obs_traj_env["is_last"].append(step_idx == max_traj_len - 1)
        obs_traj_env["is_terminal"].append(dones[env_idx])
        for key in pixel_keys:
            obs_traj_env[key].append(obs[key][env_idx])


def save_collected_traj_video(obs_traj, rollout_idx, logdir):
    """
    Save the trajectory as a video
    """
    max_frames = max(
        [len(obs_traj[i]["agentview_image"]) for i in range(len(obs_traj))]
    )

    # Collect frames and pad with zeros if the length is less than max_frames
    frames = []
    for i in range(len(obs_traj)):
        frames.append(obs_traj[i]["agentview_image"])
        if len(frames[-1]) < max_frames:
            frames[-1] += [
                np.zeros_like(frames[-1][0])
                for _ in range(max_frames - len(frames[-1]))
            ]

    frames = np.array(frames)  # Shape (n_envs, time, frame_height, frame_width, 3)
    frames = einops.rearrange(
        frames,
        "n_envs n_imgs frame_height frame_width c -> n_imgs frame_height (n_envs frame_width) c",
    )
    savedir = logdir / "collected_traj_videos/"
    os.makedirs(savedir, exist_ok=True)
    out = cv2.VideoWriter(
        str(savedir / f"rollout_{rollout_idx}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (frames.shape[2], frames.shape[1]),
    )
    for t in range(frames.shape[0]):
        out.write(frames[t])
    out.release()


def collect_onpolicy_trajs(
    num_steps,
    max_traj_len,
    base_policy,
    train_env,
    pred_horizon,
    obs_horizon,
    train_eps,
    state_only,
    save_dir=None,
    save_episodes=False,
    discard_if_not_success=False,
):
    start_time = time.time()
    """
    Collect num_trajs trajectories using base_policy in train_env
    """
    if num_steps == 0:
        print("Collecting 0 steps.")
        return

    num_envs = train_env.num_envs
    print(f"Collecting {num_steps} steps, Num Envs: {num_envs}")

    obs = train_env.reset()
    if state_only:
        pixel_keys = []
    else:
        pixel_keys = sorted([key for key in obs.keys() if "image" in key])
    n_step_collected = 0
    eps_names = []
    while n_step_collected < num_steps:
        obs_traj = [
            {
                "state": [],
                "base_action": [],
                "residual_action": [],
                "action": [],
                "reward": [],
                "is_first": [],
                "is_last": [],
                "is_terminal": [],
                "success": False,
                **{key: [] for key in pixel_keys},
            }
            for _ in range(num_envs)
        ]  # List of independent dictionaries of data for each environment

        obs = train_env.reset()
        base_policy.reset()

        dones = np.zeros(num_envs, dtype=bool)

        # Collect data for obs_list
        print(f"Collected {n_step_collected}/{num_steps}, collecting more...")
        for step_idx in range(max_traj_len):
            if np.all(dones):
                break

            action_dict = base_policy.get_action(obs)
            action = np.clip(
                action_dict["base_action"] + action_dict["residual_action"], -1, 1
            )
            obs_next, rewards, dones, infos = train_env.step(action)

            # Add obs and action to obs_list
            add_env_obs_to_dict(
                obs=obs,
                obs_traj=obs_traj,
                base_action=action_dict["base_action"],
                residual_action=action_dict["residual_action"],
                action=action,
                rewards=rewards,
                dones=dones,
                pixel_keys=pixel_keys,
                step_idx=step_idx,
                max_traj_len=max_traj_len,
            )

            all_successes = infos["success"]  # num_envs x 1
            for env_idx in range(num_envs):
                obs_traj[env_idx]["success"] = (
                    all_successes[env_idx] or obs_traj[env_idx]["success"]
                )

            obs = obs_next

        # Uncomment to visualize the collected trajectory
        # save_collected_traj_video(obs_traj, rollout_idx=n_trajs_collected, logdir=pathlib.Path("."))
        # breakpoint()

        for env_idx in range(num_envs):
            if discard_if_not_success == True and not obs_traj[env_idx]["success"]:
                cprint(
                    f"Skipping adding Env IDX: {env_idx}, Traj Len: {len(obs_traj[env_idx]['state'])}, Total Reward: {np.sum(obs_traj[env_idx]['reward'])}",
                    "red",
                )
                continue

            eps_name = add_traj_to_cache(
                env_idx=env_idx,
                obs_traj=obs_traj[env_idx],
                pixel_keys=pixel_keys,
                pred_horizon=pred_horizon,
                obs_horizon=obs_horizon,
                train_eps=train_eps,
            )
            eps_names.append(eps_name)
            cprint(
                f"Added Env IDX: {env_idx}, Traj Len: {len(obs_traj[env_idx]['state'])}, Total Reward: {np.sum(obs_traj[env_idx]['reward'])}",
                "green",
            )
            n_step_collected += len(obs_traj[env_idx]["state"])

        if n_step_collected >= num_steps:
            break

    print(
        f"Time taken to collect {n_step_collected}/{num_steps} steps: {time.time() - start_time:.2f} seconds"
    )

    if save_dir is not None and save_episodes:
        print("Saving Episodes to Disk: ", eps_names, " at ", save_dir)
        save_episodes(
            directory=save_dir, episodes={name: train_eps[name] for name in eps_names}
        )
    return n_step_collected


def mixed_sample(
    batch_size,
    expert_dataset,
    train_dataset,
    device,
    remove_obs_stack=True,
    sqil_discriminator=False,
):
    """
    Sample 50% from expert dataset and 50% from self.train_eps
    If remove_obs_stack is True, keep only latest obs in the batch
    """
    assert batch_size % 2 == 0, "Batch Size should be even."

    expert_batch = next(expert_dataset)
    train_batch = next(train_dataset)

    # Merge the two batches
    data_batch = {}
    for key in expert_batch.keys():
        if key in train_batch.keys():
            expert_batch[key] = torch.tensor(expert_batch[key], dtype=torch.float32)
            train_batch[key] = torch.tensor(train_batch[key], dtype=torch.float32)
            data_batch[key] = torch.cat(
                [expert_batch[key], train_batch[key]], dim=0
            ).to(device)

    # # SQIL discriminator, +1 for expert, -1 for all other
    if sqil_discriminator:
        data_batch["reward"] = torch.cat(
            [
                torch.ones_like(expert_batch["reward"]),
                -torch.ones_like(train_batch["reward"]),
            ],
            dim=0,
        ).to(device)

    if remove_obs_stack:
        data_batch = select_latest_obs(data_batch)

    return data_batch


def select_latest_obs(obs: dict):
    # Removes the stacked observations, keeping only the latest one
    # Returns a copy of the observations with removed stacked dimensions
    obs_out = {}
    obs_out["state"] = copy.deepcopy(obs["state"][..., -1])
    if "agentview_image" in obs.keys():
        obs_out["agentview_image"] = copy.deepcopy(obs["agentview_image"][..., -1])
    if "robot0_eye_in_hand_image" in obs.keys():
        obs_out["robot0_eye_in_hand_image"] = copy.deepcopy(
            obs["robot0_eye_in_hand_image"][..., -1]
        )
    # Keep all other things same
    for key in obs.keys():
        if key not in obs_out.keys():
            obs_out[key] = obs[key]
    return obs_out


def add_traj_to_buffer(
    env_idx, obs_traj, pixel_keys, pred_horizon, obs_horizon, success_buffer, failure_buffer
):
    """
    Traj IDX: ID of teh collected trajectory
    obs_traj: Dictionary of observations collected
    pixel_keys: List of pixel keys in the observation
    """
    # Assign unique eps_name
    eps_name = f"traj_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{env_idx}"

    # Stack Observations for State and Pixel Keys
    stacked_obs = {}
    stacked_obs["state"] = get_obs_stacked(obs_traj["state"], obs_horizon)
    for key in pixel_keys:
        stacked_obs[key] = get_obs_stacked(obs_traj[key], obs_horizon)

    # Stack base and residual actions
    stacked_base_acts = get_act_stacked(obs_traj["base_action"], pred_horizon)
    stacked_residual_acts = get_act_stacked(obs_traj["residual_action"], pred_horizon)

    # Stack Actions
    stacked_actions = get_act_stacked(obs_traj["action"], pred_horizon)

    # Fill the transitions in self.train_eps
    for idx in range(len(obs_traj["state"])):
        transition = defaultdict(np.array)
        for key in stacked_obs.keys():
            transition[key] = stacked_obs[key][idx]

        transition["base_action"] = stacked_base_acts[idx]
        transition["residual_action"] = stacked_residual_acts[idx]
        transition["action"] = stacked_actions[idx]
        transition["reward"] = np.array(obs_traj["reward"][idx], dtype=np.float32)

        transition["is_first"] = np.array(obs_traj["is_first"][idx], dtype=np.bool_)
        transition["is_last"] = np.array(obs_traj["is_last"][idx], dtype=np.bool_)
        transition["is_terminal"] = np.array(
            obs_traj["is_terminal"][idx], dtype=np.bool_
        )
        
        if obs_traj["success"]:
            transition["success"] = transition["reward"] == 0.0 or transition["reward"] == 3.0
            add_to_cache(success_buffer, eps_name, transition)
        else:
            transition["success"] = np.array(0, dtype=np.int_)
            add_to_cache(failure_buffer, eps_name, transition)

    return eps_name

def stack_trajs(trajs, skip_keys):
    """
    Concatenate buffer trajectories key-by-key into one batch.
    trajs: iterable of trajectory dicts (as stored by add_traj_to_buffer)
    skip_keys: keys to leave out of the batch
    Returns: dict of arrays with shape (sum_T, 1, ...), ready for the preprocessor's d4r_* methods
    """
    merged = {}
    for traj in trajs:
        for key in traj.keys():
            if key in skip_keys:
                continue
            merged.setdefault(key, []).extend(traj[key])
    return {key: np.stack(vals)[:, np.newaxis, ...] for key, vals in merged.items()}
