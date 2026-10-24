import numpy as np
import torch
from termcolor import cprint

from far.classes.rollout_utils import select_latest_obs
from far.dreamer import tools


def make_mixed_dataset(
    expert_dataset, train_dataset, p_expert_data, device, remove_obs_stack=True
):
    while True:
        if np.random.rand() < p_expert_data:
            sample = next(expert_dataset)
        else:
            sample = next(train_dataset)

        # Put sample on device
        sample = {
            k: torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in sample.items()
        }

        if remove_obs_stack:
            sample = select_latest_obs(sample)

        yield sample


def make_retrain_dp_dataset(replay_buffer, expert_eps, config):
    dp_batch_size = config.dp["batch_size"]

    # Make the datasets
    replay_dataset = tools.make_dataset(
        replay_buffer, batch_length=1, batch_size=dp_batch_size
    )
    expert_dataset = tools.make_dataset(
        expert_eps,
        batch_length=1,
        batch_size=dp_batch_size,
    )

    p_expert_data = config.train_dp_mppi_params["dp_expert_buffer_ratio"]
    print("Made Retrain DP Dataset with dp_expert_buffer_ratio: ", p_expert_data)
    print("Number of Trajectories in Replay Buffer: ", len(replay_buffer.keys()))
    print("Number of Trajectories in Expert Buffer: ", len(expert_eps.keys()))

    # Make mixed dataset
    mixed_dataset = make_mixed_dataset(
        expert_dataset=expert_dataset,
        train_dataset=replay_dataset,
        p_expert_data=p_expert_data,
        device=config.device,
        remove_obs_stack=False,  # Keep stacked obs for DP training
    )
    return mixed_dataset


def count_n_transitions(eps):
    n_transitions = 0
    for key in eps.keys():
        n_transitions += len(eps[key]["action"])
    return n_transitions


def label_expert_eps(expert_eps, dreamer_class):
    """
    If logdir/base_labelled_expert_eps.pkl exists, load it in self.expert_eps
    Else iterate through the self.expert_eps,
        - add label "base_action": output from base policy get_action_direct
        - add label "total_action": output from base policy get_action [Residual = 0 initially]
    """
    cprint(
        "Labeling expert episodes",
        "yellow",
    )
    for key in expert_eps.keys():
        data_traj_i = expert_eps[key].copy()

        # Stack across the length of the trajectory, add dummy BL dimension
        for k, v in data_traj_i.items():
            data_traj_i[k] = np.stack(v)
            data_traj_i[k] = np.expand_dims(
                data_traj_i[k], axis=1
            )  # Add dummy BL dimension

        # Process in chunks of 64
        base_action_raw = []
        for i in range(
            0, len(data_traj_i["action"]), 64
        ):  # Process in chunks of 64 to avoid OOM
            # Get the slice of the data
            data_traj_i_slice = {
                k: v[i : i + 64] if i + 64 <= len(v) else v[i:]  # Handle last chunk
                for k, v in data_traj_i.items()
            }
            # Pass the slice through the base policy to get the base action
            base_action_raw_chunk = (
                dreamer_class._task_behavior.base_policy.get_action_direct(
                    data_traj_i_slice
                )
            )
            # Detach and convert to numpy, remove BL dimension
            base_action_raw_chunk = (
                base_action_raw_chunk.detach()
                .cpu()
                .numpy()
                .squeeze(1)  # Remove BL dimension
            )  # Shape (..., act_dim, pred_horizon)

            # Transpose to get the shape (..., act_dim, pred_horizon)
            base_action_raw_chunk = base_action_raw_chunk.transpose(0, -1, -2)

            # Append to the list
            base_action_raw.append(
                base_action_raw_chunk
            )  # Append the chunk to the list of base actions

        # Concatenate all chunks to get the final base action
        base_action_raw = np.concatenate(
            base_action_raw, axis=0
        )  # Shape (..., act_dim, pred_horizon)

        # Make into list and of order act_dim x pred_horizon
        base_action_raw_list = [
            base_action_raw[i] for i in range(base_action_raw.shape[0])
        ]

        # Convert to list across dim 0
        expert_eps[key][
            "base_action"
        ] = base_action_raw_list  # ... x act_dim x pred_horizon

        # Get residual_action = (action - base_action)
        residual_action_list = []
        for i in range(len(base_action_raw)):
            expert_action = expert_eps[key]["action"][i]  # act_dim x pred_horizon
            residual_action_i = (
                expert_action - base_action_raw[i]
            )  # expert_action[0] - base_action[0] = (act_dim,)
            residual_action_list.append(residual_action_i)

        expert_eps[key][
            "residual_action"
        ] = residual_action_list  # [(act_dim,)] x num_episodes
    print(f"Labelled {len(expert_eps.keys())} expert episodes.")
