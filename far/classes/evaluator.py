import os
import random
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import einops
import imageio
import numpy as np
import torch
import copy, json
from PIL import Image

from far.dreamer.tools import set_seed_everywhere
from far.classes.rollout_utils import cprint
from far.dreamer import tools
from far.classes.rollout_utils import add_env_obs_to_dict, add_traj_to_buffer, stack_trajs
import collections


# ============================================================
# Robot state save / restore (for backtracking)
# ============================================================

def _unwrap_robosuite_env(env, suite):
    """Return the underlying robosuite env (robocasa wraps it five levels deep)."""
    if suite == "robocasa":
        return env.env.env.env.env.env
    return env


def save_robot_state(env, suite):
    if suite == "maniskill":
        robot = env.env.env.unwrapped.agent
        return {
            "qpos": robot.robot.get_qpos().clone(),
            "qvel": robot.robot.get_qvel().clone(),
        }

    suite_env = _unwrap_robosuite_env(env, suite)
    robot = suite_env.robots[0]
    return {
        "qpos": suite_env.sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
        "qvel": suite_env.sim.data.qvel[robot._ref_joint_vel_indexes].copy(),
    }



def load_robot_state(env, state, suite):
    if suite == "maniskill":
        u_env = env.env.env.unwrapped
        # Step A: force-sync position and velocity in the physics engine
        u_env.agent.robot.set_qpos(state["qpos"])
        u_env.agent.robot.set_qvel(state["qvel"])

        # Step B: the critical step! Sync the controller's target position,
        # otherwise the controller will try to pull the robot back to its previous action target.
        u_env.agent.controller.reset()
        # u_env.agent.controller.set_drive_targets(current_qpos)
        return

    suite_env = _unwrap_robosuite_env(env, suite)
    robot = suite_env.robots[0]
    suite_env.sim.data.qpos[robot._ref_joint_pos_indexes] = state["qpos"]
    suite_env.sim.data.qvel[robot._ref_joint_vel_indexes] = state["qvel"]
    suite_env.sim.forward()

# ============================================================
# Episode bookkeeping
# ============================================================

@dataclass(frozen=True)
class EpisodeStats:
    """Running per-env episode metrics; every field has shape (num_envs,)."""
    successes: np.ndarray
    total_rewards: np.ndarray
    total_orig_rewards: np.ndarray
    length_episodes: np.ndarray
    dones: np.ndarray


def init_episode_stats(num_envs: int) -> EpisodeStats:
    return EpisodeStats(
        successes=np.zeros(num_envs, dtype=bool),
        total_rewards=np.zeros(num_envs),
        total_orig_rewards=np.zeros(num_envs),
        length_episodes=np.zeros(num_envs),
        dones=np.zeros(num_envs, dtype=bool),
    )


def update_episode_stats(stats: EpisodeStats, step_rewards, step_dones, step_infos) -> EpisodeStats:
    """Fold one env step into the running metrics (rewards only count while the env is not done)."""
    not_done = 1 - stats.dones
    total_orig_rewards = stats.total_orig_rewards
    if "orig_reward" in step_infos:
        total_orig_rewards = total_orig_rewards + not_done * step_infos["orig_reward"]
    # Succeeded currently or previously
    successes = np.logical_or(stats.successes, step_infos["success"])
    return EpisodeStats(
        successes=successes,
        total_rewards=stats.total_rewards + not_done * step_rewards,
        total_orig_rewards=total_orig_rewards,
        # Count steps until success
        length_episodes=stats.length_episodes + np.logical_not(successes) * 1,
        # Done currently or Succeeded
        dones=np.logical_or(step_dones, successes),
    )


class TTASeedResult(NamedTuple):
    """Metrics of one TTA seed. For the TTA_CL method most fields are per-eval lists."""
    success_rate: object
    recovery_rate: object
    total_avg_reward: object
    avg_length_episode: object
    total_avg_orig_reward: object
    success_trial: list
    run_times: list
    seed_images: list
    eef_poses: Optional[list]


def is_valid_tta_buffer(masks, buffer) -> bool:
    """TTA needs at least one selected step and no empty trajectory in the buffer."""
    return np.sum(masks) > 0 and all(len(traj["state"]) > 0 for traj in buffer.values())


def first_success_end_index(success_frames: np.ndarray, env_idx: int) -> int:
    """Index one past the first successful frame of env_idx, or the rollout length if it never succeeded."""
    where_arr = np.where(success_frames[:, env_idx] == True)[0]
    return where_arr[0] + 1 if len(where_arr) > 0 else len(success_frames)


# ============================================================
# Failure-step selection and positive-action mining for TTA
# ============================================================

def argmin_steps(value_diff: torch.Tensor, traj_len: int) -> np.ndarray:
    """Steps with the largest value drop: max(10, 10% of the trajectory) of them."""
    sample_n = max(10, int(traj_len * 0.1))
    return value_diff.argsort()[:sample_n].cpu().numpy()


def suppress_close_steps(steps: np.ndarray, min_gap: int) -> np.ndarray:
    """1D NMS: keep a step only if it is at least min_gap after the last kept one."""
    kept = []
    for s in steps:
        if len(kept) == 0 or s - kept[-1] >= min_gap:
            kept.append(s)
    return np.array(kept)


def chunk_gae_advantages(rewards, value, gamma, gae_lambda, ac_horizon) -> np.ndarray:
    """GAE over each action chunk of length ac_horizon starting at every t."""
    advantages = []
    for t in range(len(rewards) - ac_horizon):
        chunk_gae = 0
        # Back-to-front over t + ac_horizon - 1 ... t
        for k in reversed(range(ac_horizon)):
            curr_idx = t + k
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[curr_idx] + gamma * value[curr_idx + 1] - value[curr_idx]
            # A_t = delta_t + (gamma * lambda) * A_{t+1}
            chunk_gae = delta + (gamma * gae_lambda) * chunk_gae
        advantages.append(chunk_gae.cpu().numpy())
    return np.array(advantages)


def mean_chunk_distance(candidates: np.ndarray, ref_action: np.ndarray) -> np.ndarray:
    """candidates: (N, ac_chunk, ac_dim), ref_action: (ac_chunk, ac_dim) -> mean per-step L2 distance (N,)."""
    return np.mean(np.linalg.norm(candidates - ref_action, ord=2, axis=-1), axis=1)


def select_random_positives(candidates, pref_scores, v):
    """Pick up to 8 random candidates as positives. Returns ((k, ac_dim, ac_chunk), advantages (k,))."""
    top_k = min(8, len(pref_scores))
    idx = np.random.choice(len(pref_scores), size=top_k, replace=False)
    return candidates[idx].transpose(0, 2, 1), pref_scores[idx] - v


def select_distant_top_k_positives(candidates, pref_scores, neg_action, v, top_k):
    """
    Drop candidates that are too close to (or too far from) the failed action,
    then keep the top_k by critic score. Returns ((k, ac_dim, ac_chunk), advantages (k,)).
    """
    dists_to_fail = mean_chunk_distance(candidates, neg_action)
    # Keep candidates farther than average from the failure but within 1.0
    safe_mask = np.bitwise_and(dists_to_fail > np.mean(dists_to_fail), dists_to_fail < 1.0)
    safe_candidates = candidates[safe_mask]
    safe_scores = pref_scores[safe_mask]

    # Edge case: too few candidates survive the filter
    if len(safe_candidates) < 8:
        print("Warning: All samples are too close to the failed action!")
        safe_candidates = candidates
        safe_scores = pref_scores

    top_k = min(top_k, len(safe_scores))
    idx = np.argsort(safe_scores)[-top_k:][::-1]
    return safe_candidates[idx].transpose(0, 2, 1), safe_scores[idx] - v


def select_farthest_of_top_k(candidates, pref_scores, fail_action) -> np.ndarray:
    """Among the top 8 candidates by critic score, return the one farthest from the failure: (ac_dim, ac_chunk)."""
    pool = candidates[np.argsort(pref_scores)[-8:][::-1]]
    best_idx = np.argmax(mean_chunk_distance(pool, fail_action))
    return pool[best_idx].transpose(1, 0)

class ModelEvaluator:
    def __init__(
        self,
        agent,
        envs,
        default_seed,
        visualize=False,
        parent_output_dir="",
        eval_num_runs=10,
        step=0,
        MAX_EPISODE_LENGTH=2000,
        exp_retriever=None,
        max_retries=5,
        env_step_start=0,
    ):
        self.eval_num_runs = eval_num_runs
        self.default_seed = default_seed
        self.agent = agent # DiffusionBasePolicy
        self.envs = envs
        self.visualize = visualize
        self.parent_output_dir = parent_output_dir
        self.MAX_EPISODE_LENGTH = MAX_EPISODE_LENGTH
        self.step = step
        self.NUM_RUNS = int(np.ceil(eval_num_runs / envs.num_envs))
        self.retriever = exp_retriever
        self.max_retries = max_retries
        self.env_step_start = env_step_start
        self.env_step = 0

    def enter_seed(self, seed):
        # Store random states for torch and numpy
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        self.torch_cuda_state = torch.cuda.get_rng_state_all()
        # Set the seeds
        set_seed_everywhere(seed)

    def exit_seed(self):
        # Reset the seeds
        set_seed_everywhere(self.default_seed)

        # Restore random states for torch and numpy
        # (To continue same sequence outside the scope of enter_seed and exit_seed)
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state_all(self.torch_cuda_state)

    @property
    def _step_dir(self):
        return self.parent_output_dir / f"step_{self.step}"

    def _print_eval_header(self):
        print(
            f"Evaluating Agent num_eval_runs: {self.eval_num_runs}, num_envs: {self.envs.num_envs}, NUM_RUNS: {self.NUM_RUNS}"
        )

    def _add_traj(self, env_idx, traj, pixel_keys, buffer):
        """Chunk one collected trajectory and store it in buffer (success or failure). Returns its episode name."""
        return add_traj_to_buffer(
            env_idx=env_idx,
            obs_traj=traj,
            pixel_keys=pixel_keys,
            pred_horizon=self.agent.config.pred_horizon,
            obs_horizon=self.agent.config.obs_horizon,
            success_buffer=buffer,
            failure_buffer=buffer,
        )

    # Write meta ENV that operates on list of envs and can do things like step and all
    def evaluate_agent_seed(self, seed):
        self.enter_seed(seed)

        # Make Environment and Agent
        envs = self.envs
        num_envs = envs.num_envs

        # Run the Evaluation
        stats = init_episode_stats(num_envs)
        image_frames = []
        handcam_frames = []
        success_frames = []
        step_rewards_seed = {
            "gt_reward": [],
            "reward_output": [],
        }

        step_obs = envs.reset()
        self.agent.reset()

        print(f"EVALUATING {num_envs} ENVIRONMENTS on SEED {seed}")
        for step_id in range(self.MAX_EPISODE_LENGTH):
            if np.all(stats.dones):
                break

            action = self.agent.get_action(step_obs)
            step_obs, step_rewards, step_dones, step_infos = envs.step(action)
            stats = update_episode_stats(stats, step_rewards, step_dones, step_infos)

            if self.visualize:
                self._collect_visualization_frames(
                    step_obs, step_rewards, stats.successes,
                    image_frames, handcam_frames, success_frames, step_rewards_seed
                )

        # Exit the seed
        self.exit_seed()

        return (
            np.mean(stats.successes),
            np.mean(stats.total_rewards),
            np.mean(stats.length_episodes),
            np.mean(stats.total_orig_rewards),
            [image_frames, handcam_frames],
            success_frames,
            step_rewards_seed,
        )

    def evaluate_agent(self):
        start_time = time.time()
        success_rates = []
        total_avg_rewards = []
        total_avg_orig_rewards = []
        episode_lengths = []
        self._print_eval_header()
        for seed in range(self.NUM_RUNS):
            print("Seed:", seed)
            (
                success_rate,
                total_avg_reward,
                avg_length_episode,
                total_avg_orig_reward,
                seed_images,
                seed_success_frames,
                step_rewards_seed,
            ) = self.evaluate_agent_seed(seed)
            success_rates.append(success_rate)
            total_avg_rewards.append(total_avg_reward)
            episode_lengths.append(avg_length_episode)
            total_avg_orig_rewards.append(total_avg_orig_reward)
            print(f"\tAverage Success Rate: {success_rate}")
            print(f"\tAverage Total Reward: {total_avg_reward}")
            print(f"\tAverage Episode Length: {avg_length_episode}")
            print(f"\tAverage Total Orig Reward: {total_avg_orig_reward}")
            if self.visualize:
                os.makedirs(self._step_dir, exist_ok=True)

                # Save npz files of rollout if we are conducting a highres rollout to generate evaluation videos
                if "highres" in str(self.step):
                    self._save_rollout_npz(seed, seed_images, seed_success_frames, step_rewards_seed)
                else:
                    # Save videos of the rollouts
                    for i, image_frame in enumerate(seed_images):
                        video_path = f"{self.parent_output_dir}/step_{self.step}/seed_{seed}_cam_{i}_succ_{success_rate:.2f}_rew_{total_avg_reward:.2f}.mp4"
                        self.save_video(image_frame, seed_success_frames, video_path)

        avg_success_rate = np.mean(success_rates)
        avg_total_avg_reward = np.mean(total_avg_rewards)
        episode_length = np.mean(episode_lengths)
        avg_total_avg_orig_reward = np.mean(total_avg_orig_rewards)
        print(f"Average Success Rate: {avg_success_rate}")
        print(f"Average Total Average Reward: {avg_total_avg_reward}")
        print(f"Average Episode Length: {episode_length}")
        print(f"Average Total Average Orig Reward: {avg_total_avg_orig_reward}")
        print("Time taken:", time.time() - start_time)
        self.envs.reset()
        return (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_avg_orig_reward,
        )

    def _compute_value_and_pref_scores(self, imgs, obs, actions):
        pref_net = self.agent.pref_net
        v_net = self.agent.v_net
        value = v_net.get_value(imgs, obs).squeeze(-1)
        q1, q2 = pref_net.get_preference(imgs, obs, actions.reshape((actions.shape[0], -1)))
        pref_scores = torch.min(q1.squeeze(-1), q2.squeeze(-1))
        return value, pref_scores

    def _score_traj(self, traj):
        """
        Preprocess a buffer trajectory on GPU and score every step with V(s) and Q(s, a).
        Returns (imgs, obs, value, pref_scores), each indexed by time step.
        """
        train_data = stack_trajs([traj], skip_keys=())
        (imgs, obs), actions, _ = self.agent.preprocessor.d4r_preprocess_batch(train_data, training=False)
        if imgs is not None:
            imgs = {k: v.to("cuda") for k, v in imgs.items()}
        obs, actions = obs.to("cuda"), actions.to("cuda")
        value, pref_scores = self._compute_value_and_pref_scores(imgs, obs, actions)
        return imgs, obs, value, pref_scores

    def _select_tta_steps(self, fail_traj, value, value_diff, reset_step):
        ac_horizon = self.agent.config.pred_horizon
        steps_tta = argmin_steps(value_diff, len(value))
        return steps_tta, steps_tta.min()
    

    @torch.no_grad()
    def construct_tta_dataset_with_critic(self, buffer, step_tta, obs_list, action_list, action_history):
        reset_step = step_tta
        ac_horizon = self.agent.config.pred_horizon
        for traj_id in buffer.keys():
            # In practice there is currently only one key
            cprint(f"TTA on traj: {traj_id}", "yellow")
            fail_traj = buffer[traj_id]
            _, _, value, _ = self._score_traj(fail_traj)

            value_diff = value[ac_horizon:] - value[:-ac_horizon]
            if "maniskill" in self.agent.config.task:
                value_diff = -value_diff

            steps_tta, reset_step = self._select_tta_steps(fail_traj, value, value_diff, reset_step)

        for traj_id in buffer.keys():
            cprint(f"TTA on traj: {traj_id}", "yellow")
            self.agent.reset_tta()
            fail_traj = buffer[traj_id]
            num_sample = 64

            cprint(f"Selected fail index: {steps_tta}", "blue")
            # Keep only the selected failure steps
            for key, val in fail_traj.items():
                fail_traj[key] = np.array(val)[steps_tta]

            pos_actions = []
            advantages = []
            masks = []
            for t in range(len(fail_traj['state'])):
                step_t = steps_tta[t]
                step_obs = obs_list[step_t]
                step_obs_prev = obs_list[step_t - 1] if step_t > 0 else obs_list[0]
                obs = obs_list[step_t]['agentview_image'][0]
                obs1 = fail_traj["agentview_image"][t][:, :, :, -1]
                assert np.all(obs == obs1), "Observation mismatch!"
                fail_action = fail_traj["action"][t].transpose(1,0)

                self.agent.policy.process_obs(copy.deepcopy(step_obs_prev))
                actions, pref_scores = self.agent.sample_actions(step_obs, scale=1.0, weighting=False, get_full_action=True, num_actions=num_sample, return_pref=True) 

                pref_scores = pref_scores.squeeze().detach().cpu().numpy()  # (num_sample, )
                valid_candidates = np.clip(actions, -1, 1)
                v = value[step_t].item()

                # Pick positive action(s) among the sampled candidates
                neg_action = fail_action
                top_k = 8
                pos_action, advantage = select_distant_top_k_positives(valid_candidates, pref_scores, neg_action, v, top_k)
                pos_actions.append(pos_action)
                advantages.append(advantage)
                masks.append(True)

            fail_traj['pos_actions'] = np.stack(pos_actions)
            advantages = np.stack(advantages)
            if isinstance(advantages, torch.Tensor):
                advantages = advantages.cpu().numpy()
            fail_traj['advantages'] = advantages
        return masks, reset_step


    def _init_obs_traj(self, num_envs, pixel_keys):
        return [
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
        ]

    def _save_retry_failure_traj(self, eval_i, obs_traj, pixel_keys, tmp_buffer, id2failure, reset_step):
        eps_name = self._add_traj(eval_i, obs_traj[0], pixel_keys, tmp_buffer)
        id2failure[eval_i].append(eps_name)

    def _warmup_policy(self, obs_list, reset_step):
        """Reset the observation list and warm up the policy"""
        obs_list = obs_list[:reset_step]
        if len(obs_list) >= reset_step and reset_step > 0:
            try:
                self.agent.get_action(obs_list[reset_step - 1])
            except Exception as e:
                print(f"Policy warmup error: {e}, obs_list len={len(obs_list)}, reset_step={reset_step}")
        return []

    def _collect_visualization_frames(self, step_obs, step_rewards, successes,
                                    image_frames, handcam_frames, success_frames, step_rewards_seed):
        """Collect visualization frames"""
        image_key = "agentview_image_highres" if "agentview_image_highres" in step_obs else "agentview_image"
        eye_in_hand_key = "robot0_eye_in_hand_image_highres" if "robot0_eye_in_hand_image_highres" in step_obs else "robot0_eye_in_hand_image"

        # Convert tensor to numpy
        if isinstance(step_obs[image_key], torch.Tensor):
            step_obs[image_key] = step_obs[image_key].cpu().numpy()
        if isinstance(step_obs[eye_in_hand_key], torch.Tensor):
            step_obs[eye_in_hand_key] = step_obs[eye_in_hand_key].cpu().numpy()

        image_frames.append(step_obs[image_key])
        handcam_frames.append(step_obs[eye_in_hand_key])
        success_frames.append(successes.copy())

        step_rewards_seed["gt_reward"].append(step_rewards)
        if hasattr(self.agent, "reward_output"):
            step_rewards_seed["reward_output"].append(self.agent.reward_output)

    def _save_trajectories_to_buffers(self, obs_traj, successes, successes_recovery, num_envs, pixel_keys,
                                    recovery_buffer, success_buffer, failure_buffer,
                                    tmp_buffer, id2failure, eval_i):
        """Save trajectories to the corresponding buffers"""
        for env_idx in range(num_envs):
            traj = obs_traj[env_idx]

            # Check trajectory data alignment
            if not self._check_traj_alignment(traj, pixel_keys, env_idx):
                continue

            # Save trajectories that succeeded after recovery
            if recovery_buffer is not None and successes_recovery[env_idx]:
                print(f"[Success] Saving Recovery Trajectory for Env {env_idx}, Length: {len(traj['action'])}")
                self._add_traj(env_idx, traj, pixel_keys, recovery_buffer)

            # Save trajectories that succeeded on the first try
            if success_buffer is not None and successes[env_idx] and not successes_recovery[env_idx]:
                self._add_traj(env_idx, traj, pixel_keys, success_buffer)

            # Save failure trajectories
            if failure_buffer is not None:
                for paired_eps in id2failure[eval_i]:
                    failure_buffer[paired_eps] = tmp_buffer[paired_eps]

    def _check_traj_alignment(self, traj, pixel_keys, env_idx):
        """Check that trajectory data is aligned"""
        check_keys = ["state", "action", "reward"] + pixel_keys
        lengths = [len(traj[k]) for k in check_keys if k in traj]

        if len(set(lengths)) > 1:
            print(f"[Error] Env {env_idx} data alignment mismatch: {dict(zip(check_keys, lengths))}")
            return False

        actual_len = len(traj["action"])
        if actual_len == 0:
            print(f"[Warning] Env {env_idx} trajectory is empty, skipping...")
            return False

        return True

    def _add_exploration_noise(self, action, retry, step_id, eval):
        config = self.agent.config
        if not eval:
            noise_std = getattr(config, "noise_std", 0.1)
            epsilon = getattr(config, "noise_epsilon", 1.0)
            rand = np.random.rand()
            if retry == 0:
                action += np.random.normal(0, noise_std, size=action.shape)  # Base noise for initial exploration
            elif rand < epsilon: # Epsilon-greedy style noise injection
                action += np.random.normal(0, noise_std, size=action.shape)
        elif retry >= 1 and np.random.rand() < 0.5: # Epsilon-greedy style noise injection
            action += np.random.normal(0, getattr(config, "noise_std", 0.1), size=action.shape)

        return action

    def evaluate_agent_seed_far(self, seed, recovery_buffer=None, success_buffer=None, failure_buffer=None, pref_buffer=None, eval=True):
        if seed is not None:
            self.enter_seed(seed)

        n_eval_per_seed = 1

        # Save results
        seed_successes = np.zeros(n_eval_per_seed, dtype=bool)
        seed_successes_recovery = np.zeros(n_eval_per_seed, dtype=bool)
        seed_success_trial = np.full(n_eval_per_seed, -1, dtype=int)  # -1 means not successful
        seed_total_rewards = np.zeros(n_eval_per_seed)
        seed_total_orig_rewards = np.zeros(n_eval_per_seed)
        seed_length_episodes = np.zeros(n_eval_per_seed)
        seed_image_frames = []
        seed_handcam_frames = []
        seed_success_frames = []
        seed_step_rewards_seed = []
        seed_run_times = []

        agent_copy = copy.deepcopy(self.agent)
        tmp_buffer = collections.OrderedDict()  # temporary storage for failure trajectories
        id2failure = [[] for _ in range(n_eval_per_seed)]  # tracks the failure trajectory IDs for each eval
        eef_poses = [[] for _ in range(n_eval_per_seed)]


        self.visualize_buffer = collections.OrderedDict()  # trajectory storage for visualization
        self.failure_points = collections.OrderedDict()  # storage for failure-point trajectories
        for eval_i in range(n_eval_per_seed):
            print(f"EVALUATING {eval_i}/{n_eval_per_seed} ENVIRONMENTS on SEED {seed}")

            # Make Environment and Agent
            envs = self.envs
            num_envs = envs.num_envs
            self.agent = copy.deepcopy(agent_copy)

            # Run the Evaluation
            stats = init_episode_stats(num_envs)
            successes_recovery = np.zeros(num_envs, dtype=bool)
            run_times = [[] for _ in range(num_envs)]

            # Visualization-related
            image_frames = []
            handcam_frames = []
            success_frames = []
            step_rewards_seed = {"gt_reward": [], "reward_output": []}

            step_obs = envs.reset()
            self.agent.reset()
            pixel_keys = sorted([key for key in step_obs.keys() if "image" in key])
            # Collect the full trajectory first
            obs_traj = self._init_obs_traj(num_envs, pixel_keys)
            obs_list = []
            action_list = []
            action_history = []

            robot_states = [[] for _ in range(num_envs)]

            task = self.agent.config.task.split("_")[-1]
            suite = self.agent.config.task.split("_")[0]
            reset_step = self.agent.config.reset_step[task]
            max_retries = self.max_retries

            for retry in range(max_retries):
                print(f"Retry {retry}/{max_retries}")
                self.agent.reset_tta()
                # Step 1: collect state, action pairs
                buffer = collections.OrderedDict()

                # If this is a retry, first save the previous failure trajectory
                if retry > 0 and len(obs_traj[0]["state"]) > 0:
                    self._save_retry_failure_traj(
                        eval_i, obs_traj, pixel_keys, tmp_buffer, id2failure, reset_step
                    )
                    obs_traj = self._init_obs_traj(num_envs, pixel_keys)
                    obs_list = self._warmup_policy(obs_list, reset_step)
                    action_history = []

                # Robot states are recorded from scratch on the first attempt (every attempt with staged reset)
                if retry == 0:
                    robot_states = [[] for _ in range(num_envs)]

                max_steps = self.MAX_EPISODE_LENGTH
                print(f"TTA Retry {retry}/{max_retries}, Max Steps: {max_steps}")
                recovery_mode = retry > 0
                for step_id in range(max_steps):
                    if np.all(stats.dones):
                        break

                    # Save all robot states for backtracking
                    for env_idx in range(num_envs):
                        robot_states[env_idx].append(save_robot_state(envs.envs[env_idx], suite))

                    action_kwargs = {}
                    action, original_action = self.agent.get_action(step_obs, get_original_action=True, recovery_mode=recovery_mode, **action_kwargs)

                    eef_pose = step_obs['state'][:, :3]
                    eef_poses[eval_i].append(eef_pose)

                    action = self._add_exploration_noise(action, retry, step_id, eval)

                    obs_list.append(step_obs)
                    action_list.append(action)
                    action_history.append(original_action)
                    # Note: must store step_obs, not next_obs, or the timing will be misaligned
                    obs_next, step_rewards, step_dones, step_infos = envs.step(action)
                    self.env_step += num_envs

                    add_env_obs_to_dict(
                        obs=step_obs,
                        obs_traj=obs_traj,
                        base_action=action,
                        residual_action=np.zeros_like(action),
                        action=action,
                        rewards=step_rewards,
                        dones=step_dones,
                        pixel_keys=pixel_keys,
                        step_idx=step_id,
                        max_traj_len=self.MAX_EPISODE_LENGTH,
                    )
                    step_obs = obs_next
                    
                    all_successes = step_infos["success"]  # num_envs x 1
                    for env_idx in range(num_envs):
                        obs_traj[env_idx]["success"] = (
                            all_successes[env_idx] or obs_traj[env_idx]["success"]
                        )

                    stats = update_episode_stats(stats, step_rewards, step_dones, step_infos)
                    if retry >= 1 and step_id >= 5:
                        successes_recovery = np.logical_or(successes_recovery, step_infos["success"])

                    if self.visualize:
                        self._collect_visualization_frames(
                            step_obs, step_rewards, stats.successes,
                            image_frames, handcam_frames, success_frames, step_rewards_seed
                        )

                if np.all(stats.successes):
                    print("All environments succeeded, no TTA needed.")
                    seed_success_trial[eval_i] = retry  # record the retry count at success
                    break

                # Step 2: find failure cases and construct test-time adaptation buffer
                for env_idx in range(num_envs):
                    if not stats.successes[env_idx] and len(obs_traj[env_idx]['state']) > 0:
                        self._add_traj(env_idx, obs_traj[env_idx], pixel_keys, buffer)

                # Step 3: Test-time adaptation using contrastive learning
                start_time = time.perf_counter()

                try:
                    masks, reset_step = self.construct_tta_dataset_with_critic(buffer, reset_step, obs_list=obs_list, action_list=action_list, action_history=action_history)
                    reset_step = 1  # if not using staged reset, just start from step 1

                except Exception as e:
                    cprint(f"Error constructing TTA dataset: {e}", "red")
                    continue

                if not is_valid_tta_buffer(masks, buffer):
                    cprint("No valid data for TTA", "red")
                    continue

                if retry <= 5:
                    self.agent.tta_update_dpo(
                        train_dataset=buffer,
                        exp_dataset=None,
                        expert_val_eps=None,
                        init_step=0,
                        train_steps=self.agent.config.tta["update_steps"],
                        lr=self.agent.config.tta["lr"]
                    )

                run_time = time.perf_counter() - start_time
                print(f"Run time: {run_time:.2f} seconds")
                # Rewind failed envs to reset_step
                for env_idx in range(num_envs):
                    run_times[env_idx].append(run_time)
                    if not stats.successes[env_idx]:
                        load_robot_state(envs.envs[env_idx], robot_states[env_idx][reset_step], suite)
                # IMPORTANT: take a no-op step after loading state to get current observation correct
                action = np.zeros_like(action)
                step_obs, step_rewards, step_dones, step_infos = envs.step(action)

            seed_successes[eval_i] = stats.successes[0]
            seed_successes_recovery[eval_i] = successes_recovery[0]
            seed_total_rewards[eval_i] = stats.total_rewards[0]
            seed_total_orig_rewards[eval_i] = stats.total_orig_rewards[0]
            seed_length_episodes[eval_i] = stats.length_episodes[0]
            seed_image_frames.append(image_frames)
            seed_handcam_frames.append(handcam_frames)
            seed_success_frames.append(success_frames)
            seed_step_rewards_seed.append(step_rewards_seed)
            seed_run_times.append(run_times[0])

            # ========== Save trajectories to buffers ==========
            self._save_trajectories_to_buffers(
                obs_traj, stats.successes, successes_recovery, num_envs, pixel_keys,
                recovery_buffer, success_buffer, failure_buffer,
                tmp_buffer, id2failure, eval_i
            )
        self.agent = agent_copy
        if seed is not None:
            # Exit the seed
            self.exit_seed()

        return (
            seed_successes.tolist(),
            seed_total_rewards.tolist(),
            seed_length_episodes.tolist(),
            seed_total_orig_rewards.tolist(),
            [seed_image_frames, seed_handcam_frames],
            seed_success_frames,
            seed_step_rewards_seed,
            action_list,
            stats.successes,
            seed_successes_recovery.tolist(),
            eef_poses,
            seed_success_trial.tolist(),
            seed_run_times
        )

    def _run_tta_seed(self, seed, rng_seed, recovery_buffer, success_buffer, failure_buffer, eval) -> TTASeedResult:
        """
        Run one TTA seed with the method selected by TTA_CL (CL retries vs. preference-guided retries)
        and save its videos when self.visualize is set.
        seed: label used in output file names; rng_seed: seed to run under (None keeps the current RNG stream).
        """
        (
            success_rate,
            total_avg_reward,
            avg_length_episode,
            total_avg_orig_reward,
            seed_images,
            seed_success_frames,
            step_rewards_seed,
            _,
            _,
            recovery_rate,
            eef_poses,
            seed_success_trial,
            seed_run_times
        ) = self.evaluate_agent_seed_far(
            rng_seed, recovery_buffer=recovery_buffer, success_buffer=success_buffer, failure_buffer=failure_buffer, eval=eval
        )
        if self.visualize:
            for eval_i in range(len(success_rate)):
                images = [seed_images[0][eval_i], seed_images[1][eval_i]]
                self.save_visualization(seed, success_rate=success_rate[eval_i], total_avg_reward=total_avg_reward[eval_i], seed_images=images, seed_success_frames=seed_success_frames[eval_i], step_rewards_seed=step_rewards_seed[eval_i], recovery_rate=recovery_rate[eval_i], eval_i=eval_i)
    
        return TTASeedResult(
            success_rate=success_rate,
            recovery_rate=recovery_rate,
            total_avg_reward=total_avg_reward,
            avg_length_episode=avg_length_episode,
            total_avg_orig_reward=total_avg_orig_reward,
            success_trial=seed_success_trial,
            run_times=seed_run_times,
            seed_images=seed_images,
            eef_poses=eef_poses,
        )

    def _report_tta_results(self, results: List[TTASeedResult], start_time):
        """Print averaged TTA metrics and write them to <step_dir>/results.json. Returns the averages."""
        success_rates = [r.success_rate for r in results]
        recovery_rates = [r.recovery_rate for r in results]
        total_avg_rewards = [r.total_avg_reward for r in results]
        # NOTE: the two lists below are swapped relative to their names (kept from the original code)
        total_avg_orig_rewards = [r.avg_length_episode for r in results]
        episode_lengths = [r.total_avg_orig_reward for r in results]

        avg_success_rate = np.mean(success_rates)
        avg_recovery_rate = np.mean(recovery_rates)
        avg_total_avg_reward = np.mean(total_avg_rewards)
        episode_length = np.mean(episode_lengths)
        avg_total_avg_orig_reward = np.mean(total_avg_orig_rewards)
        print(f"Average Success Rate: {avg_success_rate}")
        print(f"Average Recovery Rate: {avg_recovery_rate}")
        print(f"Average Total Average Reward: {avg_total_avg_reward}")
        print(f"Average Episode Length: {episode_length}")
        print(f"Average Total Average Orig Reward: {avg_total_avg_orig_reward}")
        print("Time taken:", time.time() - start_time)
        with open(self._step_dir / "results.json", "w") as f:
            json.dump({
                "env_step_start": self.env_step_start,
                "env_step": self.env_step,
                "avg_success_rate": avg_success_rate,
                "avg_recovery_rate": avg_recovery_rate,
                "avg_total_avg_reward": avg_total_avg_reward,
                "episode_length": episode_length,
                "avg_total_avg_orig_reward": avg_total_avg_orig_reward,
                "success_rates": success_rates,
                "recovery_rates": recovery_rates,
                "total_avg_rewards": total_avg_rewards,
                "total_avg_orig_rewards": total_avg_orig_rewards,
                "episode_lengths": episode_lengths,
                "success_trials": [r.success_trial for r in results],
                "run_times": [r.run_times for r in results],
            }, f, indent=4)
        return avg_success_rate, avg_total_avg_reward, episode_length, avg_total_avg_orig_reward

    def evaluate_agent_tta(self):
        """Evaluate with TTA on fixed seeds (no data collection)."""
        start_time = time.time()
        self._print_eval_header()

        eval_seeds = [0, 1, 2, 3, 4] if "DEBUG" in os.environ else list(range(50))
        results = []
        for seed in eval_seeds:
            print("Seed:", seed)
            print("Evaluating with TTA...")
            result = self._run_tta_seed(
                seed, rng_seed=seed, recovery_buffer=None, success_buffer=None, failure_buffer=None, eval=True
            )
            print(f"\tAverage Success Rate: {result.success_rate}")
            results.append(result)

        avg_success_rate, avg_total_avg_reward, episode_length, avg_total_avg_orig_reward = self._report_tta_results(results, start_time)
        self.envs.reset()
        env_step_train = self.env_step_start
        return (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_avg_orig_reward,
            env_step_train
        )

    def rollout_agent_tta(self, recovery_buffer=None, success_buffer=None, failure_buffer=None):
        """Collect on-policy TTA rollouts (with exploration noise) into the given buffers."""
        start_time = time.time()
        self._print_eval_header()

        rollout_seeds = list(range(20))
        results = []
        for seed in rollout_seeds:
            print("Seed:", seed)
            print("Rollout with TTA...")
            result = self._run_tta_seed(
                seed, rng_seed=None, recovery_buffer=recovery_buffer, success_buffer=success_buffer, failure_buffer=failure_buffer, eval=False
            )
            print(f"\tAverage Success Rate: {result.success_rate}")
            results.append(result)

        avg_success_rate, avg_total_avg_reward, episode_length, avg_total_avg_orig_reward = self._report_tta_results(results, start_time)
        env_step_train = self.env_step_start + self.env_step
        print("Env step for training:", env_step_train)
        self.envs.reset()
        return (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_avg_orig_reward,
            env_step_train
        )

    def save_video(self, frames, frame_successes, video_path):
        frames = np.array(frames)  # Shape (time, n_envs, frame_height, frame_width, 3)
        n_envs = frames.shape[1]
        frame_successes = np.array(frame_successes)  # Shape (time, n_envs)
        frame_successes = np.tile(
            frame_successes[..., None, None], (1, 1, frames.shape[2], frames.shape[3])
        )  # Shape (time, n_envs, frame_height, frame_width)

        # If success, bump up greenness in image
        frames[..., 1] = np.clip(
            np.array(frames[..., 1], dtype=np.float32)
            + np.array(frame_successes, dtype=np.float32) * 50,
            0,
            255,
        )

        # Stack in columns of 5
        n_cols = min(5, n_envs)
        n_rows = int(np.ceil(n_envs / n_cols))
        n_pad = (n_rows * n_cols) - n_envs
        if n_pad > 0:
            zero_frames = np.zeros(
                (
                    frames.shape[0],
                    n_pad,
                    frames.shape[2],
                    frames.shape[3],
                    frames.shape[4],
                )
            )
            frames = np.concatenate([frames, zero_frames], axis=1)

        # Rearrange to shape n_imgs x (n_rows * frame_height) x (n_cols * frame_width) x c
        frames = einops.rearrange(
            frames,
            "n_imgs (n_rows n_cols) frame_height frame_width c -> n_imgs (n_rows frame_height) (n_cols frame_width) c",
            n_rows=n_rows,
            n_cols=n_cols,
        )
        frames = np.uint8(frames)

        # Use imageio to save video
        imageio.mimwrite(
            video_path, frames, fps=30, codec="libx264", format="mp4", quality=10
        )

        print("Saved video to", video_path)

    def _save_rollout_npz(self, seed, seed_images, seed_success_frames, step_rewards_seed):
        """Save each env's rollout (cam_0 frames, rewards, success) up to its first success as an .npz file."""
        success_frames = np.array(seed_success_frames)
        for i in range(self.envs.num_envs):
            end_index = first_success_end_index(success_frames, i)
            unit_eval_data = {
                "cam_0": np.array(seed_images[0])[:end_index, i, ...],
                "gt_reward": np.array(step_rewards_seed["gt_reward"])[:end_index, i],
                "reward_output": np.array(step_rewards_seed["reward_output"])[:end_index, i],
                "success": success_frames[:end_index, i],
            }
            npz_path = f"{self.parent_output_dir}/step_{self.step}/seed_{seed}_env_{i}.npz"
            np.savez(npz_path, data=unit_eval_data, allow_pickle=True)
            print("Saved all_data_dict to", npz_path)

    def save_visualization(self, seed, success_rate, total_avg_reward, seed_images, seed_success_frames, step_rewards_seed, recovery_rate, eval_i=None):
        os.makedirs(self._step_dir, exist_ok=True)

        # Save npz files of rollout if we are conducting a highres rollout to generate evaluation videos
        if "highres" in str(self.step):
            self._save_rollout_npz(seed, seed_images, seed_success_frames, step_rewards_seed)
        else:
            # Save videos of the rollouts
            eval_tag = f"_eval_{eval_i}" if eval_i is not None else ""
            for i, image_frame in enumerate(seed_images):
                video_path = f"{self.parent_output_dir}/step_{self.step}/seed_{seed}{eval_tag}_cam_{i}_succ_{success_rate:.2f}_rec_{recovery_rate}_rew_{total_avg_reward:.2f}.mp4"
                self.save_video(image_frame, seed_success_frames, video_path)

    @torch.no_grad()
    def failure_detection(self, traj):
        _, _, value, _ = self._score_traj(traj)

        ac_horizon = self.agent.config.pred_horizon
        value_diff = value[ac_horizon:] - value[:-ac_horizon]

        steps_tta = argmin_steps(value_diff, len(value))

        return steps_tta, value_diff
